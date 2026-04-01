"""
LLM API Client Module

Supports vLLM Automatic Prefix Caching (APC) optimization:
- Tracks token usage per request (prompt / completion / cached)
- Accumulates prefix cache savings statistics
- Passes cache control parameters to hint server-side prefix caching
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import requests

from .config import APIConfig

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single request"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0          # Tokens that hit prefix cache
    
    @property
    def cache_hit_rate(self) -> float:
        """Prefix cache hit rate"""
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens


@dataclass
class TokenStats:
    """Cumulative token usage statistics"""
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    per_request: List[TokenUsage] = field(default_factory=list)
    
    def add(self, usage: TokenUsage):
        self.total_requests += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_cached_tokens += usage.cached_tokens
        self.per_request.append(usage)
    
    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
    
    @property
    def overall_cache_hit_rate(self) -> float:
        """Overall prefix cache hit rate"""
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_prompt_tokens
    
    @property
    def estimated_savings_ratio(self) -> float:
        """
        Estimated cost savings ratio.
        Cached tokens are typically billed at 0.25x (based on Gemini context caching),
        savings = cached_tokens * 0.75 / total_prompt_tokens
        """
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.total_cached_tokens * 0.75 / self.total_prompt_tokens
    
    def summary(self) -> str:
        """Generate statistics summary"""
        lines = [
            f"📊 Token Usage Stats ({self.total_requests} requests)",
            f"   Prompt tokens:     {self.total_prompt_tokens:,}",
            f"   Completion tokens:  {self.total_completion_tokens:,}",
            f"   Total tokens:       {self.total_tokens:,}",
            f"   Cached tokens:      {self.total_cached_tokens:,}",
            f"   Cache hit rate:     {self.overall_cache_hit_rate:.1%}",
            f"   Est. cost savings:  {self.estimated_savings_ratio:.1%}",
        ]
        if self.per_request:
            lines.append("   ────────────────────────────────")
            for i, usage in enumerate(self.per_request, 1):
                cached_info = f", cached={usage.cached_tokens}" if usage.cached_tokens > 0 else ""
                hit_info = f", hit={usage.cache_hit_rate:.0%}" if usage.cached_tokens > 0 else ""
                lines.append(
                    f"   #{i}: prompt={usage.prompt_tokens}, "
                    f"completion={usage.completion_tokens}"
                    f"{cached_info}{hit_info}"
                )
        return "\n".join(lines)


class LLMClient:
    """LLM API Client (multi-turn conversation mode with prefix cache tracking)"""
    
    # ---------------------------------------------------------------
    # Model family detection rules
    # Each entry: (family_name, list_of_prefix_keywords)
    # The first match wins; order matters.
    # ---------------------------------------------------------------
    MODEL_FAMILIES = [
        ("gpt",     ["gpt-5", "gpt-4", "gpt-3", "o1", "o3", "o4"]),
        ("claude",  ["claude"]),
        ("gemini",  ["gemini"]),
    ]
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.last_exception = None
        self.token_stats = TokenStats()
        self._model_family = self._detect_model_family()
        logger.info(f"Model family detected: {self._model_family} (model={config.model_name})")
    
    # ==================================================================
    # Model-family detection
    # ==================================================================
    
    def _detect_model_family(self) -> str:
        """
        Detect model family from model_name.
        
        Returns one of: "gpt", "claude", "gemini", "default"
        """
        name = self.config.model_name.lower()
        for family, prefixes in self.MODEL_FAMILIES:
            for prefix in prefixes:
                if name.startswith(prefix):
                    return family
        return "default"
    
    # ==================================================================
    # Build request params per model family
    # ==================================================================
    
    def _build_params(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: float,
        presence_penalty: float,
        stop: Optional[str],
        n: int,
    ) -> dict:
        """
        Build model-specific request params dict.
        
        Different LLM APIs use different parameter names:
        - GPT-5.x / OpenAI Responses API: input + max_output_tokens
        - Gemini / OpenAI Chat Completions: messages + max_completion_tokens
        - Claude: messages + max_tokens
        - Default fallback: messages + max_tokens
        """
        family = self._model_family
        
        if family == "gpt":
            # GPT-5.x / OpenAI Responses API style
            params = {
                "input": messages,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        elif family == "claude":
            # Claude / Anthropic style
            params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop_sequences": [stop] if stop else None,
            }
        elif family == "gemini":
            # Gemini / OpenAI Chat Completions style
            params = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stream_options": {"include_usage": True},
            }
        else:
            # Default: OpenAI Chat Completions compatible
            params = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stream_options": {"include_usage": True},
            }
        
        # Remove None values to keep the request clean
        params = {k: v for k, v in params.items() if v is not None}
        
        logger.debug(f"Built params for family={family}: keys={list(params.keys())}")
        return params
    
    # ==================================================================
    # Parse response per model family
    # ==================================================================
    
    def _parse_response(self, model_output: dict) -> str:
        """
        Extract text content from model output, adapting to different response formats.
        
        - GPT-5.x Responses API: output[*].content[*].text
        - Gemini / OpenAI Chat Completions: choices[0].message.content
        - Claude: content[0].text
        - Default: choices[0].message.content
        """
        family = self._model_family
        
        if family == "gpt":
            # Try GPT-5.x Responses API format first
            output_list = model_output.get("output")
            if isinstance(output_list, list):
                for item in output_list:
                    if item.get("type") == "message":
                        content_list = item.get("content", [])
                        texts = [c.get("text", "") for c in content_list if c.get("type") == "output_text"]
                        if texts:
                            return "".join(texts)
            # Fallback to standard Chat Completions format
            return model_output["choices"][0]["message"]["content"]
        
        elif family == "claude":
            # Claude format: content[0].text
            content_list = model_output.get("content")
            if isinstance(content_list, list) and content_list:
                texts = [c.get("text", "") for c in content_list if c.get("type") == "text"]
                if texts:
                    return "".join(texts)
            # Fallback
            return model_output["choices"][0]["message"]["content"]
        
        else:
            # Gemini / default: standard Chat Completions
            return model_output["choices"][0]["message"]["content"]
    
    def _parse_token_usage(self, model_output: dict) -> TokenUsage:
        """
        Extract token usage from model output.
        
        Compatible with multiple formats:
        - OpenAI Chat Completions: usage.prompt_tokens / completion_tokens
        - GPT-5.x Responses API:   usage.input_tokens / output_tokens
        - vLLM extension: usage.prompt_tokens_details.cached_tokens
        - GPT-5.x cached: usage.input_tokens_details.cached_tokens
        - Gemini mapping: usage.cached_content_token_count
        """
        usage_data = model_output.get("usage", {})
        if not usage_data:
            return TokenUsage()
        
        # Support both naming conventions:
        #   Chat Completions: prompt_tokens / completion_tokens
        #   Responses API:    input_tokens  / output_tokens
        prompt_tokens = usage_data.get("prompt_tokens", 0) or usage_data.get("input_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0) or usage_data.get("output_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Try to extract cached tokens (multiple possible field names)
        cached_tokens = 0
        
        # vLLM / Chat Completions: prompt_tokens_details.cached_tokens
        prompt_details = usage_data.get("prompt_tokens_details", {})
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens", 0)
        
        # GPT-5.x Responses API: input_tokens_details.cached_tokens
        if cached_tokens == 0:
            input_details = usage_data.get("input_tokens_details", {})
            if isinstance(input_details, dict):
                cached_tokens = input_details.get("cached_tokens", 0)
        
        # OpenAI new format
        if cached_tokens == 0:
            cached_tokens = usage_data.get("prompt_cache_hit_tokens", 0)
        
        # Gemini format
        if cached_tokens == 0:
            cached_tokens = usage_data.get("cached_content_token_count", 0)
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )
    
    def call_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[str] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None
    ) -> str:
        """
        Call LLM API (multi-turn conversation)
        
        Args:
            messages: OpenAI-compatible message list, e.g.:
                [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "user", "content": "<tool_result>...</tool_result>"},
                ]
            max_retries: Maximum retry count
            temperature: Temperature parameter
            max_tokens: Maximum output tokens
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop token
            top_p: Top-p sampling
            n: Number of responses
        
        Returns:
            Text content returned by the LLM
        
        Raises:
            Exception: All retries failed
        """
        max_retries = self.config.max_retries if max_retries is None else max_retries
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        frequency_penalty = self.config.frequency_penalty if frequency_penalty is None else frequency_penalty
        presence_penalty = self.config.presence_penalty if presence_penalty is None else presence_penalty
        top_p = self.config.top_p if top_p is None else top_p
        n = self.config.n if n is None else n
        stop = self.config.stop if stop is None else stop
        stop = None if stop == "" else stop
        
        # Build model-specific request params
        params = self._build_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
        )
        
        data = {
            "sec_info": self.config.sec_info,
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "params": json.dumps(params),
        }
        
        response = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"LLM request attempt {attempt + 1}/{max_retries + 1}")
                logger.info(f"📡 Connecting: {self.config.url} (family={self._model_family})")
                
                response = requests.post(
                    url=self.config.url,
                    headers=self.config.headers,
                    json=data,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                # Double-layer JSON parsing
                response_json = json.loads(response.text)
                model_output = json.loads(response_json['model_output'])
                content = self._parse_response(model_output)
                
                # Extract and record token usage
                token_usage = self._parse_token_usage(model_output)
                self.token_stats.add(token_usage)
                
                logger.info(
                    f"LLM request succeeded, returned {len(content)} chars | "
                    f"tokens: prompt={token_usage.prompt_tokens}, "
                    f"completion={token_usage.completion_tokens}, "
                    f"cached={token_usage.cached_tokens} "
                    f"(hit={token_usage.cache_hit_rate:.0%})"
                )
                return content
            
            except requests.exceptions.Timeout:
                logger.warning(f"Request attempt {attempt + 1} timed out")
                self.last_exception = "Request timeout"
                
                if attempt == 0:
                    logger.info("💡 Network diagnosis: check firewall/API server/network connection")
            
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {str(e)}")
                self.last_exception = str(e)
                
                if attempt == 0:
                    logger.info(f"💡 Connection error: target {self.config.url}")
            
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(f"Response parse error on attempt {attempt + 1}: {str(e)}")
                self.last_exception = str(e)
                if response and response.text:
                    logger.debug(f"Response content: {response.text[:200]}")
            
            except Exception as e:
                logger.warning(f"Unknown error on attempt {attempt + 1}: {str(e)}")
                self.last_exception = str(e)
            
            # Exponential backoff
            if attempt < max_retries:
                wait_time = min(2 ** attempt, 32)
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # All retries failed
        error_msg = f"LLM request failed after {max_retries + 1} attempts"
        if self.last_exception:
            error_msg += f", last error: {self.last_exception}"
        
        logger.error(error_msg)
        raise Exception(error_msg)
