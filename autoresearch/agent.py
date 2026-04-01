"""
Universal AI Autonomous Research Agent

Pure Agentic Loop architecture:
1. Automatically run baseline to get initial score
2. Enter LLM multi-turn tool calling loop (ReAct mode)
3. LLM submits incremental edits → system trains and evaluates → score-driven update → feedback to LLM → loop
4. LLM calls FINAL_ANSWER or exhausts tool calls to end
"""

import re
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .config import APIConfig
from .llm_client import LLMClient
from .evaluator import BaseEvaluator, DefaultEvaluator
from .evaluator_loader import load_evaluator
from .experiment import ExperimentManager
from .trainer import Trainer
from .prompt_builder import PromptBuilder
from .code_editor import CodeEditor, EditBlock, EditResult

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

# Default maximum tool call count
DEFAULT_MAX_TOOL_CALLS = 5

# Training output truncation length (prevent overly long stdout from overflowing context)
MAX_OUTPUT_LENGTH = 500

# Error output truncation length (stderr, only included when training fails)
MAX_STDERR_LENGTH = 500


class UniversalAutoResearchAgent:
    """
    Universal AI Autonomous Research Agent (Pure Agentic Loop Architecture)
    
    The entire research process is a continuous agent loop:
    1. Automatically run baseline (no LLM involved)
    2. Build initial prompt (with baseline code + results) → call LLM
    3. LLM returns analysis + tool call (run_training / FINAL_ANSWER)
    4. If run_training → apply edits → train → score-driven update → append result to context → call LLM again
    5. If FINAL_ANSWER → apply edits → final training → end
    6. Force end when max tool calls reached
    
    Algorithm directory structure:
        algorithms/{name}/
        ├── train.py        # Training code (required)
        ├── README.md       # Algorithm description (recommended, passed to LLM)
        └── evaluator.py    # Evaluator (optional, uses default evaluator otherwise)
    
    Usage example:
        ```python
        from autoresearch import UniversalAutoResearchAgent
        
        agent = UniversalAutoResearchAgent(
            algorithm_dir="algorithms/nanochat",
            max_tool_calls=5,
        )
        agent.run("algorithms/nanochat/train.py")
        ```
    """
    
    def __init__(self, algorithm_dir: str, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS, model_name: str = None):
        """
        Initialize agent
        
        Args:
            algorithm_dir: Algorithm directory path containing train.py, README.md, and optional evaluator.py
            max_tool_calls: Maximum tool calls in agent loop (includes all rounds after baseline)
            model_name: LLM model name (e.g., gpt-5, gemini-3-pro-preview), uses config default when None
        """
        # API config and LLM client
        self.api_config = APIConfig()
        if model_name:
            self.api_config.model_name = model_name
        self.api_config.validate()
        self.llm_client = LLMClient(self.api_config)
        
        # Algorithm directory
        self.algorithm_dir = Path(algorithm_dir)
        self.algorithm_name = self.algorithm_dir.name
        
        # Load README.md
        self.readme_content = self._load_readme()
        
        # Load evaluator
        self.evaluator = load_evaluator(self.algorithm_dir)
        
        # Initialize modules
        # Experiment directory isolated by algorithm/model_name for easy LLM comparison
        llm_label = self.api_config.model_name
        experiments_base_dir = Path("experiments") / self.algorithm_name / llm_label
        self.experiment_manager = ExperimentManager(experiments_base_dir, self.algorithm_name)
        self.trainer = Trainer(self.evaluator)
        self.prompt_builder = PromptBuilder(
            max_code_length=self.api_config.max_code_length,
        )
        
        # Agent loop configuration
        self.max_tool_calls = max_tool_calls
        
        # Score-driven code state management
        self.current_code: Optional[str] = None   # Current best code
        self.best_score: float = self.evaluator.worst_score  # Current best score
        
        # Experiment counter (for experiment directory numbering)
        self._experiment_counter: int = 0
    
    # ==================================================================
    # Infrastructure
    # ==================================================================
    
    def _load_readme(self) -> Optional[str]:
        """Load README.md from algorithm directory"""
        readme_path = self.algorithm_dir / "README.md"
        if not readme_path.exists():
            logger.info(f"README.md not found: {readme_path}")
            return None
        
        try:
            content = readme_path.read_text(encoding='utf-8')
            logger.info(f"Loaded README.md: {readme_path}")
            return content
        except Exception as e:
            logger.warning(f"Failed to read README.md: {e}")
            return None
    
    def _next_experiment_id(self) -> int:
        """Get next experiment ID"""
        self._experiment_counter += 1
        return self._experiment_counter
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM API (multi-turn conversation)"""
        try:
            return self.llm_client.call_with_messages(
                messages=messages,
                temperature=self.api_config.temperature,
                max_tokens=self.api_config.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ""
    
    # ==================================================================
    # Tool Parsing
    # ==================================================================
    
    @staticmethod
    def _parse_tool_call(llm_response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse tool call from LLM response
        
        Supported format:
            <tool_call>
            <tool_name>run_training</tool_name>
            <edits>
            <edit>
            <search>...</search>
            <replace>...</replace>
            </edit>
            </edits>
            </tool_call>
        
        Fault tolerance:
            - Handles LLM wrapping XML in markdown code blocks (```xml ... ```)
            - Handles extra whitespace/newlines in <tool_name> tags
            - Compatible with old format: <code>```python ... ```</code> → returns format error
            - FINAL_ANSWER allows empty <edits>
        
        Returns:
            (tool_name, edits_content, error) tuple.
            - Parse success: (tool_name, edits_content, None)
            - No tool call detected: (None, None, None)
            - Format error: (None, None, error_message)
        """
        text = llm_response
        
        # Preprocessing: remove markdown code block wrappers (```xml ... ``` or ``` ... ```)
        text = re.sub(r'```(?:xml|XML)?\s*\n(<tool_call>)', r'\1', text)
        text = re.sub(r'(</tool_call>)\s*\n```', r'\1', text)
        
        # Match <tool_call> block (relaxed whitespace matching)
        tool_call_match = re.search(
            r'<tool_call>\s*<tool_name>\s*(.*?)\s*</tool_name>(.*?)</tool_call>',
            text,
            re.DOTALL
        )
        if not tool_call_match:
            # Check for incomplete <tool_call> (LLM may have forgotten closing tag)
            partial_match = re.search(r'<tool_call>\s*<tool_name>\s*(.*?)\s*</tool_name>', text, re.DOTALL)
            if partial_match:
                return None, None, (
                    "Format error: detected <tool_call> but missing closing tag </tool_call>.\n"
                    "Please ensure complete tool call block output, including the </tool_call> closing tag."
                )
            return None, None, None
        
        tool_name = tool_call_match.group(1).strip()
        body = tool_call_match.group(2).strip()
        
        # Validate tool name
        valid_tools = {"run_training", "FINAL_ANSWER"}
        if tool_name not in valid_tools:
            return None, None, (
                f"Unknown tool name: '{tool_name}'. Available tools: run_training, FINAL_ANSWER\n"
                f"Please use the correct tool name in <tool_name>."
            )
        
        # Check for old <code> format (complete code mode)
        if '<code>' in body or '```python' in body:
            return None, None, (
                "Format error: please use incremental edit format <edits><edit><search>...</search><replace>...</replace></edit></edits>,\n"
                "not <code> or complete code blocks. Only submit diff edits, do not output complete code."
            )
        
        # Check for <edits> or <edit> blocks
        has_edits_wrapper = '<edits>' in body
        has_edit_block = '<edit>' in body
        
        if has_edits_wrapper or has_edit_block:
            return tool_name, body, None
        
        # FINAL_ANSWER allows empty body (means submit current best code directly)
        if tool_name == "FINAL_ANSWER":
            return tool_name, "", None
        
        # run_training but no valid <edit> blocks
        error_msg = (
            "Format error: run_training must contain <edit> blocks to describe code modifications.\n"
            "Correct format example:\n"
            "<tool_call>\n"
            "<tool_name>run_training</tool_name>\n"
            "<edits>\n"
            "<edit>\n"
            "<search>\n"
            "original code snippet to find\n"
            "</search>\n"
            "<replace>\n"
            "new code snippet after replacement\n"
            "</replace>\n"
            "</edit>\n"
            "</edits>\n"
            "</tool_call>"
        )
        return None, None, error_msg
    
    # ==================================================================
    # Tool Execution
    # ==================================================================
    
    def _execute_run_training(
        self,
        edits_content: str,
        code_path: str,
        tool_call_idx: int,
        llm_response: str = "",
        user_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Execute run_training: parse edits → apply to current_code → create experiment → train → evaluate → conditional update
        """
        experiment_id = self._next_experiment_id()
        print(f"   🔧 [Tool call {tool_call_idx}] run_training (experiment #{experiment_id})")
        
        # 1. Parse edit blocks
        edits = CodeEditor.parse_edits(edits_content)
        
        if not edits:
            return {
                "tool_name": "run_training",
                "success": False,
                "error": (
                    "No valid edit blocks found in output.\n"
                    "Please ensure each <edit> block contains <search> and <replace> sub-tags with proper closing.\n"
                    "Example: <edit><search>original code</search><replace>new code</replace></edit>"
                ),
                "is_final": False,
                "edit_feedback": "❌ No valid edit blocks detected",
            }
        
        # 2. Apply edits on current_code
        edit_result = CodeEditor.apply_edits(self.current_code, edits)
        candidate_code = edit_result.applied_code
        
        # Build edit feedback
        edit_feedback_parts = [
            f"Edits applied: {edit_result.applied_count}/{edit_result.total_count} edit blocks succeeded"
        ]
        if edit_result.errors:
            edit_feedback_parts.append("Failed edit blocks:")
            for err in edit_result.errors:
                edit_feedback_parts.append(f"  - {err}")
        edit_feedback = "\n".join(edit_feedback_parts)
        
        print(f"      ✏️ Edits applied: {edit_result.applied_count}/{edit_result.total_count}")
        if edit_result.errors:
            for err in edit_result.errors:
                print(f"      ⚠️ {err}")
        
        # If all edits failed, don't run training
        if edit_result.applied_count == 0:
            return {
                "tool_name": "run_training",
                "success": False,
                "error": (
                    "All edit blocks failed to match, code was not modified.\n"
                    "Reason: content in <search> does not match the current best code.\n"
                    "Please note:\n"
                    "  1. <search> must exactly match current code character-by-character (including spaces and indentation)\n"
                    "  2. If the last training score did not improve, code was auto-reverted — edit based on the reverted version\n"
                    "  3. Try to copy complete lines from the original code, avoid manually re-typing"
                ),
                "is_final": False,
                "edit_feedback": edit_feedback,
            }
        
        # 3. Create experiment directory (modification saves complete LLM output, including reasoning and code changes)
        modification = llm_response
        
        dependencies = self.evaluator.get_dependencies()
        experiment_dir = self.experiment_manager.create_experiment(
            experiment_id=experiment_id,
            original_code_path=code_path,
            modification=modification,
            dependencies=dependencies,
            code_content=candidate_code,
        )
        if not experiment_dir:
            return {
                "tool_name": "run_training",
                "success": False,
                "error": "Failed to create experiment directory",
                "is_final": False,
                "edit_feedback": edit_feedback,
            }
        print(f"      📁 Experiment dir: {experiment_dir}")
        
        # Save user prompt (for manual analysis)
        if user_prompt:
            self.experiment_manager.save_user_prompt(experiment_dir, user_prompt)
        
        # 4. Run training
        print(f"      ⚡ Running training...")
        result = self.trainer.run(experiment_dir)
        self.experiment_manager.save_training_result(experiment_dir, result)
        
        # 5. Evaluate
        evaluation = self.trainer.evaluate(result)
        self._print_result(evaluation)
        
        # 6. Score-driven conditional update
        new_score = evaluation.get("score", 0.0)
        score_delta = new_score - self.best_score
        code_updated = False
        
        if result.get("success", False) and self.evaluator.is_better(new_score, self.best_score):
            self.current_code = candidate_code
            self.best_score = new_score
            code_updated = True
            direction = "↓" if not self.evaluator.higher_is_better else "↑"
            print(f"      ✅ Score improved! {self.best_score:.4f} ({direction}{abs(score_delta):.4f}) → code updated")
        else:
            print(f"      ❌ Score not improved (this: {new_score:.4f}, best: {self.best_score:.4f}) → code unchanged")
        
        # 7. Save to history
        self.experiment_manager.save_experiment_result(experiment_id, {
            "modification": modification,
            "training_result": result,
            "evaluation": evaluation,
        })
        
        return {
            "tool_name": "run_training",
            "success": result.get("success", False),
            "result": result,
            "evaluation": evaluation,
            "experiment_dir": str(experiment_dir),
            "is_final": False,
            "edit_feedback": edit_feedback,
            "score_feedback": {
                "code_updated": code_updated,
                "best_score": self.best_score,
                "new_score": new_score,
                "score_delta": score_delta,
                "higher_is_better": self.evaluator.higher_is_better,
            },
        }
    
    def _execute_final_answer(
        self,
        edits_content: str,
        code_path: str,
        llm_response: str = "",
        user_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Execute FINAL_ANSWER: apply edits → final training → save results
        """
        # Parse and apply edits
        edits = CodeEditor.parse_edits(edits_content)
        
        if edits:
            edit_result = CodeEditor.apply_edits(self.current_code, edits)
            final_code = edit_result.applied_code
        else:
            # No edits → use current best code
            final_code = self.current_code
        
        # Create final experiment directory
        experiment_id = self._next_experiment_id()
        print(f"\n   🏁 FINAL_ANSWER: Running final training (experiment #{experiment_id})")
        
        dependencies = self.evaluator.get_dependencies()
        experiment_dir = self.experiment_manager.create_experiment(
            experiment_id=experiment_id,
            original_code_path=code_path,
            modification=llm_response or "[FINAL_ANSWER] Final code submission",
            dependencies=dependencies,
            code_content=final_code,
        )
        if not experiment_dir:
            return {
                "tool_name": "FINAL_ANSWER",
                "success": False,
                "error": "Failed to create final experiment directory",
                "is_final": True,
            }
        print(f"      📁 Experiment dir: {experiment_dir}")
        
        # Save user prompt (for manual analysis)
        if user_prompt:
            self.experiment_manager.save_user_prompt(experiment_dir, user_prompt)
        
        # Run training
        print(f"      ⚡ Running final training...")
        result = self.trainer.run(experiment_dir)
        self.experiment_manager.save_training_result(experiment_dir, result)
        
        # Evaluate
        evaluation = self.trainer.evaluate(result)
        self._print_result(evaluation)
        
        # Score-driven update
        new_score = evaluation.get("score", 0.0)
        if result.get("success", False) and self.evaluator.is_better(new_score, self.best_score):
            score_delta = new_score - self.best_score
            self.current_code = final_code
            self.best_score = new_score
            direction = "↓" if not self.evaluator.higher_is_better else "↑"
            print(f"      ✅ Final code score improved! {self.best_score:.4f} ({direction}{abs(score_delta):.4f})")
        
        # Save to history
        self.experiment_manager.save_experiment_result(experiment_id, {
            "modification": llm_response or "[FINAL_ANSWER] Final code submission",
            "training_result": result,
            "evaluation": evaluation,
        })
        
        return {
            "tool_name": "FINAL_ANSWER",
            "success": result.get("success", False),
            "evaluation": evaluation,
            "is_final": True,
        }
    
    def _build_tool_result_content(
        self,
        tool_exec_result: Dict[str, Any],
        remaining_calls: int,
    ) -> str:
        """Build tool execution result as message text to append to conversation context"""
        tool_name = tool_exec_result["tool_name"]
        success = tool_exec_result.get("success", False)
        
        if tool_name == "run_training":
            result = tool_exec_result.get("result", {})
            # Truncate output to prevent excessive length
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            output_parts = []
            if stdout:
                truncated_stdout = stdout[-MAX_OUTPUT_LENGTH:] if len(stdout) > MAX_OUTPUT_LENGTH else stdout
                if len(stdout) > MAX_OUTPUT_LENGTH:
                    truncated_stdout = f"... (output truncated, showing last {MAX_OUTPUT_LENGTH} chars) ...\n" + truncated_stdout
                output_parts.append(f"=== STDOUT ===\n{truncated_stdout}")
            if stderr and not success:
                truncated_stderr = stderr[-MAX_STDERR_LENGTH:] if len(stderr) > MAX_STDERR_LENGTH else stderr
                output_parts.append(f"=== STDERR ===\n{truncated_stderr}")
            
            output = "\n\n".join(output_parts) if output_parts else "(no output)"
            
            evaluation = tool_exec_result.get("evaluation", {})
            metrics = evaluation.get("metrics", {})
            edit_feedback = tool_exec_result.get("edit_feedback")
            score_feedback = tool_exec_result.get("score_feedback")
            
            return self.prompt_builder.build_tool_result_message(
                tool_name=tool_name,
                success=success,
                output=output,
                metrics=metrics,
                remaining_calls=remaining_calls,
                edit_feedback=edit_feedback,
                score_feedback=score_feedback,
            )
        
        # Error feedback
        error = tool_exec_result.get("error", "Unknown error")
        edit_feedback = tool_exec_result.get("edit_feedback")
        return self.prompt_builder.build_tool_result_message(
            tool_name=tool_name,
            success=False,
            output=error,
            remaining_calls=remaining_calls,
            edit_feedback=edit_feedback,
        )
    
    # ==================================================================
    # Baseline
    # ==================================================================
    
    def _run_baseline(self, code_path: str) -> Dict[str, Any]:
        """
        Run baseline (train original code directly, no LLM involved)
        Initializes current_code and best_score
        
        Returns:
            Baseline training result and evaluation
        """
        print("🧪 Running baseline (original code)...")
        
        # Read original code
        baseline_code = Path(code_path).read_text(encoding='utf-8')
        self.current_code = baseline_code
        
        experiment_id = self._next_experiment_id()
        modification = "Baseline: using original code, no modifications."
        
        dependencies = self.evaluator.get_dependencies()
        experiment_dir = self.experiment_manager.create_experiment(
            experiment_id=experiment_id,
            original_code_path=code_path,
            modification=modification,
            dependencies=dependencies,
        )
        if not experiment_dir:
            print("❌ Failed to create baseline experiment directory")
            return {"success": False, "error": "Failed to create experiment directory"}
        
        print(f"   📁 Experiment dir: {experiment_dir}")
        print(f"   ⚡ Running training...")
        
        result = self.trainer.run(experiment_dir)
        self.experiment_manager.save_training_result(experiment_dir, result)
        
        evaluation = self.trainer.evaluate(result)
        
        # Save to history
        self.experiment_manager.save_experiment_result(experiment_id, {
            "modification": modification,
            "training_result": result,
            "evaluation": evaluation,
        })
        
        self._print_result(evaluation)
        
        # Initialize best_score
        self.best_score = evaluation.get("score", 0.0)
        print(f"   📊 Baseline: code={len(baseline_code)} chars, score={self.best_score:.4f}")
        
        return {
            "success": result.get("success", False),
            "result": result,
            "evaluation": evaluation,
        }
    
    # ==================================================================
    # Agent Loop (Core)
    # ==================================================================
    
    def run(self, code_path: str):
        """
        Run the autonomous research Agent Loop
        
        This is the only public entry point. Full flow:
        1. Run baseline
        2. Build initial prompt → enter LLM multi-turn tool calling loop
        3. Output summary when termination condition is met
        
        Args:
            code_path: Target training code path (e.g., algorithms/nanochat/train.py)
        """
        self._print_banner(code_path)
        
        # ---- Step 1: Baseline ----
        baseline_result = self._run_baseline(code_path)
        if not baseline_result.get("success", False):
            print("⚠️ Baseline training failed, but still entering agent loop")
        
        # ---- Step 2: Agent Loop ----
        print(f"\n{'='*60}")
        print(f"🤖 Starting Agent Loop (max {self.max_tool_calls} tool calls)")
        print(f"   📊 Baseline score: {self.best_score:.4f}")
        print(f"{'='*60}")
        
        # Build initial messages
        messages = self.prompt_builder.build_agent_messages(
            code_content=self.current_code,
            readme_content=self.readme_content,
            max_tool_calls=self.max_tool_calls,
        )
        
        for tool_call_idx in range(1, self.max_tool_calls + 1):
            remaining = self.max_tool_calls - tool_call_idx
            
            # Call LLM
            print(f"\n   🧠 [Round {tool_call_idx}/{self.max_tool_calls}] Calling LLM...")
            llm_response = self._call_llm(messages)
            
            # Print token usage for this call
            stats = self.llm_client.token_stats
            if stats.per_request:
                last_usage = stats.per_request[-1]
                cached_info = ""
                if last_usage.cached_tokens > 0:
                    cached_info = f", 🔄cached={last_usage.cached_tokens} ({last_usage.cache_hit_rate:.0%})"
                print(f"      💰 tokens: prompt={last_usage.prompt_tokens}, completion={last_usage.completion_tokens}{cached_info}")
            
            if not llm_response:
                print("   ❌ LLM returned empty response, terminating")
                break
            
            # Parse tool call
            tool_name, edits_content, parse_error = self._parse_tool_call(llm_response)
            
            # ---- Format error ----
            if parse_error:
                print(f"   ⚠️ Format error: {parse_error[:80]}...")
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": (
                    f"<tool_result>\n"
                    f"<status>format_error</status>\n"
                    f"<output>\n{parse_error}\n</output>\n"
                    f"</tool_result>\n\n"
                    f"Please fix the format and retry. Remaining tool calls: {remaining}"
                )})
                print(f"   🔄 Format error feedback sent (remaining: {remaining})")
                continue
            
            # ---- No tool call detected ----
            if tool_name is None:
                print("   ⚠️ No tool call detected")
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": (
                    "<tool_result>\n"
                    "<status>no_tool_call</status>\n"
                    "<output>\n"
                    "Your response does not contain a <tool_call> tag. Each response must contain exactly one tool call.\n\n"
                    "Please use the following format:\n"
                    "<tool_call>\n"
                    "<tool_name>run_training</tool_name>\n"
                    "<edits>\n"
                    "<edit>\n"
                    "<search>original code to find</search>\n"
                    "<replace>replacement code</replace>\n"
                    "</edit>\n"
                    "</edits>\n"
                    "</tool_call>\n"
                    "</output>\n"
                    "</tool_result>\n\n"
                    f"Remaining tool calls: {remaining}"
                )})
                print(f"   🔄 Prompt sent (remaining: {remaining})")
                continue
            
            print(f"   📌 Tool: {tool_name}")
            
            # ---- FINAL_ANSWER ----
            if tool_name == "FINAL_ANSWER":
                # Get the last user message as this round's user prompt
                last_user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
                result = self._execute_final_answer(edits_content, code_path, llm_response, last_user_prompt)
                print("   ✅ Agent Loop ended (FINAL_ANSWER)")
                break
            
            # ---- run_training ----
            # Get the last user message as this round's user prompt
            last_user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
            tool_exec_result = self._execute_run_training(
                edits_content=edits_content,
                code_path=code_path,
                tool_call_idx=tool_call_idx,
                llm_response=llm_response,
                user_prompt=last_user_prompt,
            )
            
            # If run_training execution failed (edit parsing failure, etc.)
            if not tool_exec_result.get("success", False) and "result" not in tool_exec_result:
                # Edit failed, no actual training, return error directly
                messages.append({"role": "assistant", "content": llm_response})
                error = tool_exec_result.get("error", "Unknown error")
                edit_feedback = tool_exec_result.get("edit_feedback", "")
                error_content = self.prompt_builder.build_tool_result_message(
                    tool_name="run_training",
                    success=False,
                    output=error,
                    remaining_calls=remaining,
                    edit_feedback=edit_feedback,
                )
                messages.append({"role": "user", "content": error_content})
                print(f"   🔄 Error feedback sent (remaining: {remaining})")
                continue
            
            # Append tool result to conversation context
            messages.append({"role": "assistant", "content": llm_response})
            tool_result_content = self._build_tool_result_content(tool_exec_result, remaining)
            messages.append({"role": "user", "content": tool_result_content})
            
            print(f"   🔄 Tool result appended (remaining: {remaining})")
        else:
            # for loop completed normally (all tool calls exhausted)
            print(f"\n   ⏰ All {self.max_tool_calls} tool calls exhausted")
        
        # ---- Summary ----
        self._print_summary()
    
    # Backward compatibility alias
    def run_research_loop(self, code_path: str, max_iterations: int = None):
        """Backward compatible entry method (deprecated, use run())"""
        if max_iterations is not None:
            logger.warning("max_iterations parameter is deprecated, use max_tool_calls to control agent loop rounds")
        self.run(code_path)
    
    # ==================================================================
    # Helper Methods
    # ==================================================================
    
    def _print_banner(self, code_path: str):
        """Print startup banner"""
        print("=" * 60)
        print("🚀 Universal AI Autonomous Research Agent (Agentic Loop)")
        print("=" * 60)
        print(f"📁 Algorithm dir: {self.algorithm_dir}")
        print(f"📁 Target code: {code_path}")
        print(f"📁 Experiment dir: {self.experiment_manager.base_dir}")
        print(f"🤖 LLM model: {self.api_config.model_name}")
        print(f"🔧 Max tool calls: {self.max_tool_calls}")
        print(f"📊 Evaluator: {type(self.evaluator).__name__}")
        print(f"📖 README: {'loaded' if self.readme_content else 'not found'}")
        print("=" * 60)
    
    def _print_result(self, evaluation: dict):
        """Print experiment result"""
        status = "✅ Success" if evaluation["success"] else "❌ Failed"
        print(f"\n{status} | Score: {evaluation['score']:.4f}")
        if evaluation.get("metrics"):
            print(f"   Metrics: {evaluation['metrics']}")
    
    def _print_summary(self):
        """Print summary (including token usage statistics)"""
        history = self.experiment_manager.get_history()
        print(f"\n{'='*60}")
        print(f"📈 Research complete!")
        print(f"   Total experiments: {len(history)} (including baseline)")
        print(f"   Best score: {self.best_score:.4f}")
        print(f"   Results saved at: {self.experiment_manager.results_file}")
        
        # Token usage statistics
        stats = self.llm_client.token_stats
        if stats.total_requests > 0:
            print(f"\n{stats.summary()}")
        
        print(f"{'='*60}")
