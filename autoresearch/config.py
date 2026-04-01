"""
API configuration module

Sensitive fields (url, username, userid, token) are read from environment
variables so that credentials are never committed to version control.

Required environment variables:
    AUTORESEARCH_API_URL      - LLM API endpoint URL
    AUTORESEARCH_USERNAME     - API username
    AUTORESEARCH_USERID       - API user ID
    AUTORESEARCH_TOKEN        - API authentication token

Optional environment variables:
    AUTORESEARCH_MODEL_NAME   - LLM model name (default: gemini-3-pro-preview)
    AUTORESEARCH_MODEL_TYPE   - Model type (default: openai)
"""

import os
from typing import Dict, Optional


class APIConfig:
    """API Configuration"""
    url: str = os.environ.get(
        "AUTORESEARCH_API_URL", ""
    )
    username: str = os.environ.get("AUTORESEARCH_USERNAME", "")
    userid: str = os.environ.get("AUTORESEARCH_USERID", "")
    token: str = os.environ.get("AUTORESEARCH_TOKEN", "")
    timeout: int = 180
    max_retries: int = 3
    model_type: str = os.environ.get("AUTORESEARCH_MODEL_TYPE", "openai")
    model_name: str = os.environ.get("AUTORESEARCH_MODEL_NAME", "gemini-3-pro-preview")
    temperature: float = 0.6
    max_tokens: int = 102400
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_p: float = 0.95
    n: int = 1
    stop: Optional[str] = None
    max_code_length: int = 40980  # Max input code length limit
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "User-Agent": "ifbook-http-client"
        }
    
    @property
    def sec_info(self) -> Dict[str, str]:
        """Get security info"""
        return {
            "username": self.username,
            "userid": self.userid,
            "token": self.token
        }

    def validate(self) -> None:
        """Validate critical configuration"""
        missing = []
        if not self.username:
            missing.append("username")
        if not self.userid:
            missing.append("userid")
        if not self.token:
            missing.append("token")
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
