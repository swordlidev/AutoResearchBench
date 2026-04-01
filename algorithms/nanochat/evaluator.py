"""
NanoChat Algorithm Evaluator

This evaluator is designed for nanochat language model pretraining tasks.
Primary metric: val_bpb (validation bits per byte, lower is better)
"""

import re
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoresearch import BaseEvaluator


class NanoChatEvaluator(BaseEvaluator):
    """
    NanoChat Evaluator
    
    Focuses on the following metrics:
    - val_bpb: validation bits per byte (primary metric, lower is better)
    - mfu_percent: model FLOP utilization
    - training_seconds: training time
    """
    
    @property
    def higher_is_better(self) -> bool:
        """val_bpb is a lower-is-better metric."""
        return False
    
    def extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract metrics from nanochat training output"""
        metrics = {}
        
        # nanochat output format: "key:          value" or "key: value"
        patterns = {
            "eval_bpb": r"eval_bpb:\s*([0-9.]+)",
            "val_bpb": r"val_bpb:\s*([0-9.]+)",
            "training_seconds": r"training_seconds:\s*([0-9.]+)",
            "total_seconds": r"total_seconds:\s*([0-9.]+)",
            "peak_vram_mb": r"peak_vram_mb:\s*([0-9.]+)",
            "mfu_percent": r"mfu_percent:\s*([0-9.]+)",
            "total_tokens_M": r"total_tokens_M:\s*([0-9.]+)",
            "num_steps": r"num_steps:\s*([0-9]+)",
            "num_params_M": r"num_params_M:\s*([0-9.]+)",
            "depth": r"depth:\s*([0-9]+)",
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1)
                    metrics[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    metrics[key] = match.group(1)
        
        # Extract training loss from last few lines
        loss_pattern = r"loss:\s*([0-9.]+)"
        loss_matches = re.findall(loss_pattern, stdout)
        if loss_matches:
            metrics["final_loss"] = float(loss_matches[-1])
        
        # Detect failure (FAIL output)
        if "FAIL" in stdout:
            metrics["_failed"] = True
        
        return metrics
    
    def compute_score(self, metrics: Dict[str, Any], success: bool) -> float:
        """
        Return eval_bpb directly as the score (lower is better)
        """
        if not success or metrics.get("_failed"):
            return float("inf")
        
        eval_bpb = metrics.get("eval_bpb")
        if eval_bpb is None:
            eval_bpb = metrics.get("val_bpb")
        if eval_bpb is None:
            return float("inf")
        
        return float(eval_bpb)
    
    
    def get_dependencies(self) -> List[str]:
        """Return dependency files needed by nanochat"""
        return ["prepare.py"]
