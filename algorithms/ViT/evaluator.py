"""
ViT Algorithm Evaluator

This evaluator is designed for ViT image classification tasks (Tiny ImageNet).
Primary metric: val_acc1 (validation top-1 accuracy, higher is better)
"""

import re
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoresearch import BaseEvaluator


class ViTEvaluator(BaseEvaluator):
    """
    ViT Evaluator

    Focuses on the following metrics:
    - val_acc1: validation top-1 accuracy (primary metric, higher is better)
    - val_loss: validation loss
    - train_loss: training loss
    - train_acc1: training top-1 accuracy
    - training_time: training time (seconds)
    """

    def extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract metrics from ViT training output"""
        metrics = {}

        # Extract summary metrics at end of training
        # Format: "best val acc1: 0.1234"
        match = re.search(r"best val acc1:\s*([0-9.]+)", stdout)
        if match:
            metrics["best_val_acc1"] = float(match.group(1))

        # Format: "total epochs: 10"
        match = re.search(r"total epochs:\s*([0-9]+)", stdout)
        if match:
            metrics["total_epochs"] = int(match.group(1))

        # Format: "total steps: 1000"
        match = re.search(r"total steps:\s*([0-9]+)", stdout)
        if match:
            metrics["total_steps"] = int(match.group(1))

        # Format: "training time: 600.0s"
        match = re.search(r"training time:\s*([0-9.]+)s", stdout)
        if match:
            metrics["training_time"] = float(match.group(1))

        # Extract detailed metrics from the last epoch summary line
        # Format: "[ep001] train_loss=3.4422 train_acc1=0.3256 val_loss=2.0790 val_acc1=0.5203 10.3s 588/600s"
        epoch_pattern = r"\[ep(\d+)\]\s+train_loss=([0-9.]+)\s+train_acc1=([0-9.]+)\s+val_loss=([0-9.]+)\s+val_acc1=([0-9.]+)"
        epoch_matches = re.findall(epoch_pattern, stdout)
        if epoch_matches:
            last = epoch_matches[-1]
            metrics["last_epoch"] = int(last[0])
            metrics["train_loss"] = float(last[1])
            metrics["train_acc1"] = float(last[2])
            metrics["val_loss"] = float(last[3])
            metrics["val_acc1"] = float(last[4])

        # Extract loss from step logs during training
        # New format: "e001 s0000/0390 lr=0.000100 loss=5.2983 rem=600s"
        loss_pattern = r"loss=([0-9.]+)"
        loss_matches = re.findall(loss_pattern, stdout)
        if loss_matches:
            metrics["final_step_loss"] = float(loss_matches[-1])

        return metrics

    def compute_score(self, metrics: Dict[str, Any], success: bool) -> float:
        """
        Return best_val_acc1 directly as the score (higher is better)

        Return value range 0.0 ~ 1.0, representing top-1 accuracy
        Returns 0.0 on failure
        """
        if not success:
            return 0.0

        # Prefer best_val_acc1 (summary value at end of training)
        best_acc = metrics.get("best_val_acc1")
        if best_acc is not None:
            return float(best_acc)

        # Fallback to last epoch's val_acc1
        val_acc = metrics.get("val_acc1")
        if val_acc is not None:
            return float(val_acc)

        return 0.0

    def get_dependencies(self) -> List[str]:
        """Return dependency files needed by ViT"""
        return ["prepare.py"]
