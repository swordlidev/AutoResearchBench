"""
SST Algorithm Evaluator

This evaluator is designed for SST-2 text classification tasks (Stanford Sentiment Treebank).
Primary metric: val_acc1 (validation top-1 accuracy, higher is better)
Auxiliary metrics: val_f1, val_recall
"""

import re
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoresearch import BaseEvaluator


class SSTEvaluator(BaseEvaluator):
    """
    SST-2 Text Classification Evaluator

    Focuses on the following metrics:
    - val_acc1: validation top-1 accuracy (primary metric, higher is better)
    - val_f1: validation F1 score
    - val_recall: validation recall
    - val_loss: validation loss
    - train_loss: training loss
    - train_acc1: training top-1 accuracy
    - training_time: training time (seconds)
    """

    def extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract metrics from SST training output"""
        metrics = {}

        # ── Summary metrics at end of training ──

        # Format: "best val acc1: 0.8765"
        match = re.search(r"best val acc1:\s*([0-9.]+)", stdout)
        if match:
            metrics["best_val_acc1"] = float(match.group(1))

        # Format: "best val f1: 0.8800"
        match = re.search(r"best val f1:\s*([0-9.]+)", stdout)
        if match:
            metrics["best_val_f1"] = float(match.group(1))

        # Format: "final val loss: 0.3456"
        match = re.search(r"final val loss:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_val_loss"] = float(match.group(1))

        # Format: "final val acc1: 0.8700"
        match = re.search(r"final val acc1:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_val_acc1"] = float(match.group(1))

        # Format: "final val recall: 0.8600"
        match = re.search(r"final val recall:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_val_recall"] = float(match.group(1))

        # Format: "final val f1: 0.8750"
        match = re.search(r"final val f1:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_val_f1"] = float(match.group(1))

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

        # ── Extract detailed metrics from last epoch summary line ──
        # Format: "[ep001] train_loss=0.4567 train_acc1=0.7890 val_loss=0.3456 val_acc1=0.8765 val_recall=0.8600 val_f1=0.8800 10.3s 588/600s"
        epoch_pattern = (
            r"\[ep(\d+)\]\s+"
            r"train_loss=([0-9.]+)\s+"
            r"train_acc1=([0-9.]+)\s+"
            r"val_loss=([0-9.]+)\s+"
            r"val_acc1=([0-9.]+)\s+"
            r"val_recall=([0-9.]+)\s+"
            r"val_f1=([0-9.]+)"
        )
        epoch_matches = re.findall(epoch_pattern, stdout)
        if epoch_matches:
            last = epoch_matches[-1]
            metrics["last_epoch"] = int(last[0])
            metrics["train_loss"] = float(last[1])
            metrics["train_acc1"] = float(last[2])
            metrics["val_loss"] = float(last[3])
            metrics["val_acc1"] = float(last[4])
            metrics["val_recall"] = float(last[5])
            metrics["val_f1"] = float(last[6])

        # ── Extract metrics from intermediate eval lines ──
        # Format: "[eval] step=00100 min=2.00 train_loss=0.5000 val_loss=0.4000 val_acc1=0.8200 val_recall=0.8100 val_f1=0.8150"
        eval_pattern = (
            r"\[eval\]\s+step=(\d+).*?"
            r"val_loss=([0-9.]+)\s+"
            r"val_acc1=([0-9.]+)\s+"
            r"val_recall=([0-9.]+)\s+"
            r"val_f1=([0-9.]+)"
        )
        eval_matches = re.findall(eval_pattern, stdout)
        if eval_matches:
            last_eval = eval_matches[-1]
            metrics["last_eval_step"] = int(last_eval[0])
            # Only use eval line values as fallback when epoch summary not available
            if "val_loss" not in metrics:
                metrics["val_loss"] = float(last_eval[1])
            if "val_acc1" not in metrics:
                metrics["val_acc1"] = float(last_eval[2])
            if "val_recall" not in metrics:
                metrics["val_recall"] = float(last_eval[3])
            if "val_f1" not in metrics:
                metrics["val_f1"] = float(last_eval[4])

        # ── Extract last loss from training step logs ──
        # Format: "epoch 001 step 0000/0390 update 00000 lr 0.000100 loss 5.2983"
        loss_pattern = r"loss\s+([0-9.]+)"
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

        # Fallback to final_val_acc1
        final_acc = metrics.get("final_val_acc1")
        if final_acc is not None:
            return float(final_acc)

        # Fallback to last epoch's val_acc1
        val_acc = metrics.get("val_acc1")
        if val_acc is not None:
            return float(val_acc)

        return 0.0

    def get_dependencies(self) -> List[str]:
        """Return dependency files needed by SST"""
        return ["prepare.py"]
