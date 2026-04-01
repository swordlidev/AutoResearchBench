"""
LunarLander_PPO Algorithm Evaluator

This evaluator is designed for LunarLander-v3 reinforcement learning tasks (PPO algorithm).
Primary metric: eval_return_mean (mean evaluation episode return, higher is better)
Auxiliary metrics: eval_success_rate, eval_episode_length_mean
"""

import re
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoresearch import BaseEvaluator


class LunarLanderPPOEvaluator(BaseEvaluator):
    """
    LunarLander-v3 PPO Evaluator

    Focuses on the following metrics:
    - eval_return_mean: mean evaluation episode return (primary metric, higher is better, max ~200+)
    - eval_success_rate: proportion of episodes with return >= 200 (approximate success rate)
    - eval_episode_length_mean: mean evaluation episode length
    - pg_loss: policy gradient loss
    - v_loss: value function loss
    - entropy: policy entropy
    - training_time: training time (seconds)
    """

    # LunarLander-v3 return range is approximately [-500, +300]
    # Reference interval for normalizing eval_return_mean to [0, 1]
    RETURN_MIN = -200.0  # Below this is considered score 0
    RETURN_MAX = 250.0   # At this level is considered max score

    def extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract metrics from LunarLander PPO training output"""
        metrics = {}

        # ── Summary metrics at end of training ──

        # Format: "best eval return mean: 245.67"
        match = re.search(r"best eval return mean:\s*([+-]?[0-9.]+)", stdout)
        if match:
            metrics["best_eval_return_mean"] = float(match.group(1))

        # Format: "best eval success rate: 0.90"
        match = re.search(r"best eval success rate:\s*([0-9.]+)", stdout)
        if match:
            metrics["best_eval_success_rate"] = float(match.group(1))

        # Format: "final eval return mean: 230.50"
        match = re.search(r"final eval return mean:\s*([+-]?[0-9.]+)", stdout)
        if match:
            metrics["final_eval_return_mean"] = float(match.group(1))

        # Format: "final eval return std: 45.23"
        match = re.search(r"final eval return std:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_eval_return_std"] = float(match.group(1))

        # Format: "final eval success rate: 0.80"
        match = re.search(r"final eval success rate:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_eval_success_rate"] = float(match.group(1))

        # Format: "final eval episode length mean: 350.5"
        match = re.search(r"final eval episode length mean:\s*([0-9.]+)", stdout)
        if match:
            metrics["final_eval_episode_length_mean"] = float(match.group(1))

        # Format: "total updates: 150"
        match = re.search(r"total updates:\s*([0-9]+)", stdout)
        if match:
            metrics["total_updates"] = int(match.group(1))

        # Format: "total env steps: 614400"
        match = re.search(r"total env steps:\s*([0-9]+)", stdout)
        if match:
            metrics["total_env_steps"] = int(match.group(1))

        # Format: "training time: 540.3s"
        match = re.search(r"training time:\s*([0-9.]+)s", stdout)
        if match:
            metrics["training_time"] = float(match.group(1))

        # ── Extract metrics from intermediate eval lines (as fallback) ──
        # Format: "[eval] up=0150 return_mean=245.67 return_std=30.12 success_rate=0.90 ep_len=350.5"
        eval_pattern = (
            r"\[eval\]\s+up=(\d+)\s+"
            r"return_mean=([+-]?[0-9.]+)\s+"
            r"return_std=([0-9.]+)\s+"
            r"success_rate=([0-9.]+)\s+"
            r"ep_len=([0-9.]+)"
        )
        eval_matches = re.findall(eval_pattern, stdout)
        if eval_matches:
            last_eval = eval_matches[-1]
            metrics["last_eval_update"] = int(last_eval[0])
            if "best_eval_return_mean" not in metrics:
                # When no summary line available, take max from eval lines
                all_returns = [float(m[1]) for m in eval_matches]
                metrics["best_eval_return_mean"] = max(all_returns)
            if "final_eval_return_mean" not in metrics:
                metrics["eval_return_mean"] = float(last_eval[1])
            if "final_eval_success_rate" not in metrics:
                metrics["eval_success_rate"] = float(last_eval[3])
            if "final_eval_episode_length_mean" not in metrics:
                metrics["eval_episode_length_mean"] = float(last_eval[4])

        # ── Extract last training metrics from update logs ──
        # Format: "[up0010] step=... train_return_mean=... pg_loss=... v_loss=... entropy=... ..."
        update_pattern = (
            r"\[up(\d+)\]\s+"
            r"step=(\d+)\s+"
            r"lr=([0-9.]+)\s+"
            r"train_return_mean=([+-]?[0-9.nan]+)\s+"
            r"train_ep_len_mean=([0-9.nan]+)\s+"
            r"pg_loss=([+-]?[0-9.]+)\s+"
            r"v_loss=([0-9.]+)\s+"
            r"entropy=([0-9.]+)\s+"
            r"approx_kl=([0-9.]+)\s+"
            r"clipfrac=([0-9.]+)\s+"
            r"explained_var=([+-]?[0-9.]+)"
        )
        update_matches = re.findall(update_pattern, stdout)
        if update_matches:
            last = update_matches[-1]
            try:
                metrics["train_return_mean"] = float(last[3])
            except ValueError:
                pass  # nan
            metrics["pg_loss"] = float(last[5])
            metrics["v_loss"] = float(last[6])
            metrics["entropy"] = float(last[7])
            metrics["explained_var"] = float(last[10])

        return metrics

    def compute_score(self, metrics: Dict[str, Any], success: bool) -> float:
        """
        Normalize best_eval_return_mean to [0, 1] interval as the score

        LunarLander-v3 return range is approximately [-500, +300]
        Normalization reference: [-200, 250] -> [0.0, 1.0]
        Returns 0.0 on failure
        """
        if not success:
            return 0.0

        # Prefer best_eval_return_mean
        ret = metrics.get("best_eval_return_mean")
        if ret is None:
            ret = metrics.get("final_eval_return_mean")
        if ret is None:
            ret = metrics.get("eval_return_mean")
        if ret is None:
            return 0.0

        ret = float(ret)
        # Normalize to [0, 1]
        score = (ret - self.RETURN_MIN) / (self.RETURN_MAX - self.RETURN_MIN)
        return max(0.0, min(1.0, score))

    def get_dependencies(self) -> List[str]:
        """Return dependency files needed by LunarLander_PPO"""
        return ["prepare.py"]
