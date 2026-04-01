"""
Training execution module

Responsible for running training scripts and collecting results
"""

import subprocess
import logging
from typing import Dict, Any
from pathlib import Path

from .evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training executor
    
    Responsible for running training scripts in experiment directories
    and using evaluators to extract metrics
    """
    
    def __init__(self, evaluator: BaseEvaluator):
        """
        Initialize training executor
        
        Args:
            evaluator: Evaluator instance for extracting metrics
        """
        self.evaluator = evaluator
    
    def run(
        self,
        experiment_dir: Path,
        script_name: str = "train.py",
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Run training script
        
        Args:
            experiment_dir: Experiment directory path
            script_name: Training script name (default: train.py)
            timeout: Timeout in seconds
            
        Returns:
            Training result dict containing:
            - success: Whether training succeeded
            - return_code: Return code
            - stdout: Standard output
            - stderr: Standard error
            - Other metrics extracted by the evaluator
        """
        experiment_dir = Path(experiment_dir)
        train_script = experiment_dir / script_name
        
        if not train_script.exists():
            return {
                "success": False,
                "error": f"Training script not found: {train_script}",
                "experiment_dir": str(experiment_dir)
            }
        
        logger.info(f"Running training: {train_script}")
        
        try:
            result = subprocess.run(
                ["uv", "run", script_name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(experiment_dir)
            )
            
            # Base result
            training_result = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "experiment_dir": str(experiment_dir)
            }
            
            # Use evaluator to extract metrics
            metrics = self.evaluator.extract_metrics(result.stdout, result.stderr)
            training_result.update(metrics)
            
            return training_result
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Training timed out ({timeout}s): {experiment_dir}")
            return {
                "success": False,
                "error": f"Training timed out ({timeout}s)",
                "experiment_dir": str(experiment_dir)
            }
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "experiment_dir": str(experiment_dir)
            }
    
    def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate training results
        
        Args:
            result: Training result returned by run()
            
        Returns:
            Evaluation result dict containing:
            - success: Whether training succeeded
            - score: Score (0-100)
            - metrics: Extracted metrics
        """
        success = result.get("success", False)
        
        # Extract metrics (exclude internal fields)
        internal_keys = {'success', 'return_code', 'stdout', 'stderr', 'error', 'experiment_dir'}
        metrics = {k: v for k, v in result.items() if k not in internal_keys}
        
        # Use evaluator to compute score
        score = self.evaluator.compute_score(metrics, success)
        
        evaluation = {
            "success": success,
            "score": score,
            "metrics": metrics,
        }
        
        if not success:
            error = result.get("error", "") or result.get("stderr", "")
            evaluation["error"] = error[-500:] if len(error) > 500 else error
        
        return evaluation
