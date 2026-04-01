"""
Evaluator module

Defines the evaluator base class and default implementation.
Each algorithm can implement evaluator.py in its directory, inheriting BaseEvaluator
to define specific evaluation logic.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseEvaluator(ABC):
    """
    Evaluator base class
    
    Each algorithm should implement an evaluator.py in its directory,
    inheriting this class and implementing the relevant methods.
    
    Example:
        ```python
        from autoresearch import BaseEvaluator
        
        class MyEvaluator(BaseEvaluator):
            def extract_metrics(self, stdout, stderr):
                # Extract metrics from output
                return {"accuracy": 0.95, "loss": 0.1}
            
            def compute_score(self, metrics, success):
                # Compute overall score
                if not success:
                    return 0.0
                return metrics.get("accuracy", 0) * 100
        ```
    """
    
    @abstractmethod
    def extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Extract performance metrics from training output
        
        Args:
            stdout: Standard output
            stderr: Standard error output
            
        Returns:
            Extracted metrics dict, e.g., {"loss": 0.1, "accuracy": 0.95}
        """
        pass
    
    @property
    def higher_is_better(self) -> bool:
        """
        Whether a higher score indicates better performance.
        
        Default is True (e.g., accuracy, reward).
        Override to return False for metrics where lower is better (e.g., loss, bpb).
        """
        return True
    
    @abstractmethod
    def compute_score(self, metrics: Dict[str, Any], success: bool) -> float:
        """
        Compute overall score based on metrics
        
        Args:
            metrics: Metrics dict returned by extract_metrics
            success: Whether training completed successfully (return_code == 0)
            
        Returns:
            Overall score from 0 to 100
        """
        pass
    
    def is_better(self, new_score: float, best_score: float) -> bool:
        """
        Compare two scores based on the optimization direction.
        
        Args:
            new_score: Score from the new experiment
            best_score: Current best score
            
        Returns:
            True if new_score is better than best_score
        """
        if self.higher_is_better:
            return new_score > best_score
        else:
            return new_score < best_score
    
    @property
    def worst_score(self) -> float:
        """
        Return the worst possible score (used for initialization).
        
        For higher-is-better: -inf
        For lower-is-better: +inf
        """
        return float('-inf') if self.higher_is_better else float('inf')
    
    def get_dependencies(self) -> List[str]:
        """
        Return list of dependency files/directories to copy to experiment directory (optional)
        
        When the training script depends on other files in the same directory, list them here.
        
        Returns:
            List of dependency file/directory names, e.g., ["prepare.py", "data/", "utils.py"]
        """
        return []


class DefaultEvaluator(BaseEvaluator):
    """
    Default Evaluator
    
    Automatically used when no evaluator.py exists in the algorithm directory.
    Provides generic metric extraction and scoring logic suitable for most training tasks.
    """
    
    def extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Generic metric extraction
        
        Automatically recognizes the following output formats:
        - key: value (e.g., "loss: 0.123")
        - key = value (e.g., "accuracy = 0.95")
        - key:value (e.g., "loss:0.123")
        """
        metrics = {}
        
        # Generic pattern: key: value or key = value
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)[\s]*[:\=][\s]*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)'
        
        # Extract all matching metrics
        matches = re.findall(pattern, stdout)
        for key, value in matches:
            # Filter out keys that are clearly not metrics
            skip_keys = {'step', 'epoch', 'batch', 'iter', 'iteration', 'i', 'j', 'k', 'n', 'v'}
            if key.lower() in skip_keys:
                continue
            try:
                if '.' in value or 'e' in value.lower():
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
            except ValueError:
                metrics[key] = value
        
        return metrics
    
    def compute_score(self, metrics: Dict[str, Any], success: bool) -> float:
        """
        Default scoring logic
        
        - Run failed: 0 points
        - Run succeeded: 50 base points
        - Has loss metric < 1.0: +20 points
        - Has accuracy metric: +10-30 points based on value
        """
        if not success:
            return 0.0
        
        score = 50.0  # Base score
        
        # Common loss metrics (lower is better)
        loss_keys = ['loss', 'val_loss', 'train_loss', 'final_loss', 'best_loss']
        for key in loss_keys:
            if key in metrics:
                loss = metrics[key]
                if isinstance(loss, (int, float)) and loss < 1.0:
                    score += 20
                break
        
        # Common accuracy metrics (higher is better)
        acc_keys = ['accuracy', 'acc', 'val_accuracy', 'val_acc']
        for key in acc_keys:
            if key in metrics:
                acc = metrics[key]
                if isinstance(acc, (int, float)):
                    if acc > 0.9:
                        score += 30
                    elif acc > 0.8:
                        score += 20
                    elif acc > 0.5:
                        score += 10
                break
        
        return min(score, 100.0)
    
