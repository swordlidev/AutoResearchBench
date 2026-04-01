"""
Evaluator loader module

Responsible for dynamically loading algorithm-specific evaluators
"""

import logging
import importlib.util
from pathlib import Path
from typing import Optional

from .evaluator import BaseEvaluator, DefaultEvaluator

logger = logging.getLogger(__name__)


def load_evaluator(algorithm_dir: Optional[Path]) -> BaseEvaluator:
    """
    Load an algorithm-specific evaluator
    
    Searches for evaluator.py in the algorithm directory and dynamically loads
    the class that inherits from BaseEvaluator.
    Falls back to default evaluator if not found or loading fails.
    
    Args:
        algorithm_dir: Algorithm directory path
        
    Returns:
        Evaluator instance
    """
    if algorithm_dir is None:
        logger.info("No algorithm directory specified, using default evaluator")
        return DefaultEvaluator()
    
    algorithm_dir = Path(algorithm_dir)
    evaluator_path = algorithm_dir / "evaluator.py"
    
    if not evaluator_path.exists():
        logger.info(f"No evaluator.py found in {algorithm_dir}, using default evaluator")
        return DefaultEvaluator()
    
    try:
        # Dynamically load evaluator.py
        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module: {evaluator_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find class that inherits from BaseEvaluator
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, BaseEvaluator) and 
                obj is not BaseEvaluator):
                logger.info(f"Loaded evaluator: {name} from {evaluator_path}")
                return obj()
        
        logger.warning(f"No BaseEvaluator subclass found in evaluator.py, using default evaluator")
        return DefaultEvaluator()
        
    except Exception as e:
        logger.error(f"Failed to load evaluator: {e}, using default evaluator")
        return DefaultEvaluator()
