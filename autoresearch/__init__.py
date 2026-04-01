"""
Universal AI Autonomous Research Agent Package

Module structure:
- config.py: API configuration
- llm_client.py: LLM client (multi-turn conversation)
- evaluator.py: Evaluator base class and default implementation
- evaluator_loader.py: Evaluator dynamic loading
- experiment.py: Experiment management
- trainer.py: Training execution
- prompt_builder.py: Prompt construction (Agentic Loop)
- code_editor.py: Code editing engine (search/replace)
- agent.py: Main agent (pure Agentic Loop architecture)
"""

from .config import APIConfig
from .llm_client import LLMClient, TokenUsage, TokenStats
from .evaluator import BaseEvaluator, DefaultEvaluator
from .evaluator_loader import load_evaluator
from .experiment import ExperimentManager
from .trainer import Trainer
from .prompt_builder import PromptBuilder
from .code_editor import CodeEditor, EditBlock, EditResult
from .agent import UniversalAutoResearchAgent

__all__ = [
    # Core class
    "UniversalAutoResearchAgent",
    
    # Config and client
    "APIConfig",
    "LLMClient",
    "TokenUsage",
    "TokenStats",
    
    # Evaluators
    "BaseEvaluator",
    "DefaultEvaluator",
    "load_evaluator",
    
    # Code editing engine
    "CodeEditor",
    "EditBlock",
    "EditResult",
    
    # Auxiliary modules
    "ExperimentManager",
    "Trainer",
    "PromptBuilder",
]

__version__ = "2.0.0"
