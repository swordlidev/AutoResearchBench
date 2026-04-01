"""
Experiment management module

Responsible for experiment directory creation, file management, and result saving
"""

import json
import shutil
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Experiment Manager
    
    Responsible for:
    - Creating experiment directory structures
    - Saving code and modification suggestions
    - Copying dependency files
    - Saving experiment results and history
    """
    
    def __init__(self, base_dir: Path, algorithm_name: str):
        """
        Initialize experiment manager
        
        Args:
            base_dir: Experiment base directory (e.g., experiments/nanochat)
            algorithm_name: Algorithm name
        """
        self.base_dir = Path(base_dir)
        self.algorithm_name = algorithm_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.base_dir / "results.json"
        self.experiment_history: List[Dict[str, Any]] = []
        self.current_experiment_dir: Optional[Path] = None
        
        # Load existing experiment history
        self._load_history()
    
    def _load_history(self):
        """Load existing experiment history"""
        if self.results_file.exists():
            try:
                self.experiment_history = json.loads(self.results_file.read_text(encoding='utf-8'))
                logger.info(f"Loaded {len(self.experiment_history)} historical experiment records")
            except Exception as e:
                logger.warning(f"Failed to load experiment history: {e}")
                self.experiment_history = []
    
    def create_experiment(
        self,
        experiment_id: int,
        original_code_path: str,
        modification: str,
        dependencies: List[str] = None,
        code_content: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Create a new experiment directory
        
        Args:
            experiment_id: Experiment number
            original_code_path: Original code path
            modification: Modification description
            dependencies: List of dependency files to copy
            code_content: Pre-edited complete code (optional).
                          If provided, written directly to train.py; otherwise copies original code.
            
        Returns:
            Experiment directory path, or None on failure
        """
        try:
            # Create experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{experiment_id:03d}_{timestamp}"
            experiment_dir = self.base_dir / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Save modification suggestion
            self._save_modification_suggestion(
                experiment_dir, experiment_name, experiment_id, 
                original_code_path, modification
            )
            
            # Determine training code: prefer code_content, otherwise read from original path
            if code_content is not None:
                new_code = code_content
            else:
                new_code = Path(original_code_path).read_text(encoding='utf-8')
            (experiment_dir / "train.py").write_text(new_code, encoding='utf-8')
            
            # Copy dependency files
            source_dir = Path(original_code_path).parent
            self._copy_dependencies(source_dir, experiment_dir, dependencies or [])
            
            # Save meta info
            self._save_meta(experiment_dir, experiment_name, experiment_id, original_code_path)
            
            self.current_experiment_dir = experiment_dir
            
            logger.info(f"Created experiment directory: {experiment_dir}")
            return experiment_dir
            
        except Exception as e:
            logger.error(f"Failed to create experiment directory: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_modification_suggestion(
        self,
        experiment_dir: Path,
        experiment_name: str,
        experiment_id: int,
        original_code_path: str,
        modification: str
    ):
        """Save modification suggestion to markdown file"""
        content = f"""# Experiment {experiment_name}

## Algorithm: {self.algorithm_name}
## Experiment ID: {experiment_id}
## Generated at: {datetime.now().isoformat()}
## Original code: {original_code_path}

## Modification Description

{modification}
"""
        (experiment_dir / "modification_suggestion.md").write_text(content, encoding='utf-8')
    
    def _copy_dependencies(
        self,
        source_dir: Path,
        target_dir: Path,
        custom_deps: List[str]
    ):
        """Copy dependency files from algorithm directory to experiment directory"""
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        ignore_names = {
            "train.py",
            "README.md",
            "evaluator.py",
            "__pycache__",
            ".git",
            ".idea",
            ".vscode",
            ".DS_Store",
        }

        # Copy all code and resources from algorithm directory (excluding ignored items)
        for entry in source_dir.iterdir():
            if entry.name in ignore_names:
                continue
            src = entry
            dst = target_dir / entry.name
            try:
                if src.is_file():
                    shutil.copy2(src, dst)
                else:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                logger.debug(f"Copied dependency: {entry.name}")
            except Exception as e:
                logger.warning(f"Failed to copy dependency {entry.name}: {e}")

        # Copy custom dependencies declared by evaluator
        for dep in custom_deps or []:
            dep_path = Path(dep)
            src = dep_path if dep_path.is_absolute() else source_dir / dep_path
            if not src.exists():
                logger.warning(f"Custom dependency not found: {src}")
                continue

            rel_path = dep_path if not dep_path.is_absolute() else Path(dep_path.name)
            dst = target_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if src.is_file():
                    shutil.copy2(src, dst)
                else:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                logger.debug(f"Copied custom dependency: {dep}")
            except Exception as e:
                logger.warning(f"Failed to copy custom dependency {dep}: {e}")
    
    def _save_meta(
        self,
        experiment_dir: Path,
        experiment_name: str,
        experiment_id: int,
        original_code_path: str
    ):
        """Save experiment metadata"""
        meta = {
            "experiment_name": experiment_name,
            "algorithm_name": self.algorithm_name,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "original_code_path": str(original_code_path),
        }
        (experiment_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    
    def save_user_prompt(self, experiment_dir: Path, user_prompt: str):
        """Save user prompt for this experiment round (for manual analysis)"""
        (experiment_dir / "user_prompt.md").write_text(user_prompt, encoding='utf-8')
    
    def save_training_result(self, experiment_dir: Path, result: Dict[str, Any]):
        """Save training result to experiment directory"""
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        (experiment_dir / "training_log.txt").write_text(
            f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}",
            encoding='utf-8'
        )
        
        (experiment_dir / "training_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding='utf-8'
        )
    
    def save_experiment_result(self, experiment_id: int, result: Dict[str, Any]):
        """Save experiment result to history"""
        import time
        
        experiment_data = {
            "experiment_id": experiment_id,
            "timestamp": time.time(),
            "result": result
        }
        self.experiment_history.append(experiment_data)
        
        # Persist to file
        self.results_file.write_text(
            json.dumps(self.experiment_history, indent=2, ensure_ascii=False, default=str),
            encoding='utf-8'
        )
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get experiment history"""
        return self.experiment_history
