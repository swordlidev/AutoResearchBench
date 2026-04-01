from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Align with the existing baselines: fixed cache/data roots and time budget.
# RL does not have a static train/val dataset here, so prepare.py writes the
# experiment metadata and evaluation protocol instead of downloading samples.
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "lunar_lander_data")
DATASET_DIR = os.path.join(DATA_DIR, "lunar_lander_v3")
TIME_BUDGET = 600  # seconds

ENV_ID = "LunarLander-v3"
DEFAULT_EVAL_SEEDS = [101, 203, 307, 401, 509]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare metadata for LunarLander-v3 + PPO baseline"
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=DEFAULT_EVAL_SEEDS)
    args = parser.parse_args()

    root_dir = args.data_dir.resolve() / "lunar_lander_v3"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "logs").mkdir(parents=True, exist_ok=True)

    metadata = {
        "task_type": "reinforcement_learning",
        "env_id": args.env_id,
        "time_budget_seconds": args.time_budget,
        "has_static_dataset": False,
        "train_protocol": "online interaction with environment",
        "validation_protocol": (
            "periodic evaluation on separate episodes with fixed seeds; "
            "this replaces train/val dataset splits for this RL baseline"
        ),
        "eval_seeds": list(args.eval_seeds),
        "metrics": [
            "eval_return_mean",
            "eval_success_rate",
            "eval_episode_length_mean",
        ],
        "log_dir": str((root_dir / "logs").resolve()),
    }

    metadata_path = root_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[prepare] done")
    print(f"[prepare] experiment dir: {root_dir}")
    print(f"[prepare] metadata file:  {metadata_path}")
    print(f"[prepare] env id:         {args.env_id}")
    print(f"[prepare] time budget:    {args.time_budget}s")
    print(
        "[prepare] note: RL baseline has no fixed train/validation dataset; "
        "evaluation uses separate episodes with fixed seeds."
    )


if __name__ == "__main__":
    main()
