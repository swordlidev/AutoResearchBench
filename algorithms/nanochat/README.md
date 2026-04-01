### Algorithm Description
This is a NanoGPT training algorithm.

### Metrics
The evaluation metric is `val_bpb` (validation bits per byte), where lower is better.

### Project Structure
- `prepare.py` — Constants, data preparation and runtime utilities (do not modify)
- `train.py` — Model, optimizer, training loop, etc. (can be modified)

### Objective
The sole objective: achieve the lowest `val_bpb`. The only constraint is: the code must run without crashing and complete within the time budget.

### Notes
1. Do not modify the `prepare.py` file.
2. Do not modify the evaluation section in `train.py` to ensure fair comparison across experiments.
3. Each experiment runs on a single GPU with a fixed training duration of 5 minutes.
