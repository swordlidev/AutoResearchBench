### Algorithm Description
This is a PPO-based reinforcement learning baseline algorithm for the `LunarLander-v3` task.

### Data Description
Reinforcement learning does not use fixed training/validation sets. Training data comes from online interaction between the agent and the environment, and evaluation is performed in separate episodes.

### Metrics
Evaluation metrics include:
- `eval_return_mean`: mean evaluation episode return, higher is better.
- `eval_success_rate`: proportion of episodes with return >= 200, approximate success rate.
- `eval_episode_length_mean`: mean evaluation episode length, used for auxiliary stability assessment.

### Project Structure
- `prepare.py` — Constants, experiment directory and metadata preparation (do not modify)
- `train.py` — Model, optimizer, training loop, etc. (can be modified)

### Objective
The goal is to achieve the highest possible evaluation return within the fixed time budget while ensuring stable code execution.

### Notes
1. Do not modify the `prepare.py` file.
2. Do not modify the evaluation section in `train.py` to ensure fair comparison across experiments.
3. Each experiment runs on a single GPU with a fixed training duration of 9 minutes.
4. Requires gymnasium dependency: gymnasium[box2d]>=1.0.0
