### Algorithm Description
This is a ViT neural network-based image classification training algorithm using the Tiny ImageNet dataset.

### Data Description
The Tiny ImageNet 200 dataset contains images from 200 classes, with 500 training images and 50 validation images per class.

### Metrics
The evaluation metric is `val_acc1` (validation top-1 accuracy), where higher is better.

### Project Structure
- `prepare.py` — Constants, data download and preparation (do not modify)
- `train.py` — Model, optimizer, training loop, etc. (can be modified)

### Objective
The sole objective: achieve the highest `val_acc1`. The only constraint is: the code must run without crashing and complete within the time budget.

### Notes
1. Do not modify the `prepare.py` file.
2. Do not modify the evaluation section in `train.py` to ensure fair comparison across experiments.
3. Each experiment runs on a single GPU with a fixed training duration of 10 minutes.
