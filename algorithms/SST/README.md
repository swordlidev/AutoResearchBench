### Algorithm Description
This is a small Transformer-based binary text classification baseline using the SST-2 dataset from GLUE for sentiment classification.

### Data Description
SST-2 is the binary classification version of the Stanford Sentiment Treebank. Each sample is an English sentence labeled as positive or negative sentiment. Training set: 67,349 samples, validation set: 872 samples, test set: 1,821 samples.

### Metrics
Evaluation metrics include:
- `val_loss`: validation cross-entropy loss, lower is better.
- `val_acc1`: validation accuracy, higher is better.
- `val_recall`: positive class recall, higher is better.
- `val_f1`: positive class F1 score, combining precision and recall, higher is better.

### Project Structure
- `prepare.py` - Constants, data download and preparation. Do not modify.
- `train.py` - Model, optimizer, training loop and evaluation logic. Can be modified.

### Objective
The goal is typically to achieve higher `val_acc1` while monitoring overall performance of `val_loss`, `val_recall`, and `val_f1`. As a baseline, the focus is on clear structure, stable execution, and leaving sufficient room for subsequent automatic optimization of `train.py`.

### Notes
1. Do not modify the `prepare.py` file.
2. Avoid arbitrarily changing the evaluation criteria in `train.py`, as this would affect fair comparison between experiments.
3. Each experiment should run stably on a single GPU and complete within the fixed time budget.
