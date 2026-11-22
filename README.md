# Deep Learning Lab — 1BF24AI147

**Self Learning — Continuous Evaluation — Program 1**

Python programs to study gradient descent and multilayer perceptrons using a subset of Fashion-MNIST.

## Installation

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

The first Fashion-MNIST run downloads the dataset using torchvision. Internet access is needed once.

## Running

```bash
python -m lab01.run --group gd
python -m lab01.run --mode quick
python -m lab01.run --mode full
python -m pytest -q
```

To run just one experiment group:

```bash
python -m lab01.run --group optimizers --mode full
python -m lab01.run --group regularization --mode full
python -m lab01.run --group architecture --mode full
```

For systems without CUDA or Apple MPS, use `--device cpu`.

## Experiments

| Part | Description | Configuration |
| --- | --- | --- |
| 1 | Batch, stochastic, mini-batch gradient descent | Noisy regression data, 120 points |
| 2 | Momentum and adaptive optimizers | SGD momentum, Nesterov, RMSprop, Adam, AdamW |
| 3 | Learning-rate policies | Constant and StepLR |
| 4 | Overfitting | Plot clean training vs validation loss |
| 5 | Regularization | Dropout 0.35, L2 weight decay 0.001, random affine augmentation |
| 6 | Early stopping | Patience of 4 epochs, best validation-loss weights restored |
| 7 | Model comparison | Plain and residual MLP with equal layer widths and parameter counts |

The regression demo uses identical data and 60 epochs for batch size 120, 1 and 16. Loss is compared against both epochs and number of updates.

For Fashion-MNIST, the code selects fixed, class-balanced, disjoint subsets. Quick mode uses 1,000 training, 300 validation and 500 test examples. Full mode uses 6,000 training, 1,500 validation and 2,000 test examples. Regularization runs on a 400-image training subset to make overfitting more apparent.

All MLP trials use seed 42, batch size 64 and width 128 by default. The plain and residual networks have the same learned layer sizes (784–128–128–128–10) and trainable parameter count. The skip connection changes the computation but adds no parameters.

Learning rates: 0.025 for SGD momentum and Nesterov; 0.001 for RMSprop, Adam and AdamW. Scheduled trials halve the rate approximately every third of the training budget. Validation loss chooses the best epoch and one winner per group; the independent test subset is used only to evaluate that winner.

## Outputs

Running the scripts creates figures and CSV files in `outputs/`:

- `gradient_descent.png`
- `optimizer_validation_accuracy.png`, `optimizer_validation_loss.png`
- `learning_rate_schedules.png`
- `overfitting_train_vs_val_loss.png`
- `regularization_validation_loss.png`
- `architecture_validation_accuracy.png`, `architecture_validation_loss.png`
- `validation_results.csv`, `epoch_histories.csv`, `selected_test_results.csv`
- `confusion_matrix_*.png`, `class_metrics_*.csv`

Dataset and generated training outputs are ignored by Git. This keeps the repository focused on runnable source code. Add the graphs and measured observations to the lab record after the full run.

## Observations

The NumPy regression example has been run locally with seed 42 and 60 epochs:

| Method | MSE | Updates |
| --- | ---: | ---: |
| Batch | 0.0712 | 60 |
| Stochastic | 0.0688 | 7,200 |
| Mini-batch | 0.0636 | 480 |

Fashion-MNIST accuracy values are **not included** here because a full dataset run has not been completed in this environment. Do not use numbers from simulated image tests as Fashion-MNIST results.

See [LAB_RECORD.md](LAB_RECORD.md) for the lab write-up and [TEACHER_REQUIREMENTS.md](TEACHER_REQUIREMENTS.md) for the requirement checklist. Add later continuous-evaluation programs in `lab02/`, `lab03/`, etc.
