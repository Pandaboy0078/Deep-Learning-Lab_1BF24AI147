# Self Learning — Continuous Evaluation

**USN:** 1BF24AI147  
**Program 1:** Gradient descent and MLP experiments

## Aim

To demonstrate batch, stochastic and mini-batch gradient descent, compare optimizers and learning-rate strategies on Fashion-MNIST, investigate overfitting and regularization, apply early stopping, and compare plain and residual MLP models.

## Software

Python, NumPy, Matplotlib, PyTorch, torchvision.

## Basic formulas

- Linear regression: `prediction = weight*x + bias`
- Mean squared error: `MSE = (1/n) * sum((prediction - y)^2)`
- Gradient descent: `parameter = parameter - learning_rate*gradient`
- Momentum: `velocity = momentum*velocity + gradient`
- L2 regularization penalizes large weights.
- Residual block: `output = ReLU(x + F(x))`

## Procedure

1. Generate 120 noisy regression points with a fixed seed.
2. Train using batch sizes 120, 1 and 16; compare loss versus epoch and update count.
3. Download Fashion-MNIST and form fixed, class-balanced train, validation and test subsets.
4. Train the same MLP with SGD momentum, Nesterov, RMSprop, Adam and AdamW; compare constant and StepLR policies.
5. Study overfitting using only 400 training images and compare dropout, L2, augmentation, early stopping, and their combination.
6. Train plain and residual MLPs with identical dimensions, trainable parameter count, data split, optimizer and epoch budget.
7. Select the best models using validation loss, test each selected model once, and save comparison tables and confusion matrices.

## Observations: Regression

Values measured on the synthetic NumPy example with seed 42 and 60 epochs:

| Method | Final weight | Final bias | MSE | Updates |
| --- | ---: | ---: | ---: | ---: |
| Batch | 2.8367 | 1.9971 | 0.0712 | 60 |
| Stochastic | 2.9100 | 2.0522 | 0.0688 | 7200 |
| Mini-batch (16) | 2.9867 | 1.9883 | 0.0636 | 480 |

All methods reduced loss, but they needed different numbers of parameter updates. The epoch and optimizer-step plots show different perspectives on training efficiency.

## Observations: Fashion-MNIST (quick run)

The GitHub Actions run on 22 November 2026 used real Fashion-MNIST images with 1,000 training, 300 validation and 500 test examples. The regularization comparison used 400 training examples. These results were recorded using `--mode quick`.

| Experiment | Best configuration (lowest validation loss) | Validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Optimizers and learning rates | RMSprop with scheduled learning rate | 71.33% | 67.8% |
| Regularization | No regularization | 60.33% | 63.8% |
| Plain versus residual MLP | Residual MLP | 74.67% | 70.6% |

The small-data baseline reached 70.0% training accuracy and 60.33% validation accuracy, a gap of 9.67 percentage points. In the seven-epoch quick run, dropout, L2 and augmentation did not improve validation loss over the baseline. Early stopping did not trigger, because the best validation loss was reached at the last epoch. The residual MLP reached 74.67% validation accuracy, compared with 71.67% for the plain MLP, under the same four-epoch training budget.

These are short-run observations only, not a claim that a particular optimizer or regularization strategy is always best. More epochs are needed to judge how regularization and early stopping affect overfitting.

**Result figures:** See [RESULTS.md](RESULTS.md) for all 12 measured PNG graphs, and [outputs/](outputs/) for the CSV result tables.

## Discussion prompts

- Does mini-batch gradient descent produce a useful compromise between frequent updates and stable gradients?
- Which optimizer converges fastest, and does StepLR help in the given training budget?
- What is the difference between train and validation loss in the overfitting experiment?
- How does dropout, L2, data augmentation or early stopping affect validation performance?
- Does adding identity skip connections improve validation accuracy under controlled conditions?

## Result

All the requested experiments have been implemented, and the quick run produced Fashion-MNIST plots and result tables. A full-mode run is recommended before making stronger conclusions.
