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

## Observations: Fashion-MNIST

**Fill these tables only after running `python -m lab01.run --mode full`.**

| Experiment | Best configuration (validation loss) | Validation accuracy | Held-out test accuracy |
| --- | --- | --- | --- |
| Optimizers and rate policies | Not yet measured | — | — |
| Regularization and early stopping | Not yet measured | — | — |
| Plain vs residual MLP | Not yet measured | — | — |

Attach these plots from `outputs/`: optimizer validation accuracy, learning rate schedules, training/validation loss, regularization validation loss, and architecture validation accuracy.

## Discussion prompts

- Does mini-batch gradient descent produce a useful compromise between frequent updates and stable gradients?
- Which optimizer converges fastest, and does StepLR help in the given training budget?
- What is the difference between train and validation loss in the overfitting experiment?
- How does dropout, L2, data augmentation or early stopping affect validation performance?
- Does adding identity skip connections improve validation accuracy under controlled conditions?

## Result

The requested experiment implementations and graph-generation routines are included. Final Fashion-MNIST findings must be written from the measured full-run CSV results rather than assumed in advance.
