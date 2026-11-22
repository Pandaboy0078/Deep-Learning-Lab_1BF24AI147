# Program 1 requirement checklist

**USN:** 1BF24AI147

| Teacher requirement | File / output |
| --- | --- |
| Batch, stochastic and mini-batch GD | `lab01/gradient_descent.py`, `gradient_descent.png` |
| MLP on a Fashion-MNIST subset | `MLP`, `build_loaders` in `lab01/fashion_experiments.py` |
| Momentum and adaptive optimizers | `make_optimizer`, optimizer trials |
| Constant vs scheduled learning rate | StepLR and constant comparison, `learning_rate_schedules.png` |
| Investigate overfitting | Small-data baseline, training vs validation loss plot |
| Evaluate dropout | Dropout 0.35 trial |
| Evaluate L2 regularization | Weight decay 0.001 trial |
| Evaluate data augmentation | Random affine training transform |
| Apply early stopping | Validation-loss patience 4 |
| Compare plain and residual MLP | Equal-width, equal-parameter models with same data and training budget |
| Git repository named with USN | `Deep-Learning-Lab_1BF24AI147` |

## Verification status

- [x] All experiment types implemented in source
- [x] NumPy gradient descent measured
- [x] Basic automated tests available
- [ ] Execute full Fashion-MNIST run and save real plots/results
- [ ] Copy actual observations into the lab record
- [x] Project source pushed to GitHub

Later continuous-evaluation programs can be placed in the same repository as `lab02/`, `lab03/`, etc.
