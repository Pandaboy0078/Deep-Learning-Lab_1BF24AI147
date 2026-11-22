"""Batch, stochastic and mini-batch gradient descent on a small regression dataset."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def make_regression_data(n=120, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return x, 3*x + 2 + rng.normal(0, 0.25, n)

def gradient_descent(x, y, batch_size, epochs=60, learning_rate=0.08, seed=42):
    if len(x) != len(y) or len(x) == 0 or not 1 <= batch_size <= len(x):
        raise ValueError("Invalid data or batch size")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive")
    rng = np.random.default_rng(seed)
    weight, bias, updates = 0., 0., 0
    epoch_losses = []
    step_losses = [(0, float(np.mean((weight*x + bias-y)**2)))]
    for epoch in range(epochs):
        shuffled = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            ids = shuffled[start:start+batch_size]
            error = weight*x[ids] + bias-y[ids]
            weight -= learning_rate*2*np.mean(error*x[ids])
            bias -= learning_rate*2*np.mean(error)
            updates += 1
            step_losses.append((updates, float(np.mean((weight*x+bias-y)**2))))
        epoch_losses.append(float(np.mean((weight*x+bias-y)**2)))
    return dict(weight=weight, bias=bias, updates=updates,
                epoch_losses=epoch_losses, step_losses=step_losses)

def run_demo(output_dir, epochs=60, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = make_regression_data(seed=seed)
    methods = {"Batch":len(x), "Stochastic":1, "Mini-batch (16)":16}
    results = {name:gradient_descent(x,y,b,epochs=epochs,seed=seed)
               for name,b in methods.items()}
    fig, ax = plt.subplots(1,3,figsize=(16,4.5))
    for name, r in results.items():
        ax[0].plot(range(1,epochs+1),r["epoch_losses"],label=name)
        steps=np.asarray(r["step_losses"])
        ax[1].plot(steps[:,0],steps[:,1],label=name)
    ax[0].set(title="Loss versus epochs",xlabel="Epoch",ylabel="MSE")
    ax[1].set(title="Loss versus updates",xlabel="Updates",ylabel="MSE",ylim=(0,None))
    ax[0].legend(); ax[1].legend()
    ax[2].scatter(x,y,s=15,alpha=.4,label="Data")
    xx=np.linspace(x.min(),x.max(),80)
    for name,r in results.items():
        ax[2].plot(xx,r["weight"]*xx+r["bias"],label=name)
    ax[2].set(title="Fitted lines",xlabel="x",ylabel="y")
    ax[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output_dir/"gradient_descent.png",dpi=160);plt.close(fig)
    for name,r in results.items():
        print(f"{name}: MSE={r['epoch_losses'][-1]:.4f}; updates={r['updates']}")
    return results

if __name__=="__main__":
    run_demo("outputs")
