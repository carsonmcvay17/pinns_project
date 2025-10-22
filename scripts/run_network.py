# imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from troesch.pinn_utils import PINN, SimpleNN
from troesch.gen_data import GenData

# device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pinn_class = PINN()
img_save_path = '/figs'
img_save_path = pinn_class.get_project_path(img_save_path)


# initialize
data = GenData()


# lambda vals
lambda_vals = [1, 3, 4, 5, 10]
results = {}


# Boundary data, imposing hard boundary conditions
x_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)
y_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)

# collocation points
M = 50
x_colloc = torch.rand((M, 1), dtype=torch.float32).to(device)
x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)

# loop over lambda_vals
for lam in lambda_vals:
    print(f"\n--- Training PINN for lambda = {lam}---")
    model = SimpleNN(hidden_size=10, n_layers=3).to(device)
    pinn = PINN(lambda_param=lam, device=device)

    physics_losses, total_losses = pinn.train_pinn(
        model, x_data, y_data, x_colloc, epochs=10000, lr=1e-3, lambda_physics=1.0
    )

    # evals
    y_pred = model(x_test).detach().cpu().numpy().flatten()

    # analytical approx
    y_true = data.true_sol(x_test.cpu().numpy(), lam=lam).flatten()

    error = np.abs(y_pred - y_true)

    results[lam] = {
        'y_pred': y_pred,
        'y_true': y_true,
        "error": error
    }

# plot sols
plt.figure(figsize=(7,5))
for lam in lambda_vals:
    plt.plot(x_test.cpu().numpy(), results[lam]["y_pred"], label=f"lambda={lam}")
plt.scatter([0, 1], [0, 1], color='black', s=20)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("PINN Solutions for Different Lambda")
plt.grid()
plt.legend()
outfile = os.path.join(img_save_path, 'Solutions.jpg')
plt.savefig(outfile)
plt.show()


# plot errors
for lam in lambda_vals:
    plt.figure(figsize=(7,5))
    plt.plot(
        x_test.cpu().numpy(),
        results[lam]['error'],
        color='red',
        label=f"λ = {lam}"
    )
    plt.xlabel('x')
    plt.ylabel('|y_pred - y_true|')
    plt.title(f"Absolute Error for λ = {lam}")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # save individual file
    outfile_err = os.path.join(img_save_path, f'Error_lambda{lam}.jpg')
    plt.savefig(outfile_err, dpi=300)
    plt.show()







