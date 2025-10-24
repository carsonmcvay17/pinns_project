# imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from troesch.pinn_utils import PINN, SimpleNN
from troesch.gen_data import GenData

# device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pinn_class = PINN()
img_save_path = '/figs'
img_save_path = pinn_class.get_project_path(img_save_path)

# fixing randomness because I am getting different results when I run it!
seed = 42 

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# initialize
data = GenData()


# lambda vals
lambda_vals = [1, 3, 4, 5, 10]
results = {}


# Boundary data, imposing hard boundary conditions (not used anymore)
x_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)
y_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)

# collocation points
M = 50

x_colloc = torch.linspace(0, 1, M).reshape(-1, 1).to(device)
x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
x_error = torch.linspace(0, 1, 11).reshape(-1, 1).to(device)

# "true" sols found in the table in the paper
y_paper1 = [0, 0.084623, 0.170101, 0.257297, 0.347106, 0.440474, 0.538414, 0.642024, 0.752527, 0.871314, 1]
y_paper3 = [0, 0.025946, 0.054248, 0.087495, 0.128777, 0.182056, 0.252747, 0.348805, 0.483138, 0.680163, 1]
y_paper10 = [0, 4.211e-5, 1.299e-4, 3.589e-4, 9.779e-4, 2.659e-3, 7.228e-3, 1.966e-2, 5.373e-2, 1.521e-1, 1]

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
    y_pred_paper = model(x_error).detach().cpu().numpy().flatten()

    # analytical approx
    y_true = data.true_sol(x_test.cpu().numpy(), lam=lam).flatten()

    error = np.abs(y_pred - y_true)
    # they only have the paper error for lambda=1, 3, 10
    if lam == 1:
        paper_error = np.abs(y_pred_paper - np.array(y_paper1))
    elif lam == 3:
        paper_error = np.abs(y_pred_paper - np.array(y_paper3))
    elif lam == 10:
        paper_error = np.abs(y_pred_paper - np.array(y_paper10))
    else:
        # just have something so that the code doesn't break
        paper_error = np.abs(y_pred - y_true)

    results[lam] = {
        'y_pred': y_pred,
        'y_true': y_true,
        "error": error, 
        "paper_error": paper_error
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

# plot weird errors from the tables in the paper
lam_error = [1, 3, 10]
for lam in lam_error:
    plt.figure(figsize=(7,5))
    plt.plot(
        x_error.cpu().numpy(),
        results[lam]['paper_error'],
        color='red',
        label=f"λ = {lam}"
    )
    plt.xlabel('x')
    plt.ylabel('|y_pred - y_paper|')
    plt.title(f"Absolute Error for λ = {lam}")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # save individual file
    outfile_err = os.path.join(img_save_path, f'Paper_Error_lambda{lam}.jpg')
    plt.savefig(outfile_err, dpi=300)
    plt.show()







