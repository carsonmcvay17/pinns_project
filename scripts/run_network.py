# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from troesch.pinn_utils import PINN, SimpleNN

# device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# instantiate model and pinn
model = SimpleNN(hidden_size=10, n_layers=3).to(device)
pinn = PINN(lambda_param=2, device=device)

# Boundary data, imposing hard boundary conditions
x_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)
y_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)

# collocation points
M = 50
x_colloc = torch.rand((M, 1), dtype=torch.float32).to(device)

# train
data_losses, physics_losses, total_losses = pinn.train_pinn(
    model, x_data, y_data, x_colloc, epochs=10000, lr=1e-3, lambda_physics=1.0
)

# evals
x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
y_pred = model(x_test).detach().cpu().numpy()

plt.plot(x_test.cpu(), y_pred, label="PINN solution")
plt.scatter([0, 1], [0, 1], color='red', label="BCs")
plt.legend()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Troesch Problem PINN Solution")
plt.show()
