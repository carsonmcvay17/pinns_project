# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

class SimpleNN(nn.Module):
    def __init__(self, hidden_size=10, n_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_size))

        for _ in range(n_layers):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_size, hidden_size))

        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        n_out = self.network(x)
        return x + x * (1-x) * n_out
    

class PINN:
    """
    contains all the stuff and things to make the network including loss functions etc
    """
    def __init__(self, lambda_param=1.0, device='cpu'):
        self.lambda_param = lambda_param
        self.device = device
    
    def physics_loss(self, model, x_colloc):
        """
        Physics loss defined by
        L=1/M /sum_j=1^M |F(y(x_j;theta);x_j)|^2
        where M is num training points and 
        F(y(x)) = y''(x)-lambda sinh(lambda y(x))
        """
        x_colloc = x_colloc.clone().detach().requires_grad_(True)

        y = model(x_colloc)

        # first derivative
        dy_dx = torch.autograd.grad(y, x_colloc, torch.ones_like(y), create_graph=True)[0]

        # second derivative
        d2y_dx2 = torch.autograd.grad(dy_dx, x_colloc, torch.ones_like(y), create_graph=True)[0]

        # compute residual
        lambda_ = self.lambda_param
        F = d2y_dx2 - lambda_ * torch.sinh(lambda_ * y)

        # physics loss ms residual
        loss_physics = torch.mean(F**2)

        return loss_physics

    def train_pinn(self, model, x_data, y_data, x_colloc, epochs=10000, lr=1e-3, lambda_physics=1):
        """
        Train PINN
        """
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # store loss history
        physics_losses = []
        total_losses = []

        pbar = tqdm(range(epochs), desc="training pinn ")
        for epoch in pbar:
            optimizer.zero_grad()

            # data loss
            y_pred = model(x_data)
            loss_data = criterion(y_pred, y_data)

            # physics loss
            loss_physics = self.physics_loss(model, x_colloc)

            # total loss
            total_loss = lambda_physics * loss_physics

            # backprop
            total_loss.backward()
            optimizer.step()

            # store losses
            physics_losses.append(loss_physics.item())
            total_losses.append(total_loss.item())

            if (epoch + 1) % 2000 ==0:
                pbar.set_postfix({'Loss': f'{loss_data.item():.6f}', 
                              'Physics': f'{loss_physics.item():.6f}', 
                              'Total': f'{total_loss.item():.6f}'})
        return physics_losses, total_losses
    
    def get_project_path(self, path_str: str) -> Path:
        """
        Return path relative to root
        """
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR
        while PROJECT_ROOT.name != "pinns_project":
            PROJECT_ROOT = PROJECT_ROOT.parent
        
        # remove leading slash if present
        path_str = path_str.lstrip("/\\")
        return PROJECT_ROOT / Path(path_str)





    

