import torch
import numpy as np
from scipy.integrate import quad


mu = 0.01 / np.pi

def u(grid):
    
    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t), limit=100)[0] / quad(integrand2, -np.inf, np.inf, args=(x, t), limit=100)[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)

grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
u_exact_test = u(grid_test).reshape(-1)


# Получение ошибок для уравнения Бюргерса
def get_errors(model, type="u"):
    error_test = torch.sqrt(torch.mean((u_exact_test - model(grid_test).reshape(-1)) ** 2))
    return error_test

# Получение PINN модели
def get_PINN(layer_sizes, device):
    layers = []
    for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
        layer = torch.nn.Linear(i, j)
        layers.append(layer)
        layers.append(torch.nn.Tanh())
    layers = layers[:-1]
    return torch.nn.Sequential(*layers).to(device)
