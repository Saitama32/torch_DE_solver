import torch
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('cuda')

domain = Domain()
domain.variable('t', [0, 1], n_points=41)
domain.variable('x', [0, 1], n_points=41)

A = 0.5
C = 2


def func(grid):
    x, t = grid[:, 1], grid[:, 0]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + \
           A * torch.sin(2 * C * np.pi * x) * torch.cos(4 * C * np.pi * t)


boundaries = Conditions()

# Boundary conditions at x=0
boundaries.dirichlet({'t': [0, 1], 'x': 0}, value=func)

# Boundary conditions at x=1
boundaries.dirichlet({'t': [0, 1], 'x': 1}, value=func)

# Initial conditions at t=0
boundaries.dirichlet({'t': 0, 'x': [0, 1]}, value=func)

# Initial conditions (operator) at t=0
bop4 = {
    'du/dt':
        {
            'coeff': 1,
            'du/dt': [0],
            'pow': 1,
        }
}

boundaries.operator({'t': 0, 'x': [0, 1]}, operator=bop4, value=func)

equation = Equation()

# operator is 4*d2u/dx2-1*d2u/dt2=0
wave_eq = {
    '-C*d2u/dx2**1':
        {
            'coeff': -4,
            'd2u/dx2': [1, 1],
            'pow': 1
        },
    'd2u/dt2**1':
        {
            'coeff': 1,
            'd2u/dt2': [0, 0],
            'pow': 1
        }
}

equation.add(wave_eq)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 1))

img_dir = os.path.join(os.path.dirname(__file__), 'wave_eq_img')

model = Model(net, domain, equation, boundaries)

model.compile("autograd", lambda_operator=1, lambda_bound=100)

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                     loss_window=1000,
                                     no_improvement_patience=1000,
                                     patience=10,
                                     randomize_parameter=0,
                                     abs_loss=0.1,
                                     info_string_every=500)

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3}, gamma=0.9, decay_every=1000)

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_es, cb_cache, cb_plots])
