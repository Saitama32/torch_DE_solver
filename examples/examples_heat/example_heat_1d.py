import torch
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.solution import Solution
import time

solver_device('cpu')


def func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = 500 + x
    for i in range(1, 100):
        sln += 8 * np.exp(-1 / 4 * np.pi ** 2 * t * (2 * i - 1) ** 2) * (
                    (-1) ** i + 250 * np.pi * (1 - 2 * i)) * np.sin(1 / 2 * np.pi * x * (2 * i - 1)) / (
                           np.pi - 2 * np.pi * i) ** 2
    return sln


def heat_experiment(grid_res, CACHE):
    exp_dict_list = []

    domain = Domain()

    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    """
    Preparing boundary conditions (BC)
    
    For every boundary we define three items
    
    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality
    
    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    
    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)
    
    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}
    
    Meaning c1*u*d2u/dx2 has the form
    
    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}
    
    None is for function without derivatives
    
    
    bval=torch.Tensor prescribed values at every point in the boundary
    """

    boundaries = Conditions()

    # Boundary conditions at x=0
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=500)

    # Boundary conditions at x=1

    # u'(1,t)=1
    bop2 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1
            }
    }

    boundaries.operator({'x': 1, 't': [0, 1]}, operator=bop2, value=1)

    # Initial conditions at t=0
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=0)

    """
    Defining wave equation
    
    Operator has the form
    
    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    
    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)
    
    
    
    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}
    
    c1 may be integer, function of grid or tensor of dimension of grid
    
    Meaning c1*u*d2u/dx2 has the form
    
    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}
    
    None is for function without derivatives
    
    
    """

    equation = Equation()

    # operator is 4*d2u/dx2-1*d2u/dt2=0
    heat_eq = {
        'du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1
            },
        '-d2u/dx2**1':
            {
                'coeff': -1,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    equation.add(heat_eq)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    t = domain.variable_dict['t']

    h = abs((t[1] - t[0]).item())

    model = Model(net, domain, equation, boundaries)

    model.compile('NN', lambda_operator=1, lambda_bound=10, h=h)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_NN_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=500)

    cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    if CACHE:
        callbacks = [cb_cache, cb_es, cb_plots]
    else:
        callbacks = [cb_es, cb_plots]

    start = time.time()

    model.train(optimizer, 1e6, save_model=CACHE, callbacks=callbacks)

    end = time.time()

    rmse_x_grid = np.linspace(0, 1, grid_res + 1)
    rmse_t_grid = np.linspace(0, 1, grid_res + 1)

    rmse_x = torch.from_numpy(rmse_x_grid)
    rmse_t = torch.from_numpy(rmse_t_grid)

    rmse_grid = torch.cartesian_prod(rmse_x, rmse_t).float()

    error_rmse = torch.sqrt(torch.mean(((func(rmse_grid) - net(rmse_grid)) / 500) ** 2))

    exp_dict_list.append(
        {'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().numpy(), 'type': 'wave_eqn',
         'cache': CACHE})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    return exp_dict_list


nruns = 10

exp_dict_list = []

CACHE = True

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(heat_experiment(grid_res, CACHE))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res', column='time', fontsize=42, figsize=(20, 10))
df.boxplot(by='grid_res', column='RMSE', fontsize=42, figsize=(20, 10), showfliers=False)
df.to_csv('benchmarking_data/heat_experiment_10_100_cache={}.csv'.format(str(CACHE)))
