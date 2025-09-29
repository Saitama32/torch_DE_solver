# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn",
  workspace="saitama32"
)


import torch
import os
import sys
import time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data


experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "farm_transitions_Burgers_1d_basic_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(base_dir)


def exact_func(grid, beta=5):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sin(np.pi * x) * torch.cos(2 * np.pi * t) + 0.5 * \
          torch.sin(beta * np.pi * x) * torch.cos(2 * beta * np.pi * t)
    return sln

def wave_1d_basic_experiment(x_res, t_res, beta=5):
    exp_dict_list = []

    x_min, x_max = 0, 1
    t_max = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], x_res)
    domain.variable('t', [0, t_max], t_res)

    x = domain.variable_dict['x']
    t = domain.variable_dict['t']

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    init_func = torch.sin(torch.pi * x) + 0.5 * torch.sin(beta * torch.pi * x)

    # u(x, 0) = f_init(x, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=init_func)

    # u_t(x, 0) = 0
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [x_min, x_max], 't': 0}, operator=bop, value=0)

    # Boundary conditions ##############################################################################################

    # u(0, t) = f_bnd(x, t)
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)

    # u(1, t) = f_bnd(x, t)
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: d2u/dt2 - 4 * d2u/dx2 = 0

    wave_eq = {
        'd2u/dt2**1':
            {
                'coeff': 1,
                'd2u/dt2': [1, 1],
                'pow': 1
            },
        '-C*d2u/dx2**1':
            {
                'coeff': -4,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    equation.add(wave_eq)

    neurons = 200
    pde_dim_in = 2
    pde_dim_out = 1

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    if torch.cuda.device_count() > 1:
        print("Использую", torch.cuda.device_count(), "GPU!")
        net = torch.nn.DataParallel(net)

    net = net.cuda()

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    model = Model(net, domain, equation, boundaries)
    model_layers = [pde_dim_in, neurons, neurons, neurons, pde_dim_out]

    grid  = 1 #заглушка, чтобы не падало

    model.compile('autograd', lambda_operator=1, lambda_bound=100)
    u_exact_test = exact_func(grid_test).reshape(-1)
    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]




    # os.path.join(os.path.dirname(__file__), 'wave_1d_basic_img')
    img_dir = os.path.join(os.path.dirname(__file__), 'burgers_1d_img')


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=1)


    optimizer = {
        'Adam':{
            'lr':[1e-2, 1e-3, 1e-4],
            'epochs':[100, 1000, 2500]
        },
        'LBFGS':{
            'lr':[1, 5e-1, 1e-1],
            'epochs':[100, 500, 1500]
        },
        'PSO':{
            'lr':[0.0, 1e-3, 1e-4],
            'epochs':[100, 200, 300]
        },
        # 'NNCG':{
        #     'lr':[1, 5e-1, 1e-1],
        #     "precond_update_frequency": [5, 10],
        #     'epochs':[6, 11, 21]
        # }
    }


    # optimizer = Optimizer('Adam', {'lr': 1e-4})

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "device": device
    }

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 100,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False
    }

    loss_surface_params = {
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": img_dir
    }

    rl_agent_params = {
        "n_save_models": 10,
        "n_trajectories": 1000,
        "tolerance": 0.814, 
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit" : 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "exact_solution": exact_func,
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "lr": 1e-3,
        "exp": experiment,
    }

    model.train(optimizer,
                5e5,
                save_model=True,
                callbacks=[cb_es],
                rl_agent_params=rl_agent_params,
                models_concat_flag=False,
                model_name='rl_optimization_agent',
                equation_params=equation_params,
                AE_model_params=AE_model_params,
                AE_train_params=AE_train_params,
                loss_surface_params=loss_surface_params)

    return exp_dict_list


if __name__ == "__main__":
    x_res = 257
    t_res = 101
    beta = 5

    exp_dict_list = wave_1d_basic_experiment(x_res, t_res, beta)
