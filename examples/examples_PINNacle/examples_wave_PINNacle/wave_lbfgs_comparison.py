# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_comparison",
  workspace="saitama32"
)


import torch
import os
import sys
import time
import numpy as np
import random
import tempfile

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
    "description": "comparison_rl_vs_lbfgs"
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

def wave_1d_basic_experiment(seed, x_res, t_res, beta=5):
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

    optim_param = {'history_size': 100,
                    "line_search_fn": 'strong_wolfe',
                    "lr": 1}

    
    optim = Optimizer('LBFGS', optim_param)
    model.train(optim,
                5000,
                save_model=True,
                callbacks=[cb_es],
                models_concat_flag=False,
                equation_params=equation_params)
    
    x = torch.linspace(0, 1, x_res)    # сетка по x

    grid = torch.cartesian_prod(torch.linspace(0, 1, x_res), torch.linspace(0, 1, t_res))
    
    error_op_rmse_train = torch.sqrt(torch.mean((exact_func(grid).reshape(-1, 1) - net(grid)) ** 2))
    variable_dict = domain.variable_dict
    bconds = boundaries.build(variable_dict)
    error_bnd_rmse_train = torch.sum(torch.stack([
                        torch.sqrt(torch.mean(
                            (b["bval"].reshape_as(net(b["bnd"])) - net(b["bnd"])) ** 2, dtype=torch.float32
                        ))
                        for b in bconds
                    ]))
    error_rmse_train_full = error_op_rmse_train + error_bnd_rmse_train

    error_l2re_train = torch.sqrt(torch.sum(
    (exact_func(grid).reshape(-1, 1) - net(grid)) ** 2) / torch.sum(exact_func(grid).reshape(-1, 1) ** 2))
    print(f"Train full RMSE: {error_rmse_train_full}, Train op RMSE: {error_op_rmse_train}, Train bnd RMSE: {error_bnd_rmse_train}, L2RE op: {error_l2re_train}")


    # Test errors
    domain_test = Domain()
    domain_test.variable('x', [x_min, x_max], 100)
    domain_test.variable('t', [0, t_max], 100)
    variable_dict = domain_test.variable_dict
    bconds = boundaries.build(variable_dict)

    error_op_rmse_test = torch.sqrt(torch.mean((exact_func(grid_test).reshape(-1, 1) - net(grid_test)) ** 2))
    error_bnd_rmse_test = torch.sum(torch.stack([
                    torch.sqrt(torch.mean(
                        (b["bval"].reshape_as(net(b["bnd"])) - net(b["bnd"])) ** 2, dtype=torch.float32
                    ))
                    for b in bconds
                ]))
    error_rmse_test_full = error_op_rmse_test + error_bnd_rmse_test
    error_l2re_test = torch.sqrt(torch.sum(
        (exact_func(grid_test).reshape(-1, 1) - net(grid_test)) ** 2) / torch.sum(exact_func(grid_test).reshape(-1, 1) ** 2))
    print(f"Train full RMSE: {error_rmse_test_full}, Train op RMSE: {error_op_rmse_test}, Train bnd RMSE: {error_bnd_rmse_test}, L2RE op: {error_l2re_test}")

    
    experiment.log_parameters({
    "error_op_rmse_train": error_op_rmse_train.item(),
    "error_bnd_rmse_train": error_bnd_rmse_train.item(),
    "error_rmse_train_full": error_rmse_train_full.item(),
    "error_l2re_train": error_l2re_train.item(),
    "error_op_rmse_test": error_op_rmse_test.item(),
    "error_bnd_rmse_test": error_bnd_rmse_test.item(),
    "error_rmse_test_full": error_rmse_test_full.item(),
    "error_l2re_test": error_l2re_test.item()
    })

    experiment.log_parameters({
        'name': 'LBFGS',
        'history_size': 100,
        "line_search_fn": 'strong_wolfe'})
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_params:
        torch.save(model.net.state_dict(), tmp_params.name)
        params_path = tmp_params.name

    # --- логируем как модельные файлы ---
    experiment.log_model(
        name="PINN_optim",
        file_or_folder=params_path,
        file_name=f"model_PINN_{seed}_.pt",
        overwrite=True,
        metadata={"type": "optimizer_state"}
            )

    return exp_dict_list


if __name__ == "__main__":
    x_res = 257
    t_res = 101
    beta = 5

    # список сидов для экспериментов
    seeds = [123, 234, 345, 456, 567]  # можно расширить список

    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # (опционально, если используешь CUDA)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # запуск эксперимента
        exp_dict_list = wave_1d_basic_experiment(seed, x_res, t_res, beta)

