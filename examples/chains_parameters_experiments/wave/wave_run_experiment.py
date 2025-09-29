# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
from comet_ml import start
from comet_ml.integration.pytorch import log_model

import torch
import os
import sys
import time
import numpy as np
import argparse
import sys
import traceback
import random
import tempfile
import datetime



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data


def set_random_seed(seed): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(base_dir)

def parse_params(params):
    """
    Преобразует список ['beta', '5', 'gamma', '0.1'] -> {'beta': 5, 'gamma': 0.1}
    """
    if params is None:
        return {}
    parsed = {}
    for i in range(0, len(params), 2):
        key = params[i]
        val = params[i+1]
        try:
            if "." in val or "e" in val.lower():
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass  # оставить строкой
        parsed[key] = val
    return parsed


def exact_func(grid, beta=5):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sin(np.pi * x) * torch.cos(2 * np.pi * t) + 0.5 * \
          torch.sin(beta * np.pi * x) * torch.cos(2 * beta * np.pi * t)
    return sln


def wave_1d_basic_experiment(experiment_args):

    optimizer = []
    x_min, x_max = 0, 1
    t_max = 1

    x_res = experiment_args["num_x"]
    t_res = experiment_args["num_t"]
    beta = experiment_args["pde_params"].get("beta") 

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

    neurons = experiment_args["num_neurons"]
    n_layers = experiment_args["num_layers"]
    pde_dim_in = 2
    pde_dim_out = 1
    layers = []

    # первый слой: вход → скрытый
    layers.append(torch.nn.Linear(pde_dim_in, neurons))
    layers.append(torch.nn.Tanh())

    # скрытые слои
    for _ in range(n_layers - 2):  # вычитаем входной и выходной
        layers.append(torch.nn.Linear(neurons, neurons))
        layers.append(torch.nn.Tanh())

    # выходной слой
    layers.append(torch.nn.Linear(neurons, pde_dim_out))

    net = torch.nn.Sequential(*layers)
    net.to(device)


    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=1)
    if isinstance(experiment_args["opt"], list):
        for opt_name in experiment_args["opt"]:
            if opt_name == "Adam":
                opt_params = parse_params(experiment_args[f"opt_params_{opt_name}"])
                opt_dict = {
                    "name": opt_name,
                    "params": opt_params
                }
                optimizer.append(opt_dict)
            if opt_name == "LBFGS":
                opt_params = parse_params(experiment_args[f"opt_params_{opt_name}"])
                opt_dict = {
                    "name": opt_name,
                    "params": opt_params
                }
                optimizer.append(opt_dict)
        
        model.train(optim, 10, save_model=False, callbacks=[cb_es], info_string_every=20)

    else:
        opt_type = experiment_args["opt"]
        opt_params = parse_params(experiment_args["opt_params"])
        epochs = experiment_args["epochs"]
        optim = Optimizer(opt_type, opt_params)
        model.train(optim, epochs, save_model=False, callbacks=[cb_es], info_string_every=20)

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

    
    experiment_args["experiment"].log_parameters({
    "error_op_rmse_train": error_op_rmse_train.item(),
    "error_bnd_rmse_train": error_bnd_rmse_train.item(),
    "error_rmse_train_full": error_rmse_train_full.item(),
    "error_l2re_train": error_l2re_train.item(),
    "error_op_rmse_test": error_op_rmse_test.item(),
    "error_bnd_rmse_test": error_bnd_rmse_test.item(),
    "error_rmse_test_full": error_rmse_test_full.item(),
    "error_l2re_test": error_l2re_test.item()
    })

                # Сохраняем модель во временные файлы
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_optim:
        torch.save(net.state_dict(), tmp_optim.name)
        optim_path = tmp_optim.name

    # --- логируем как модельные файлы ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_args["experiment"].exp.log_model(
        name="rl_agent_optim",
        file_or_folder=optim_path,
        file_name=f"model_{opt_type}_{timestamp}.pt",
        overwrite=True,
    )






def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--opt', nargs='+', type=str, default=['lbfgs'],
                        help='optimizer(s) to use')
    parser.add_argument('--opt_params', nargs='+', type=str,
                        default=None, help='optimizer parameters')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--num_x', type=int, default=257,
                        help='number of spatial sample points (power of 2 + 1)')
    parser.add_argument('--num_t', type=int, default=101,
                        help='number of temporal sample points')
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('--comet_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')

    # Extract arguments from parser
    args = parser.parse_args()
    # set initial seed
    initial_seed = args.seed
    set_random_seed(initial_seed)

    experiment = start(
        api_key="aP71fQTYPNqfsYWvudPPmoBl5",
        project_name=args.comet_project,
        workspace="saitama32"
        )
    
    experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "farm_transitions_Burgers_1d_basic_RL_optimizer"
    })


    # organize arguments for the experiment into a dictionary for logging purpose
    pde_params = parse_params(args.pde_params)
    experiment_args = {
        "initial_seed": args.seed,
        "pde": args.pde,
        "pde_params": pde_params,
        "opt": args.opt,
        "opt_params": args.opt_params,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "loss": args.loss,
        "num_x": args.num_x,
        "num_t": args.num_t,
        "num_res": args.num_res, 
        "epochs": args.epochs,
        "comet_project": args.comet_project,
        "device": f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu',
        "experiment": experiment
    }

    # print out arguments
    print("Seed set to: {}".format(initial_seed))
    print("Selected PDE type: {}".format(experiment_args["pde"]))
    print("Specified PDE coefficients: {}".format(
        experiment_args["pde_params"]))
    print("Optimizer to use: {}".format(experiment_args["opt"]))
    print("Specified optimizer parameters: {}".format(
        experiment_args["opt_params"]))
    print("Number of layers: {}".format(experiment_args["num_layers"]))
    print("Number of neurons per layer: {}".format(experiment_args["num_neurons"]))
    print("Number of spatial points (x): {}".format(experiment_args["num_x"]))
    print("Number of temporal points (t): {}".format(experiment_args["num_t"]))
    print("Number of random residual points to sample: {}".format(experiment_args["num_res"]))
    print("Number of epochs: {}".format(experiment_args["epochs"]))
    print("Weights and Biases project: {}".format(
        experiment_args["comet_project"]))
    print("GPU to use: {}".format(experiment_args["device"]))

    experiment.log_parameters(experiment_args)

    # initialize model
    try:
        wave_1d_basic_experiment(experiment_args)
    # log error and traceback info to W&B, and exit gracefully
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise e

if __name__ == "__main__":
    main()