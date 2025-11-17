from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_heat_2d_comparison",
  workspace="saitama32"
)

import torch
import numpy as np
import os
import sys
import time
import random
import tempfile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "comparison_Heat_2d_basic_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)


eps = 1
N = 10
k = torch.arange(N)


def exact_func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = torch.sum((torch.sin(k * x[:, None]) + torch.sin(k * y[:, None])) * torch.exp(-k ** 2 * t[:, None]))
    return sln


def heat_2d_long_time_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    y_min, y_max = 0, 2 * torch.pi
    t_max = 0.01

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()

    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], 10)

    domain_test = Domain()
    grid_test_res = 80

    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('y', [y_min, y_max], grid_test_res)
    domain_test.variable('t', [0, t_max], 10)

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=lambda grid: torch.sum(
        torch.sin(k * grid[:, 0][:, None]) + torch.sin(k * grid[:, 1][:, None])))

    # Boundary conditions (periodic): ##################################################################################

    # u(0, y, t) = u(2*pi, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                        {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}])

    # u(x, 0, t) = u(x, 2*pi, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                        {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}])

    equation = Equation()

    # Operator: du/dt -  epsilon * (u_xx + u_yy) = 0

    heat_LT = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 0
            },
        '-epsilon * d2u/dx2**1':
            {
                'coeff': -eps,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-epsilon * d2u/dy2**1':
            {
                'coeff': -eps,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(heat_LT)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
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


    model_layers = [pde_dim_in, neurons, neurons, neurons, neurons, pde_dim_out]

    grid = domain.build('NN').to(device)
    grid_test = domain_test.build('NN').to(device)

    model = Model(net, domain, equation, boundaries)

    
    model.compile('autograd', lambda_operator=1, lambda_bound=100)
    u_exact_test = exact_func(grid_test).reshape(-1)

    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_long_time_img')


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=100,
                                         randomize_parameter=1e-4,
                                         info_string_every=10)

    optim_param = {'history_size': 100,
                    "line_search_fn": 'strong_wolfe',
                    "lr": 0.5}

    
    optim = Optimizer('LBFGS', optim_param)
    model.train(optim,
                5000,
                save_model=True,
                callbacks=[cb_es],
                models_concat_flag=False,
                equation_params=equation_params)
    

    net = model.net.to(device)
    grid_test = grid_test.to(device)
    u_exact = exact_func(grid).to(device).reshape(-1, 1)
    u_pred = net(grid)
    diff = u_exact - u_pred
    error_op_rmse_train = torch.sqrt(torch.mean(diff ** 2))
    variable_dict = domain.variable_dict
    bconds = boundaries.build(variable_dict)

    boundary_rmse_lst = []
    for b in bconds:
        if isinstance(b["bnd"], torch.Tensor):
            bnd_lst = [b["bnd"]]
        for bnd in bnd_lst:
            net_bnd = net(bnd)
            try:
                result = (b["bval"].reshape_as(net_bnd) - net_bnd) ** 2
            except:
                result = (torch.full(net_bnd.shape, b["bval"].item()) - net_bnd) ** 2
            boundary_rmse_lst.append(torch.sqrt(torch.mean(result)))

    error_bnd_rmse_train = torch.sum(torch.stack(boundary_rmse_lst))
    

    error_rmse_train_full = error_op_rmse_train + error_bnd_rmse_train

    error_l2re_train = torch.sqrt(torch.sum(
    (u_exact - net(grid)) ** 2) / torch.sum(u_exact ** 2))
    print(f"Train full RMSE: {error_rmse_train_full}, Train op RMSE: {error_op_rmse_train}, Train bnd RMSE: {error_bnd_rmse_train}, L2RE op: {error_l2re_train}")


    # Test errors
    domain_test = Domain()
    grid_test_res = 80

    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('y', [y_min, y_max], grid_test_res)
    domain_test.variable('t', [0, t_max], 10)
    variable_dict = domain_test.variable_dict
    bconds = boundaries.build(variable_dict)
    u_exact_test = exact_func(grid_test).to(device).reshape(-1, 1)
    error_op_rmse_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2))
    boundary_rmse_lst = []
    for b in bconds:
        if isinstance(b["bnd"], torch.Tensor):
            bnd_lst = [b["bnd"]]
        for bnd in bnd_lst:
            net_bnd = net(bnd)
            try:
                result = (b["bval"].reshape_as(net_bnd) - net_bnd) ** 2
            except:
                result = (torch.full(net_bnd.shape, b["bval"].item()) - net_bnd) ** 2
            boundary_rmse_lst.append(torch.sqrt(torch.mean(result)))

    error_bnd_rmse_test = torch.sum(torch.stack(boundary_rmse_lst)) 

    error_rmse_test_full = error_op_rmse_test + error_bnd_rmse_test
    error_l2re_test = torch.sqrt(torch.sum(
        (u_exact_test - net(grid_test)) ** 2) / torch.sum(u_exact_test ** 2))
    print(f"Train full RMSE: {error_rmse_test_full}, Train op RMSE: {error_op_rmse_test}, Train bnd RMSE: {error_bnd_rmse_test}, L2RE op: {error_l2re_test}")

    
    experiment.log_metrics({
    "error_op_rmse_train": error_op_rmse_train.item(),
    "error_bnd_rmse_train": error_bnd_rmse_train.item(),
    "error_rmse_train_full": error_rmse_train_full.item(),
    "error_l2re_train": error_l2re_train.item(),
    "error_op_rmse_test": error_op_rmse_test.item(),
    "error_bnd_rmse_test": error_bnd_rmse_test.item(),
    "error_rmse_test_full": error_rmse_test_full.item(),
    "error_l2re_test": error_l2re_test.item()
    }, step=seed)
    
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
    
    
    experiment.log_parameters({
        "seed": seed,
        "grid_res": grid_res
    })


    return exp_dict_list


if __name__ == "__main__":
    grid_res = 200
    # список сидов для экспериментов
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список
    # seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]   # можно расширить список

    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        exp_dict_list = heat_2d_long_time_experiment(grid_res)

