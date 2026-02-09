from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_KS_comparison",   
  workspace="saitama32"
)
import torch
import numpy as np
import os
import sys
import time
import random
import tempfile
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data


experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "comparison_KS_basic_RL_optimizer"
})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_key",
        type=str,
        default=None,
        help="Comet experiment key for comparison runs"
    )
    return parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/Kuramoto_Sivashinsky.npy"))

alpha = 100 / 16
beta = 100 / 16**2
gamma = 100 / 16**4


def kuramoto_sivashinsky_experiment(grid_res, exp_key=None):
    exp_dict_list = []

    x_min, x_max = 0, 2 * np.pi
    t_max = 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    domain_test = Domain()
    grid_test_res = 80
    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('t', [0, t_max], grid_test_res)


    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    init_func = lambda grid: torch.cos(grid[:, 0]) * (1 + torch.sin(grid[:, 0]))

    # u(x, 0) = cos(x) * (1 + sin(x))
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=init_func)

    # Boundary conditions (periodic in x) #############################################################################

    # u(0, t) = u(2π, t)  for all t in [0, t_max]
    boundaries.periodic([
        {'x': x_min, 't': [0, t_max]},
        {'x': x_max, 't': [0, t_max]},
    ])

    equation = Equation()

    # Operator: u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxx = 0

    KS_equation = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1
            },
        'alpha * u * du/dx**1':
            {
                'coeff': alpha,
                'term': [[None], [0]],
                'pow': [1, 1]
            },
        'beta * d2u/dx2**1':
            {
                'coeff': beta,
                'term': [0, 0],
                'pow': 1
            },
        'gamma * d4u/dx4**1':
            {
                'coeff': gamma,
                'term': [0, 0, 0, 0],
                'pow': 1
            },
    }

    equation.add(KS_equation)

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
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model_layers = [pde_dim_in, neurons, neurons, neurons, neurons, neurons, pde_dim_out]

    grid = domain.build('NN').to(device)
    grid_test = domain_test.build('NN').to(device)

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'kuramoto_sivashinsky_img')

    u_exact_test = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out).reshape(-1)
        
    
    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         randomize_parameter=1e-4,
                                         info_string_every=10)

    optim_param = {'history_size': 100,
                        "line_search_fn": 'strong_wolfe',
                        "lr": 0.5}

        
    optim = Optimizer('LBFGS', optim_param)
    model.train(optim,
                7000,
                save_model=True,
                callbacks=[cb_es],
                models_concat_flag=False,
                equation_params=equation_params)

    

    net = model.net.to(device)
    grid_test = grid_test.to(device)
    u_exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape(-1, 1)
    u_pred = net(grid)
    diff = u_exact - u_pred
    error_op_mse_train = torch.mean(diff ** 2)
    error_op_rmse_train = torch.sqrt(torch.mean(diff ** 2))
    variable_dict = domain.variable_dict
    bconds = boundaries.build(variable_dict)
    boundary_err_sq = []
    with torch.no_grad():
        for b in bconds:
            btype = b.get("type", None)

            # оставляем только сравнение u на границе
            if btype != "dirichlet" and btype != "periodic":
                continue

            if btype == "periodic":
                bnd_left, bnd_right = b["bnd"]
                for bnd in (bnd_left, bnd_right):
                    bnd = bnd.to(device)
                    u_pred = net(bnd)
                    u_ex = exact_solution_data(bnd, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape_as(u_pred)
                    boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)
            else:
                bnd = b["bnd"].to(device)
                u_pred = net(bnd)
                u_ex = exact_solution_data(bnd, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape_as(u_pred)
                boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)

    error_bnd_mse_train = torch.mean(torch.cat(boundary_err_sq))


    error_bnd_rmse_train = torch.sqrt(torch.mean(torch.cat(boundary_err_sq)))
    error_rmse_train_full = error_op_rmse_train + error_bnd_rmse_train

    error_l2re_train = torch.sqrt(torch.sum(
    (u_exact - net(grid)) ** 2) / torch.sum(u_exact ** 2))
    print(f"Train full RMSE: {error_rmse_train_full}, Train op RMSE: {error_op_rmse_train}, Train bnd RMSE: {error_bnd_rmse_train}, L2RE op: {error_l2re_train}")


    # Test errors
    variable_dict = domain_test.variable_dict
    bconds = boundaries.build(variable_dict)
    u_exact_test = exact_solution_data(grid_test, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape(-1, 1)
    error_op_mse_test = torch.mean((u_exact_test - net(grid_test)) ** 2)
    error_op_rmse_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2))
    boundary_err_sq = []
    with torch.no_grad():
        for b in bconds:
            btype = b.get("type", None)

            # оставляем только сравнение u на границе
            if btype != "dirichlet" and btype != "periodic":
                continue

            if btype == "periodic":
                bnd_left, bnd_right = b["bnd"]
                for bnd in (bnd_left, bnd_right):
                    bnd = bnd.to(device)
                    u_pred = net(bnd)
                    u_ex = exact_solution_data(bnd, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape_as(u_pred)
                    boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)
            else:
                bnd = b["bnd"].to(device)
                u_pred = net(bnd)
                u_ex = exact_solution_data(bnd, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape_as(u_pred)
                boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)

    error_bnd_mse_test = torch.mean(torch.cat(boundary_err_sq))

    error_bnd_rmse_test = torch.sqrt(torch.mean(torch.cat(boundary_err_sq)))
    error_rmse_test_full = error_op_rmse_test + error_bnd_rmse_test
    error_l2re_test = torch.sqrt(torch.sum(
        (u_exact_test - net(grid_test)) ** 2) / torch.sum(u_exact_test ** 2))
    print(f"Train full RMSE: {error_rmse_test_full}, Train op RMSE: {error_op_rmse_test}, Train bnd RMSE: {error_bnd_rmse_test}, L2RE op: {error_l2re_test}")

    
    experiment.log_metrics({
        # RMSE
        "error_op_rmse_train": error_op_rmse_train.item(),
        "error_bnd_rmse_train": error_bnd_rmse_train.item(),
        "error_rmse_train_full": error_rmse_train_full.item(),

        "error_op_rmse_test": error_op_rmse_test.item(),
        "error_bnd_rmse_test": error_bnd_rmse_test.item(),
        "error_rmse_test_full": error_rmse_test_full.item(),

        # MSE (новое)
        "error_op_mse_train": error_op_mse_train.item(),
        "error_bnd_mse_train": error_bnd_mse_train.item(),

        "error_op_mse_test": error_op_mse_test.item(),
        "error_bnd_mse_test": error_bnd_mse_test.item(),

        # остальное
        "error_l2re_train": error_l2re_train.item(),
        "error_l2re_test": error_l2re_test.item(),
    }, step=seed)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_params:
        torch.save(net.state_dict(), tmp_params.name)
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
    grid_res = 100
    # список сидов для экспериментов
    # seeds = [901, 1012]   # можно расширить список
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список
    # seeds = [123, 234, 345, 456, 567]

    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        exp_dict_list = kuramoto_sivashinsky_experiment(grid_res)
