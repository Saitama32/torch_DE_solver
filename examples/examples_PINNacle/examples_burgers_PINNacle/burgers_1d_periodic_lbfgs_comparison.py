from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_burgers_comaprison",
  workspace="saitama32"
)



import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import numpy as np
import random
import tempfile

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
    "description": "comparison_rl_vs_lbfgs"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)
mu = np.pi / 150
N = 3
k = torch.arange(N)


def exact_func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sum(torch.sin(k * x[:, None]) * torch.exp(k ** 2 * t[:, None]), dim=1, keepdim=True)
    return sln

def burgers_1d_periodic_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    t_max = 0.1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], 11)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # u(x, 0) = -sin(pi * x)
    boundaries.dirichlet(
        {'x': [x_min, x_max], 't': 0},
        value=lambda grid: torch.sum(
            torch.sin(k * grid[:, 0][:, None]),
            dim=1, keepdim=True
        )
    )

    # Boundary conditions ##############################################################################################

    # u(0, t) = u(2*pi, t)
    boundaries.periodic([{'x': x_min, 't': [0, t_max]},
                         {'x': x_max, 't': [0, t_max]}])

    equation = Equation()

    # Operator: u_t + u * u_x - mu * u_xx = 0

    burgers_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

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

    model_layers = [2, neurons, neurons, neurons, neurons, 1]

    start = time.time()

    # net = mat_model(domain, equation)

    grid = domain.build('NN').to(device)
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    u_exact_test = exact_func(grid_test).reshape(-1)

    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    model = Model(net, *equation_params[3:-1])

    model.compile('autograd', lambda_operator=1, lambda_bound=10)


    img_dir = os.path.join(os.path.dirname(__file__), 'burgers_1d_img')

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
    u_exact = exact_func(grid).reshape(-1)
    u_pred = net(grid)
    diff = u_exact - u_pred
    error_op_rmse_train = torch.sqrt(torch.mean(diff ** 2))
    variable_dict = domain.variable_dict
    bconds = boundaries.build(variable_dict)
    boundary_err_sq = []
    with torch.no_grad():
        for b in bconds:
            btype = b.get("type", None)

            # periodic: b["bnd"] = [left, right]
            if btype == "periodic":
                bnd_left, bnd_right = b["bnd"]
                for bnd in (bnd_left, bnd_right):
                    bnd = bnd.to(device)
                    u_pred = net(bnd)
                    u_ex = exact_func(bnd).to(device)
                    boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)

            # non-periodic: b["bnd"] is Tensor
            else:
                if not isinstance(b.get("bnd", None), torch.Tensor):
                    continue
                bnd = b["bnd"].to(device)
                u_pred = net(bnd)
                u_ex = exact_func(bnd).to(device)
                boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)


    error_bnd_rmse_train = torch.sqrt(torch.mean(torch.cat(boundary_err_sq)))
    

    error_rmse_train_full = error_op_rmse_train + error_bnd_rmse_train

    error_l2re_train = torch.sqrt(torch.sum(
    (u_exact - net(grid)) ** 2) / torch.sum(u_exact ** 2))
    print(f"Train full RMSE: {error_rmse_train_full}, Train op RMSE: {error_op_rmse_train}, Train bnd RMSE: {error_bnd_rmse_train}, L2RE op: {error_l2re_train}")


    # Test errors
    domain_test = Domain()
    domain_test.variable('x', [x_min, x_max], grid_res)
    domain_test.variable('t', [0, t_max], 11)
    variable_dict = domain_test.variable_dict
    bconds = boundaries.build(variable_dict)
    u_exact_test = exact_func(grid_test).to(device).reshape(-1, 1)
    error_op_rmse_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2))
    boundary_err_sq = []
    with torch.no_grad():
        for b in bconds:
            btype = b.get("type", None)

            # periodic: b["bnd"] = [left, right]
            if btype == "periodic":
                bnd_left, bnd_right = b["bnd"]
                for bnd in (bnd_left, bnd_right):
                    bnd = bnd.to(device)
                    u_pred = net(bnd)
                    u_ex = exact_func(bnd).to(device)
                    boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)

            # non-periodic: b["bnd"] is Tensor
            else:
                if not isinstance(b.get("bnd", None), torch.Tensor):
                    continue
                bnd = b["bnd"].to(device)
                u_pred = net(bnd)
                u_ex = exact_func(bnd).to(device)
                boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)


    error_bnd_rmse_test = torch.sqrt(torch.mean(torch.cat(boundary_err_sq)))
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

    return exp_dict_list

if __name__ == "__main__":
    grid_res = 100
    # список сидов для экспериментов
    # seeds = [678, 789, 890, 901, 1012]   # можно расширить список
    # seeds = [123, 234, 345, 456, 567]   # можно расширить список
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список
   
    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        exp_dict_list = burgers_1d_periodic_experiment(grid_res=grid_res)

