from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_diffusion_1d_comparison",
  workspace="saitama32"
)


import torch
import os
import sys
import time
import random
import numpy as np
import tempfile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_diffusion')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "compare_diffusion_1d_basic_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)


epsilon = 1
N = 23
k = torch.arange(N)


def exact_func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sum(torch.sin(k * x[:, None]) * torch.exp(-epsilon * k ** 2 * t[:, None]))
    return sln


def diffusion_1d_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    t_max = 0.1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], 10)

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0},
                         value=lambda grid: torch.sum(torch.sin(k * grid[:, 0][:, None])))

    # Boundary conditions (periodic): ##################################################################################

    # u(0, t) = u(2*pi, t)
    boundaries.periodic([{'x': x_min, 't': [0, t_max]},
                         {'x': x_max, 't': [0, t_max]}])

    equation = Equation()

    # Operator 1:  ut - ε1 * u_xx = 0

    diffusion_1d = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            },
        '-epsilon * d2u/dx2**1':
            {
                'coeff': -epsilon,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(diffusion_1d)

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

    start = time.time()

    # net = mat_model(domain, equation)
    grid_res_test = 80
    domain_test = Domain()
    domain_test.variable('x', [x_min, x_max], grid_res_test)
    domain_test.variable('t', [0, t_max], 10)

    grid = domain.build('NN').to(device)
    grid_test = domain_test.build('NN').to(device)
    u_exact_test = exact_func(grid).reshape(-1)

    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'diffusion_1d_img')


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    optimizer = {
        'Adam':{
            'lr':[1e-2, 1e-3, 1e-4],
            'epochs':[101, 1000, 2500]
        },
        'LBFGS':{
            'lr':[1, 5e-1, 1e-1],
            'epochs':[101, 500, 1500]
        },
        'PSO':{
            'lr':[0.0, 1e-3, 1e-4],
            'epochs':[101, 200, 300]
        },
    }

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
        "tolerance": 0.00014624762116, 
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

    comparison_params = {
        "seed": seed, 
        "total_epochs": 5000,
        "experiment_key": "0e63aa646c804af08eedada9bce4563c",
        "grid_res": grid_res
    }

    experiment.log_parameters(rl_agent_params)
    experiment.log_parameters(comparison_params)


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
                loss_surface_params=loss_surface_params,
                comparison_param=comparison_params)
    

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

    return exp_dict_list


if __name__ == "__main__":
    grid_res = 100
    # список сидов для экспериментов
    # seeds = [789, 890, 901, 1012]   # можно расширить список
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список
    # seeds = [123, 234, 345, 456, 567]   # можно расширить список

    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        exp_dict_list = diffusion_1d_experiment(grid_res)