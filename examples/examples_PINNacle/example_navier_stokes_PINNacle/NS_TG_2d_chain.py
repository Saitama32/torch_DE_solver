from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_NS_TG_2d_tolerance",
  workspace="saitama32"
)

import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_navier_stokes')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, cache, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device


experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "tol_definition_NS_TG_2d_basic_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)


ro = 1
mu = 2 * torch.pi / 100


def exact_func_u(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = -torch.cos(x[:, None]) * torch.sin(y[:, None]) * torch.exp(-2 * mu * t[:, None])
    return sln


def exact_func_v(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = torch.sin(x[:, None]) * torch.cos(y[:, None]) * torch.exp(-2 * mu * t[:, None])
    return sln


def exact_func_p(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = -1 / 4 * (torch.cos(2 * x[:, None]) + torch.cos(2 * y[:, None])) * torch.exp(-4 * mu * t[:, None])
    return sln


def navier_stokes_2d_TG_vortex_experiment(grid_res):
    exp_dict_list_u, exp_dict_list_v, exp_dict_list_p = [], [], []

    x_min, x_max = 0, 2 * torch.pi
    y_min, y_max = 0, 2 * torch.pi
    t_max = 2

    pde_dim_in = 3
    pde_dim_out = 3

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], 11)

    domain_test = Domain()
    domain_test.variable('x', [x_min, x_max], grid_res)
    domain_test.variable('y', [y_min, y_max], grid_res)
    domain_test.variable('t', [0, t_max], 11)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: -torch.cos(grid[:, 0]) * torch.sin(grid[:, 1]),
                         var=0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: torch.sin(grid[:, 0]) * torch.cos(grid[:, 1]),
                         var=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: -1 / 4 * (torch.cos(2 * grid[:, 0]) + torch.sin(2 * grid[:, 1])),
                         var=2)

    # Boundary conditions ##############################################################################################

    # u-function

    # u(x_min, y, t) = u(x_max, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                         {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}],
                        var=0)
    # u(x, y_min, t) = u(x, y_max, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                         {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}],
                        var=0)

    # v-function

    # v(x_min, y, t) = v(x_max, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                         {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}],
                        var=1)
    # v(x, y_min, t) = v(x, y_max, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                         {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}],
                        var=1)

    # p-function

    # p(x_min, y, t) = p(x_max, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                         {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}],
                        var=2)
    # p(x, y_min, t) = p(x, y_max, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                         {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}],
                        var=2)

    equation = Equation()

    # Operator 1: u_x + v_y = 0

    NS_1 = {
        'du/dx':
            {
                'coeff': 1,
                'term': [0],
                'pow': 1,
                'var': 0
            },
        'dv/dy':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 1
            }
    }

    # Operator 2: u_t + u * u_x + v * u_y + 1 / ro * p_x - mu * (u_xx + u_yy) = 0

    NS_2 = {
        'du/dt':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 0
            },
        'u * du/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        'v * du/dy':
            {
                'coeff': 1,
                'term': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 0]
            },
        '1/ro * dp/dx':
            {
                'coeff': 1 / ro,
                'term': [0],
                'pow': 1,
                'var': 2
            },
        '-mu * d2u/dx2':
            {
                'coeff': -mu,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-mu * d2u/dy2':
            {
                'coeff': -mu,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    # Operator 3: v_t + u * v_x + v * v_y + 1 / ro * p_y - mu * (v_xx + v_yy) = 0

    NS_3 = {
        'dv/dt':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 1
            },
        'u * dv/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 1]
            },
        'v * dv/dy':
            {
                'coeff': 1,
                'term': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 1]
            },
        '1/ro * dp/dy':
            {
                'coeff': 1 / ro,
                'term': [1],
                'pow': 1,
                'var': 2
            },
        '-mu * d2v/dx2':
            {
                'coeff': -mu,
                'term': [0, 0],
                'pow': 1,
                'var': 1
            },
        '-mu * d2v/dy2':
            {
                'coeff': -mu,
                'term': [1, 1],
                'pow': 1,
                'var': 1
            }
    }

    equation.add(NS_1)
    equation.add(NS_2)
    equation.add(NS_3)

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
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model_layers = [pde_dim_in, neurons, neurons, neurons, neurons, neurons, neurons, pde_dim_out]

    grid = domain.build('NN').to(device)
    grid_test = domain_test.build('NN').to(device)

    model = Model(net, domain, equation, boundaries)

    
    model.compile('autograd', lambda_operator=1, lambda_bound=100)
    u_exact_test = exact_func_p(grid_test).reshape(-1)

    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_long_time_img')

    cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         info_string_every=10,
                                         randomize_parameter=1e-5)

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
        "tolerance": 0, 
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit" : 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "exact_solution": [exact_func_p, exact_func_u, exact_func_v],
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "lr": 1e-3,
        "exp": experiment,
    }

    # backup_params = {
    #     "experiment_key" : "b0dae86c42924e4484b8bd194e2d58d9",
    # }
    backup_params = None
    experiment.log_parameters(rl_agent_params)
    # experiment.log_parameters(backup_params)

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
                backup_params=backup_params)



if __name__ == "__main__":
    grid_res = 50

    navier_stokes_2d_TG_vortex_experiment(grid_res)