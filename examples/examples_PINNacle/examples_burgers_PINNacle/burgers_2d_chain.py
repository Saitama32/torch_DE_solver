from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="api_key",
  project_name="project_name",
  workspace="workspace"
)
import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data
from tedeous.utils import init_data

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "farm_transitions_Heat_2d_long_time_basic_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/burgers2d_0.npy"))
data_init_u = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/burgers2d_init_u_0.npy"))
data_init_v = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/burgers2d_init_v_0.npy"))
# datapath = "../../PINNacle_data/burgers2d_0.npy"
# data_init_u = "../../PINNacle_data/burgers2d_init_u_0.npy"
# data_init_v = "../../PINNacle_data/burgers2d_init_v_0.npy"

mu = 0.001


def init_w(x, y, size, L):
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    result = torch.zeros(size)
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            new_component = a[i, j] * torch.sin(2 * torch.pi * (i * x + j * y)) \
                      + b[i, j] * torch.cos(2 * torch.pi * (i * x + j * y))
            result += new_component

    return result


def init_u(grid):
    x, y = grid[:, 0], grid[:, 1]
    size = int(len(x) ** 1)
    L = int(torch.max(x))
    c_u = torch.randn(size)
    return 2 * init_w(x, y, size, L) + c_u


def init_v(grid):
    x, y = grid[:, 0], grid[:, 1]
    size = int(len(x) ** 1)
    L = int(torch.max(x))
    c_v = torch.randn(size)
    return 2 * init_w(x, y, size, L) + c_v


def burgers_2d_coupled_experiment(grid_res):
    exp_dict_list_u, exp_dict_list_v = [], []

    x_min, L = 0, 4
    y_min, L = 0, 4
    T = 1
    # grid_res = 20

    pde_dim_in = 3
    pde_dim_out = 2

    domain = Domain()
    domain.variable('x', [x_min, L], grid_res)
    domain.variable('y', [y_min, L], grid_res)
    domain.variable('t', [0, T], grid_res)

    domain_test = Domain()
    grid_test_res = 80
    domain_test.variable('x', [x_min, L], grid_test_res)
    domain_test.variable('y', [y_min, L], grid_test_res)
    domain_test.variable('t', [0, T], grid_test_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # # With use custom functions for IC
    #
    # # u(x, y, 0)
    # boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_u, var=0)
    #
    # # v(x, y, 0)
    # boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_v, var=1)

    # With use IC data

    init_u_data = lambda grid: init_data(grid[:, :2], data_init_u)
    init_v_data = lambda grid: init_data(grid[:, :2], data_init_v)

    # u(x, y, 0)
    boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_u_data, var=0)

    # v(x, y, 0)
    boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_v_data, var=1)

    # Boundary conditions ##############################################################################################

    # u(0, y, t) = u(L, y, t)
    boundaries.periodic([{'x': 0, 'y': [0, L], 't': [0, T]}, {'x': L, 'y': [0, L], 't': [0, T]}], var=0)

    # u(x, 0, t) = u(x, L, t)
    boundaries.periodic([{'x': [0, L], 'y': 0, 't': [0, T]}, {'x': [0, L], 'y': L, 't': [0, T]}], var=0)

    # v(0, y, t) = v(L, y, t)
    boundaries.periodic([{'x': 0, 'y': [0, L], 't': [0, T]}, {'x': L, 'y': [0, L], 't': [0, T]}], var=1)

    # v(x, 0, t) = v(x, L, t)
    boundaries.periodic([{'x': [0, L], 'y': 0, 't': [0, T]}, {'x': [0, L], 'y': L, 't': [0, T]}], var=1)

    equation = Equation()

    # Operator 1: u_t + u * u_x + v * u_y - mu * (u_xx + u_yy) = 0

    burgers_u = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [2],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1.,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '+v*du/dy':
            {
                'coeff': 1.,
                'u*du/dy': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-mu*d2u/dy2':
            {
                'coeff': -mu,
                'd2u/dy2': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    # Operator 2: v_t + u * v_x + v * v_y - mu * (v_xx + v_yy) = 0

    burgers_v = {
        'dv/dt**1':
            {
                'coeff': 1.,
                'dv/dt': [2],
                'pow': 1,
                'var': 1
            },
        '+u*dv/dx':
            {
                'coeff': 1.,
                'u*dv/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 1]
            },
        '+v*dv/dy':
            {
                'coeff': 1.,
                'v*dv/dy': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 1]
            },
        '-mu*d2v/dx2':
            {
                'coeff': -mu,
                'd2v/dx2': [0, 0],
                'pow': 1,
                'var': 1
            },
        '-mu*d2v/dy2':
            {
                'coeff': -mu,
                'd2v/dy2': [1, 1],
                'pow': 1,
                'var': 1
            }
    }

    equation.add(burgers_u)
    equation.add(burgers_v)

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

    u_exact_test = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out).reshape(-1)
    
    
    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         randomize_parameter=1e-4,
                                         info_string_every=10)


    img_dir = os.path.join(os.path.dirname(__file__), 'burgers_2d_coupled_img')

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
        "tolerance": 0.0,
        "prev_tol": 0.0,
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit" : 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "exact_solution": datapath,
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

    return  exp_dict_list_u, exp_dict_list_v


if __name__ == "__main__":
    grid_res = 100

    exp_dict_list = burgers_2d_coupled_experiment(grid_res)