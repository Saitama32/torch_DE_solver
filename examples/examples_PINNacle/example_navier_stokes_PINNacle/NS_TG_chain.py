from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_heat_2d_tolerance",
  workspace="saitama32"
)


import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

k = 1
ro = 1
mu = 2 * torch.pi / 100

x_min, x_max = 0, 2 * torch.pi
y_min, y_max = 0, 2 * torch.pi
t_max = 2
grid_res = 100

def NS_TG_chain():

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    x = domain.variable_dict['x']
    y = domain.variable_dict['y']
    t = domain.variable_dict['t']

    boundaries = Conditions()

    # Initial conditions (TG vortex) ###################################################################################

    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                        value=torch.sin(k * x) * torch.cos(k * y),
                        var=0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                        value=-torch.cos(k * x) * torch.sin(k * y),
                        var=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                        value=-ro / 4 * (torch.cos(2 * k * x) + torch.cos(2 * k * y)),
                        var=2)

    # Boundary conditions for u-function ###############################################################################

    # u(x_min, y, t) = sin(x)
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=torch.sin(x), var=0)
    # u(x_max, y, t) = -sin(x)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=-torch.sin(x), var=0)
    # u(x, y_min, t) = 0
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=0, var=0)
    # u(x, y_max, t) = 0
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=0, var=0)

    # Boundary conditions for v-function ###############################################################################

    # v(x_min, y, t) = 0
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=1)
    # v(x_max, y, t) = 0
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=1)
    # v(x, y_min, t) = -sin(y)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=-torch.sin(y), var=1)
    # v(x, y_max, t) = sin(y)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=torch.sin(y), var=1)

    # Boundary conditions for p-function ###############################################################################

    # p(x, y_max, t) = 0
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=2)
    # v(x, y_max, t) = 0
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=2)

    equation = Equation()

    # operator 1: # operator: u_x + v_y = 0
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

    # operator 2: u * u_x + v * u_y + 1 / ro * p_x - mu * (u_xx + u_yy) = 0
    NS_2 = {
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

    # operator 3: u * v_x + v * v_y + 1 / ro * p_y - mu * (v_xx + v_yy) = 0
    NS_3 = {
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
        torch.nn.Linear(3, neurons),
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
        torch.nn.Linear(neurons, 3)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1000, tol=0.1)

    cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                        loss_window=100,
                                        no_improvement_patience=1000,
                                        patience=10,
                                        info_string_every=5,
                                        randomize_parameter=1e-5)

    img_dir = os.path.join(os.path.dirname(__file__), 'navier_stokes_2d_TG_vortex_img')

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
        "tolerance": 0.0410, 
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
