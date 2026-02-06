from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_poisson_2d_cg_opimization",
  workspace="saitama32"
)

import torch
import os
import sys
import time
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "farm_transitions_poisson_2d_cg_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)
datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_boltzmann2d.npy"))

mu_1 = 1
mu_2 = 4
k = 8
A = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_key",
        type=str,
        default=None,
        help="Comet experiment key for backup / resume"
    )
    return parser.parse_args()


def poisson_2d_irregular_geometry_experiment(grid_res, log_key=None):
    if log_key == "True":
        log_key = True
    elif log_key == "False":
        log_key = False    
    exp_dict_list = []

    x_min, x_max = -1, 1
    y_min, y_max = -1, 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    
    domain_test = Domain()
    grid_test_res = 80
    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('y', [y_min, y_max], grid_test_res)

    boundaries = Conditions()

    # Circle type of removed domains ###################################################################################

    removed_domains_lst = [
        {'circle': {'center': (0.5, 0.5), 'radius': 0.2}},
        {'circle': {'center': (0.4, -0.4), 'radius': 0.4}},
        {'circle': {'center': (-0.2, -0.7), 'radius': 0.1}},
        {'circle': {'center': (-0.6, 0.5), 'radius': 0.3}}
    ]

    # Boundary conditions ##############################################################################################

    # CSG boundaries

    boundaries.dirichlet({'circle': {'center': (0.5, 0.5), 'radius': 0.2}}, value=1)
    boundaries.dirichlet({'circle': {'center': (0.4, -0.4), 'radius': 0.4}}, value=1)
    boundaries.dirichlet({'circle': {'center': (-0.2, -0.7), 'radius': 0.1}}, value=1)
    boundaries.dirichlet({'circle': {'center': (-0.6, 0.5), 'radius': 0.3}}, value=1)

    # Non CSG boundaries

    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=0.2)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=0.2)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=0.2)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=0.2)

    def forcing_term(grid):
        x, y = grid[:, 0], grid[:, 1]
        return -A * (mu_1 ** 2 + mu_2 ** 2 + x ** 2 + y ** 2) * \
               torch.sin(mu_1 * torch.pi * x) * \
               torch.sin(mu_2 * torch.pi * y)

    equation = Equation()

    # Operator: -d2u/dx2 - d2u/dy2 + k ** 2 * u = f(x, y)

    poisson = {
        '-d2u/dx2':
            {
                'coeff': -1.,
                'term': [0, 0],
                'pow': 1,
            },
        '-d2u/dy2':
            {
                'coeff': -1.,
                'term': [1, 1],
                'pow': 1,
            },
        'k ** 2 * u':
            {
                'coeff': k ** 2,
                'term': [None],
                'pow': 1
            },
        'f(x, y)':
            {
                'coeff': forcing_term,
                'term': [None],
                'pow': 0
            }
    }

    equation.add(poisson)

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

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

    img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_irregular_geometry_img')

    u_exact_test = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out).reshape(-1)
    
    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         info_string_every=10,
                                         randomize_parameter=1e-4)

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
        "finetune_AE_model": False,
        "log_key": log_key
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
        "tolerance": 0.671255528252029,
        "prev_tol": 0.0,
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit" : 1000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "exact_solution": datapath,
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "lr": 5e-4,
        "exp": experiment,
        "log_key": log_key,
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

    return exp_dict_list



if __name__ == "__main__":
    args = parse_args()
    grid_res = 100

    exp_dict_list = poisson_2d_irregular_geometry_experiment(grid_res, log_key=args.log_key)






