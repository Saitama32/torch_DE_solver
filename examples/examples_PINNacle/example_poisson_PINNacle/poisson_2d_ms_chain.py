from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_poisson_2d_ms_pinnacle_tolerance",
  workspace="saitama32"
)

import torch
import os
import sys
import numpy as np
import time

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
    "description": "farm_transitions_poisson_2d_ms_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_manyarea.npy"))
datapath_a_cof = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_a_coef.npy"))
datapath_f_cof = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_f_coef.npy"))



def poisson_2d_many_subdomains_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -10, 10
    y_min, y_max = -10, 10

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    domain_test = Domain()
    grid_test_res = 80
    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('y', [y_min, y_max], grid_test_res)

    split = (5, 5)
    freq = 2
    block_size = np.array([(x_max - x_min + 2e-5) / split[0], (y_max - y_min + 2e-5) / split[1]])

    a_cof = np.load(datapath_a_cof)
    f_cof = np.load(datapath_f_cof).reshape(split[0], split[1], freq, freq)

    boundaries = Conditions()

    # Operator: u + du/dn = 0

    def bop_generation(func_coeff, deriv_coeff, deriv_dim):
        bop = {
            'u':
                {
                    'coeff': func_coeff,
                    'term': [None],
                    'pow': 1
                },
            'du/dn':
                {
                    'coeff': deriv_coeff,
                    'term': [deriv_dim],
                    'pow': 1
                }
        }
        return bop

    bop_x_min = bop_generation(1, -1, 0)
    boundaries.robin({'x': x_min, 'y': [y_min, y_max]}, operator=bop_x_min, value=lambda grid: -grid[:, 1])

    bop_x_max = bop_generation(1, 1, 0)
    boundaries.robin({'x': x_max, 'y': [y_min, y_max]}, operator=bop_x_max, value=lambda grid: -grid[:, 1])

    bop_y_min = bop_generation(1, -1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_min}, operator=bop_y_min, value=lambda grid: -grid[:, 1])

    bop_y_max = bop_generation(1, 1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_max}, operator=bop_y_max, value=lambda grid: -grid[:, 1])

    def compute_domain(grid):
        reduced_x = (grid - np.array([x_min, y_min]) + 1e-5)
        dom = np.floor(reduced_x / block_size).astype("int32")
        res = reduced_x - dom * block_size
        return dom, res

    def compute_a_coeff(grid):
        dom, _ = compute_domain(grid)
        return a_cof[dom[0], dom[1]]

    a_coeff = np.vectorize(compute_a_coeff, signature="(2)->()")

    def compute_forcing_term(grid):
        dom, res = compute_domain(grid)

        def f_fn(coef):
            ans = coef[0, 0]
            for i in range(coef.shape[0]):
                for j in range(coef.shape[1]):
                    tmp = np.sin(np.pi * np.array((i, j)) * (res / block_size))
                    ans += coef[i, j] * tmp[0] * tmp[1]
            return ans

        return f_fn(f_cof[dom[0], dom[1]])

    forcing_term = np.vectorize(compute_forcing_term, signature="(2)->()")

    def get_a_coeff(grid):
        device_origin = grid.device
        grid = grid.detach().cpu()
        return torch.Tensor(a_coeff(grid)).unsqueeze(dim=-1).to(device_origin)

    def get_forcing_term(grid):
        device_origin = grid.device
        grid = grid.detach().cpu()
        return torch.Tensor(forcing_term(grid)).unsqueeze(dim=-1).to(device_origin)

    equation = Equation()

    # Operator: −∇(a(x)∇u) = f(x, y)

    poisson = {
        'a * d2u/dx2':
            {
                'coeff': get_a_coeff,
                'term': [0, 0],
                'pow': 1,
            },
        'a * d2u/dy2':
            {
                'coeff': get_a_coeff,
                'term': [1, 1],
                'pow': 1,
            },
        'f(x, y)':
            {
                'coeff': get_forcing_term,
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

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_many_subdomains_img')

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
        "n_transitions_reinit" : 1000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "exact_solution": datapath,
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "lr": 5e-4,
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

    return exp_dict_list


if __name__ == "__main__":
    grid_res = 100

    exp_dict_list = poisson_2d_many_subdomains_experiment(grid_res)
