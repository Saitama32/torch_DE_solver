from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_heat_2d_random_tolerance",
  workspace="saitama32"
)

import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "farm_transitions_Heat_2d_1d_basic_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

eps = 1.0

C_ring = torch.tensor([0.123456, 0.654321, 0.345612, 0.216543, 0.561234, 0.432165], dtype=torch.float32)
factor_ring = 10 ** 4
U_SCALE = float(10 ** 4)


def enforce_hermitian(kx, ky, h):
    k_to_idx = {(int(kx[i].item()), int(ky[i].item())): i for i in range(len(kx))}

    visited = set()

    for i in range(len(kx)):
        k = (kx[i].item(), ky[i].item())
        if k in visited:
            continue

        k_neg = (-k[0], -k[1])
        j = k_to_idx.get(k_neg, None)

        if j is None:
            continue

        visited.add(k)
        visited.add(k_neg)

        if i == j:
            h[i] = torch.real(h[i].clone())
            continue

        avg = 0.5 * (h[i] + torch.conj(h[j]))
        h[i] = avg
        h[j] = torch.conj(avg)

    return h, k_to_idx


def make_gaussian_init(N=10, seed=None, device="cpu"):
    """
    Реализация начальных условий heat_random из статьи (формула (33)):
      - N = 10 (kx,ky in {-5,...,4})
      - кольца n=1..6 с C_ring
      - ĝ(k)=1e4*sqrt(C_n/H(n))*h(k), H(n)=sum_{ring n} |h(k)|^2
      - ĝ(k)=0 при |k|>=13/2
      - h(-k)=conj(h(k)) => g(x) вещественная
    """
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    # N=10 => [-5, -4, ..., 4]
    kx_range = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)
    ky_range = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)

    kx_grid, ky_grid = torch.meshgrid(kx_range, ky_range, indexing="ij")
    kx = kx_grid.flatten()
    ky = ky_grid.flatten()

    k_sq = kx ** 2 + ky ** 2
    abs_k = torch.sqrt(k_sq)

    # h(k) ~ complex Gaussian
    real = torch.randn(len(kx), device=device)
    imag = torch.randn(len(kx), device=device)
    h = torch.complex(real, imag)

    # enforce h(-k)=conj(h(k))
    h, k_to_idx = enforce_hermitian(kx, ky, h)

    g_hat = torch.zeros_like(h)
    eps_small = 1e-12

    # кольца n=1..6
    for n in range(1, 7):
        mask = (abs_k >= n - 0.5) & (abs_k < n + 0.5)
        if mask.sum() == 0:
            continue

        H_n = torch.sum(torch.abs(h[mask]) ** 2).clamp_min(eps_small)
        scale = factor_ring * torch.sqrt(C_ring[n - 1] / H_n)
        g_hat[mask] = scale * h[mask]

    # cutoff: |k| >= 13/2
    g_hat[abs_k >= 6.5] = 0.0

    g_hat = g_hat.to(dtype=torch.complex64, device=device)

    def init_func(grid):
        x = grid[:, 0:1]
        y = grid[:, 1:2]

        device_grid = grid.device
        kx_local = kx.to(device_grid).unsqueeze(0)
        ky_local = ky.to(device_grid).unsqueeze(0)
        g_local = g_hat.to(device_grid)

        phase = x @ kx_local + y @ ky_local

        g_r = torch.real(g_local).unsqueeze(1)  # (M,1)
        g_i = torch.imag(g_local).unsqueeze(1)  # (M,1)

        u = phase.cos().matmul(g_r) - phase.sin().matmul(g_i)     # (N,1)
        return u / U_SCALE

    def exact_func(grid):
        x = grid[:, 0:1]
        y = grid[:, 1:2]
        t = grid[:, 2:3]

        device_grid = grid.device
        kx_local = kx.to(device_grid).unsqueeze(0)
        ky_local = ky.to(device_grid).unsqueeze(0)
        k_sq_local = k_sq.to(device_grid).unsqueeze(0)
        g_local = g_hat.to(device_grid)

        phase = x @ kx_local + y @ ky_local
        decay = torch.exp(-eps * t @ k_sq_local)

        g_r = torch.real(g_local).unsqueeze(1)
        g_i = torch.imag(g_local).unsqueeze(1)

        cos_part = phase.cos() * decay
        sin_part = phase.sin() * decay

        u = cos_part.matmul(g_r) - sin_part.matmul(g_i)  # (N,1)
        return u / U_SCALE

    return init_func, exact_func


def heat_2d_gaussian_init_experiment(grid_res, seed=None):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    y_min, y_max = 0, 2 * torch.pi
    t_max = 0.01

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], 6)

    domain_test = Domain()
    grid_test_res = 80

    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('y', [y_min, y_max], grid_test_res)
    domain_test.variable('t', [0, t_max], 6)

    boundaries = Conditions()

    init_func, exact_func = make_gaussian_init(N=10, seed=seed, device='cuda')

    # Initial condition ################################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: init_func(grid).to(grid.device))

    # Boundary conditions ###################################################################################

    # u(0, y, t) = u(2*pi, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                         {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}])

    # u(x, 0, t) = u(x, 2*pi, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                         {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}])

    # Operator: du/dt -  epsilon * (u_xx + u_yy) = 0

    equation = Equation()

    heat_LT = {
        'du/dt**1': {
            'coeff': 1,
            'term': [2],
            'pow': 1,
            'var': 0
        },
        '-epsilon * d2u/dx2**1': {
            'coeff': -eps,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
        '-epsilon * d2u/dy2**1': {
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
    u_exact_test = exact_func(grid_test).reshape(-1)
    
    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_gaussian_init_img')


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         randomize_parameter=1e-4,
                                         info_string_every=10)

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
        "tolerance": 0.01235231757164,
        "prev_tol": 0.0,
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

    return exp_dict_list

if __name__ == "__main__":
    grid_res = 100

    exp_dict_list = heat_2d_gaussian_init_experiment(grid_res)