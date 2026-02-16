from comet_ml import start
from comet_ml.integration.pytorch import log_model


import torch
import os
import sys
import numpy as np
import time
import random
import tempfile
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
from tedeous.error_calc_utils import boundary_report


device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_manyarea.npy"))
datapath_a_cof = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_a_coef.npy"))
datapath_f_cof = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_f_coef.npy"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_key",
        type=str,
        default=None,
        help="Comet experiment key for backup / resume"
    )
    parser.add_argument(
        "--exp_key",
        type=str,
        default=None,
        help="Comet experiment key for comparison runs"
    )
    return parser.parse_args()

def poisson_2d_many_subdomains_experiment(grid_res, log_key=None, exp_key=None):
    if log_key == "True":
        log_key = True
    elif log_key == "False":
        log_key = False    

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

    def exact_fn(pts):
        return exact_solution_data(
            pts,
            datapath,
            pde_dim_in,
            pde_dim_out,
            t_dim_flag=("t" in list(domain.variable_dict.keys())),
        )

    split = (5, 5)
    freq = 2
    block_size = np.array([(x_max - x_min + 2e-5) / split[0], (y_max - y_min + 2e-5) / split[1]])

    a_cof = np.load(datapath_a_cof)
    f_cof = np.load(datapath_f_cof).reshape(split[0], split[1], freq, freq)

    # --- torch tensors for coefficients (keep on GPU) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    a_t = torch.as_tensor(a_cof, dtype=torch.float32, device=device)  # shape зависит от файла
    f_t = torch.as_tensor(f_cof, dtype=torch.float32, device=device)  # (sx, sy, freq, freq)

    dx = (x_max - x_min) / split[0]
    dy = (y_max - y_min) / split[1]

    sx, sy = split

    def dom_and_local(grid: torch.Tensor):
        """
        grid: (N,2) on GPU
        returns:
        ix, iy: (N,) long indices of block
        rx, ry: (N,1) local coords in [0,1] inside the block
        """
        x = grid[:, 0]
        y = grid[:, 1]

        ix = torch.floor((x - x_min) / dx).long()
        iy = torch.floor((y - y_min) / dy).long()

        # clamp to be safe on boundary
        ix = ix.clamp(0, sx - 1)
        iy = iy.clamp(0, sy - 1)

        x0 = x_min + ix.float() * dx
        y0 = y_min + iy.float() * dy

        rx = ((x - x0) / dx).unsqueeze(1)  # (N,1)
        ry = ((y - y0) / dy).unsqueeze(1)  # (N,1)

        return ix, iy, rx, ry
    
    def a_and_grads(grid: torch.Tensor):
        """
        Returns:
        a  (N,1)
        ax (N,1) = ∂a/∂x
        ay (N,1) = ∂a/∂y
        """
        ix, iy, rx, ry = dom_and_local(grid)

        # Case A: a = a0 + ax_hat*rx + ay_hat*ry  (linear in local coords)
        # a_t shape: (sx,sy,3) where [:,:,0]=a0, [:,:,1]=ax_hat, [:,:,2]=ay_hat
        if a_t.ndim == 3 and a_t.shape[-1] == 3:
            p = a_t[ix, iy]  # (N,3)
            a0 = p[:, 0:1]
            ax_hat = p[:, 1:2]
            ay_hat = p[:, 2:3]

            a = a0 + ax_hat * rx + ay_hat * ry
            ax = ax_hat / dx
            ay = ay_hat / dy
            return a, ax, ay

        # Case B (optional): bilinear on local square:
        # a = a00 + a10*rx + a01*ry + a11*rx*ry
        if a_t.ndim == 3 and a_t.shape[-1] == 4:
            p = a_t[ix, iy]  # (N,4)
            a00 = p[:, 0:1]
            a10 = p[:, 1:2]
            a01 = p[:, 2:3]
            a11 = p[:, 3:4]

            a = a00 + a10*rx + a01*ry + a11*rx*ry
            ax = (a10 + a11*ry) / dx
            ay = (a01 + a11*rx) / dy
            return a, ax, ay

        # Fallback: piecewise-constant
        # a_t shape (sx,sy)
        a = a_t[ix, iy].unsqueeze(1)
        ax = torch.zeros_like(a)
        ay = torch.zeros_like(a)
        return a, ax, ay


    def get_a(grid):
        a, _, _ = a_and_grads(grid.to(device))
        return a


    def get_ax(grid):
        _, ax, _ = a_and_grads(grid.to(device))
        return ax


    def get_ay(grid):
        _, _, ay = a_and_grads(grid.to(device))
        return ay
    
    i_idx = torch.arange(freq, device=device, dtype=torch.float32).view(1, freq, 1)
    j_idx = torch.arange(freq, device=device, dtype=torch.float32).view(1, 1, freq)

    def get_f(grid: torch.Tensor):
        grid = grid.to(device)
        ix, iy, rx, ry = dom_and_local(grid)

        coef = f_t[ix, iy]  # (N,freq,freq)

        # sin(pi*i*rx)*sin(pi*j*ry)
        sinx = torch.sin(torch.pi * i_idx * rx.view(-1,1,1))  # (N,freq,1)
        siny = torch.sin(torch.pi * j_idx * ry.view(-1,1,1))  # (N,1,freq)
        basis = sinx * siny                                   # (N,freq,freq)

        fval = torch.sum(coef * basis, dim=(1,2), keepdim=True)  # (N,1)
        return fval




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
    boundaries.robin({'x': x_min, 'y': [y_min, y_max]}, operator=bop_x_min, value=0.0)

    bop_x_max = bop_generation(1, 1, 0)
    boundaries.robin({'x': x_max, 'y': [y_min, y_max]}, operator=bop_x_max, value=0.0)

    bop_y_min = bop_generation(1, -1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_min}, operator=bop_y_min, value=0.0)

    bop_y_max = bop_generation(1, 1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_max}, operator=bop_y_max, value=0.0)


    equation = Equation()

    # Operator: −∇(a(x)∇u) = f(x, y)

    poisson = {
    '-a * d2u/dx2': {'coeff': lambda g: -get_a(g),  'term': [0, 0], 'pow': 1, 'var': 0},
    '-ax * du/dx':  {'coeff': lambda g: -get_ax(g), 'term': [0],    'pow': 1, 'var': 0},

    '-a * d2u/dy2': {'coeff': lambda g: -get_a(g),  'term': [1, 1], 'pow': 1, 'var': 0},
    '-ay * du/dy':  {'coeff': lambda g: -get_ay(g), 'term': [1],    'pow': 1, 'var': 0},

    '-f(x,y)':      {'coeff': lambda g: -get_f(g),  'term': [None], 'pow': 0, 'var': 0},
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

    optimizer = Optimizer('Adam', {'lr': 1e-3})

    model.train(optimizer, 300, save_model=True, callbacks=[cb_es])
    net = model.net.to(device)
    grid_test = grid_test.to(device)
    u_exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape(-1, 1)
    u_pred = net(grid)
    diff = u_exact - u_pred
    error_op_mse_train = torch.mean(diff ** 2)
    error_op_rmse_train = torch.sqrt(torch.mean(diff ** 2))
    variable_dict = domain.variable_dict
    bconds = boundaries.build(variable_dict)
    # boundary_err_sq = []
    # with torch.no_grad():
    #     for b in bconds:
    #         btype = b.get("type", None)

    #         # оставляем только сравнение u на границе
    #         if btype != "dirichlet" and btype != "periodic":
    #             continue

    #         if btype == "periodic":
    #             bnd_left, bnd_right = b["bnd"]
    #             for bnd in (bnd_left, bnd_right):
    #                 bnd = bnd.to(device)
    #                 u_pred = net(bnd)
    #                 u_ex = exact_solution_data(bnd, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape_as(u_pred)
    #                 boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)
    #         else:
    #             bnd = b["bnd"].to(device)
    #             u_pred = net(bnd)
    #             u_ex = exact_solution_data(bnd, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape_as(u_pred)
    #             boundary_err_sq.append((u_pred - u_ex).reshape(-1) ** 2)

    # error_bnd_mse_train = torch.mean(torch.cat(boundary_err_sq))


    # error_bnd_rmse_train = torch.sqrt(torch.mean(torch.cat(boundary_err_sq)))

    report = boundary_report(
        net=net,
        grid_for_dtype=grid,
        bconds=bconds,
        exact_solution_fn=exact_fn,
        mode="autograd",
        derivative_points=2,
    )
    error_bnd_mse_train = report["exact_u_mse"]
    error_bnd_rmse_train = report["exact_u_rmse"]
    print({k: float(v) for k, v in report.items()})

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
    report = boundary_report(
        net=net,
        grid_for_dtype=grid_test,
        bconds=bconds,
        exact_solution_fn=exact_fn,
        mode="autograd",
        derivative_points=2,
    )
    error_bnd_mse_test = report["exact_u_mse"]
    error_bnd_rmse_test = report["exact_u_rmse"]

  
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
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список

    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        exp_dict_list = poisson_2d_many_subdomains_experiment(grid_res)
