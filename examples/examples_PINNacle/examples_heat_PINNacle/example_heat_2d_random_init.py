import torch
# torch.set_printoptions(threshold=10_000, linewidth=200, edgeitems=50)
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

solver_device('gpu')

eps = 1.0

C_ring = torch.tensor([0.123456, 0.654321, 0.345612, 0.216543, 0.561234, 0.432165], dtype=torch.float32)
factor_ring = 10 ** 4
U_SCALE = float(1e4)


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


import torch.nn as nn


class HardICPeriodicNet(nn.Module):
    """
    u(x,y,t) = u0(x,y) + (t/T) * base([sin x, cos x, sin y, cos y, t])

    - периодичность по x,y встроена через sin/cos
    - начальное условие при t=0 выполняется точно
    - совместимость с tedeous plot-callback: net[-1].out_features
    """
    def __init__(self, base: nn.Sequential, u0_func, T: float, out_dim: int = 1):
        super().__init__()
        self.base = base
        self.u0_func = u0_func
        self.T = float(T)
        self.width_out = [out_dim]  # fallback для некоторых частей tedeous

    def __getitem__(self, idx):
        return self.base[idx]

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: [N,3] = (x,y,t)
        x = grid[:, 0:1]
        y = grid[:, 1:2]
        t = grid[:, 2:3]

        feat = torch.cat(
            [torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y), t],
            dim=1
        )

        # u0 берём на тех же (x,y,t), но твой init_func игнорирует t — это ок
        u0 = self.u0_func(grid).to(grid.device)

        corr = self.base(feat)

        # hard IC: при t=0 добавка = 0
        return u0 + (t / self.T) * corr



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

    bconds = boundaries.build(domain.variable_dict)
    # print(bconds)
    # ic = [b for b in bconds if b["type"] == "dirichlet"]
    # print("IC dict:", ic)
    # print("IC bnd shape:", ic["bnd"].shape)  # должно быть [961,3] при grid_res=30
    # print("IC val shape:", init_func(ic["bnd"].to("cuda")).shape)  # [961,1]
    # print("IC bnd:", ic["bnd"])
    # print("IC val:", init_func(ic["bnd"].to("cuda")))

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
    pde_dim_out = 1

    # base-net принимает 5 входов: sinx, cosx, siny, cosy, t
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

    # base = torch.nn.Sequential(
    #     torch.nn.Linear(5, neurons),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(neurons, neurons),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(neurons, neurons),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(neurons, neurons),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(neurons, pde_dim_out)
    # )
    # # Обёртка: жёсткое IC + периодичность
    # net = HardICPeriodicNet(
    #     base=base,
    #     u0_func=lambda grid: init_func(grid),
    #     T=t_max,
    #     out_dim=pde_dim_out
    # )


    # for m in net.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.xavier_normal_(m.weight)
    #         torch.nn.init.zeros_(m.bias)

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_gaussian_init_img')

    # cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    # cb_plots = plot.Plots(save_every=500,
    #                       print_every=None,
    #                       img_dir=img_dir,
    #                       img_dim='2d',
    #                       scatter_flag=False,
    #                       plot_axes=[0, 1],
    #                       fixed_axes=[2],
    #                       n_samples=4,
    #                       img_rows=2,
    #                       img_cols=2)

    optimizer = Optimizer('LBFGS', {'lr': 1})

    callbacks = [cb_es]

    model.train(optimizer, 100, save_model=False, callbacks=callbacks)

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda').eval()

    with torch.no_grad():
        grid = domain.build('NN').to('cuda')
        pred = net(grid)
        exact_on_grid = exact_func(grid)

        # absolute RMSE on full grid
        grid_rmse = torch.sqrt(torch.mean((pred - exact_on_grid) ** 2))

        # relative RMSE on full grid (RMS-normalized)
        grid_rms_exact = torch.sqrt(torch.mean(exact_on_grid ** 2)) + 1e-12
        grid_rel_rmse = (grid_rmse / grid_rms_exact)

        # --- boundary exact metrics ---
        bconds = boundaries.build(domain.variable_dict)

        boundary_err_sq = []   # (u_pred - u_exact)^2 over boundary points
        boundary_ex_sq = []    # (u_exact)^2 over boundary points

        for b in bconds:
            btype = b.get("type", None)

            if btype == "periodic":
                bnd_left, bnd_right = b["bnd"]
                bnd_left = bnd_left.to('cuda')
                bnd_right = bnd_right.to('cuda')

                u_left = net(bnd_left)
                u_right = net(bnd_right)

                ex_left = exact_func(bnd_left)
                ex_right = exact_func(bnd_right)

                boundary_err_sq.append((u_left - ex_left).reshape(-1) ** 2)
                boundary_err_sq.append((u_right - ex_right).reshape(-1) ** 2)

                boundary_ex_sq.append(ex_left.reshape(-1) ** 2)
                boundary_ex_sq.append(ex_right.reshape(-1) ** 2)

            else:
                bnd = b["bnd"].to('cuda')
                u = net(bnd)
                ex = exact_func(bnd)

                boundary_err_sq.append((u - ex).reshape(-1) ** 2)
                boundary_ex_sq.append(ex.reshape(-1) ** 2)

        boundary_exact_rmse = torch.sqrt(torch.mean(torch.cat(boundary_err_sq)))
        boundary_rms_exact = torch.sqrt(torch.mean(torch.cat(boundary_ex_sq))) + 1e-12
        boundary_exact_rel_rmse = boundary_exact_rmse / boundary_rms_exact

    print("GridExact RMSE:", grid_rmse.item())
    print("GridExact RelRMSE:", grid_rel_rmse.item())
    print("BoundaryExact RMSE:", boundary_exact_rmse.item())
    print("BoundaryExact RelRMSE:", boundary_exact_rel_rmse.item())



    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        "GridExact RMSE:": grid_rmse.item(),
        "GridExact RelRMSE:": grid_rel_rmse.item(),
        'BoundaryExact RMSE': boundary_exact_rmse.item(),
        "BoundaryExact RelRMSE:": boundary_exact_rel_rmse.item(),
        'type': 'heat_2d_gaussian_init',
        'cache': True
    })

    return exp_dict_list


nruns = 1

exp_dict_list = []
for grid_res in range(100, 1001, 100):
    for r in range(nruns):
        exp_dict_list.append(heat_2d_gaussian_init_experiment(grid_res, seed=r))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_gaussian_init_experiment.csv')
