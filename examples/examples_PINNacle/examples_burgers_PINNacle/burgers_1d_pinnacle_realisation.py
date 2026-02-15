import torch
import os
import sys
import time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/burgers1d.npy"))

mu = 0.01 / np.pi


def burgers_1d_experiment(x_res, t_res):
    exp_dict_list = []

    x_min, x_max = -1, 1
    t_max = 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], x_res)
    domain.variable('t', [0, t_max], t_res)

    grid_test_res = 80
    domain_test = Domain()
    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('t', [0, t_max], grid_test_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # u(x, 0) = -sin(pi * x)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=lambda grid: -torch.sin(np.pi * grid[:, 0]))

    # Boundary conditions ##############################################################################################

    # u(x_min, t) = 0
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)

    # u(x_max, t) = 0
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

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
    grid_test = domain_test.build('NN').to(device)
    # data file contains time snapshots; enable time dimension handling
    u_exact_test = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1)

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


    optimizer = Optimizer('Adam', {'lr': 0.001})

    model.train(optimizer, 3000, callbacks=[cb_es])

    u_exact_train = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse_train = torch.sqrt(torch.mean((u_exact_train - net_predicted) ** 2))

    error_l2re_train = torch.sqrt(torch.sum(
    (u_exact_train - net(grid)) ** 2) / torch.sum(u_exact_train ** 2))
    print("error_rmse_train: ", error_rmse_train)
    print("error_l2re_train: ", error_l2re_train)

    u_exact_test = exact_solution_data(grid_test, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    net_predicted = net(grid_test)

    # evaluate on the test grid using the matching ground truth
    error_rmse_test = torch.sqrt(torch.mean((u_exact_test - net_predicted) ** 2))

    error_l2re_test = torch.sqrt(torch.sum(
    (u_exact_test - net(grid_test)) ** 2) / torch.sum(u_exact_test ** 2))
    print("error_rmse_test: ", error_rmse_test)
    print("error_l2re_test: ", error_l2re_test)

if __name__ == "__main__":
    x_res = 100
    t_res = 100

    exp_dict_list = burgers_1d_experiment(x_res, t_res)

