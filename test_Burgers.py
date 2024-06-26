import torch
import numpy as np
from scipy.integrate import quad
import time
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tedeous'), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model_test import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device
from tedeous.models import mat_model
from tedeous.data_handler import save_data_and_graph


solver_device('cpu')

mode = 'autograd'

mu = 0.02 / np.pi

##Domain class for doamin initialization
domain = Domain()
domain.variable('x', [-1, 1], 20)
domain.variable('t', [0, 1], 20)

boundaries = Conditions()

##initial cond
x = domain.variable_dict['x']
boundaries.dirichlet({'x': [-1, 1], 't': 0}, value=-torch.sin(np.pi*x))

##boundary cond
boundaries.dirichlet({'x': -1, 't': [0, 1]}, value=0)

##boundary cond
boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)

equation = Equation()

## equation part
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



def exact(grid):
    mu = 0.02 / np.pi

    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t))[0] / quad(integrand2, -np.inf, np.inf, args=(x, t))[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)


## model part
for i in range(10):
    if mode in ('NN', 'autograd'):
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1)
        )
    else:
        net = mat_model(domain, equation)


    img_dir=os.path.join(os.path.dirname('tedeous/tutorials'), 'Burg_eq_img')


    model = Model(net, domain, equation, boundaries)

    model.compile(mode, lambda_operator=1, lambda_bound=10)

    cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                        loss_window=100,
                                        no_improvement_patience=1000,
                                        patience=3,
                                        randomize_parameter=1e-5,
                                        info_string_every=1000)

    cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)


# optimizer = Optimizer('Adam', {'lr': 1e-3})

# model.train(optimizer, 10000, save_model=False, callbacks=[cb_es, cb_plots])

# grid = domain.build(mode)

# u_exact = exact(grid).to('cpu')

# u_exact = check_device(u_exact).reshape(-1)

# u_pred = check_device(net(grid)).reshape(-1)

# error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

# print('RMSE_grad= ', error_rmse.item())


    optimizer = Optimizer('PSO', {'pop_size': 10,
                                'b': 0.5,
                                'c2': 0.05,
                                'variance': 5e-3,
                                'c_decrease': True,
                                'lr': 5e-3})

    tik = time.time()
    model.train(optimizer, 3000, info_string_every=100, save_model=False, callbacks=[cb_es, cb_plots])
    tak = time.time()

    grid = domain.build(mode)

    u_exact = exact(grid).to('cpu')

    u_exact = check_device(u_exact).reshape(-1)

    u_pred = check_device(net(grid)).reshape(-1)

    error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

    model.dict_of_learning['RMSE_pso'] = error_rmse.item()

    save_data_and_graph.save_data_and_graph(model.dict_of_learning,"Burg_eq", "PSO_old")

    print('RMSE_pso= ', error_rmse.item()) 
    print('Time_taken= ', tak-tik)