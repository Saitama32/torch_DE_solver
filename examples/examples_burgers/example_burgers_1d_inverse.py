import torch
import numpy as np
import scipy
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, inverse_task
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import parameter_registr

solver_device('cuda')

domain = Domain()

domain.variable('x', [-1, 1], 60, dtype='float64')
domain.variable('t', [0, 1], 60, dtype='float64')

boundaries = Conditions()

data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__), '../wolfram_sln/Burgers.mat')))

x = torch.tensor(data['x']).reshape(-1)
t = torch.tensor(data['t']).reshape(-1)

usol = data['usol']

bnd1 = torch.cartesian_prod(x, t).float()
bndval1 = torch.tensor(usol).reshape(-1, 1)

id_f = np.random.choice(len(bnd1), 2000, replace=False)

bnd1 = bnd1[id_f]
bndval1 = bndval1[id_f]

boundaries.data(bnd=bnd1, operator=None, value=bndval1)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
)

parameters = {'lam1': 2., 'lam2': 0.2}  # true parameters: lam1 = 1, lam2 = -0.01*pi

parameter_registr(net, parameters)

equation = Equation()

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
            'coeff': net.lam1,
            'u*du/dx': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 0]
        },
    '-mu*d2u/dx2':
        {
            'coeff': net.lam2,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

equation.add(burgers_eq)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

img_dir = os.path.join(os.path.dirname(__file__), 'burgers_eq_img')

cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=3,
                                     abs_loss=1e-5,
                                     randomize_parameter=1e-5,
                                     info_string_every=5000)

cb_plots = plot.Plots(save_every=5000, print_every=None, img_dir=img_dir)

cb_params = inverse_task.InverseTask(parameters=parameters, info_string_every=5000)

optimizer = Optimizer('Adam', {'lr': 1e-4})

model.train(optimizer, 25e3, save_model=False, callbacks=[cb_es, cb_plots, cb_params])
