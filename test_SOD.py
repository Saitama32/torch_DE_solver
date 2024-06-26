import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import sys


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model_test import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer_test import Optimizer
from tedeous.device import solver_device, device_type
from tedeous.data_handler import save_data_and_graph

solver_device('cpu')

p_l = 1
v_l = 0
Ro_l = 1
gam_l = 1.4

p_r = 0.1
v_r = 0
Ro_r = 0.125
gam_r = 1.4

x0 = 0.5
h = 0.05

domain = Domain()
domain.variable('x', [0, 1], 21)
domain.variable('t', [0, 0.2], 21)

boundaries = Conditions()
## BOUNDARY AND INITIAL CONDITIONS
# p:0, v:1, Ro:2

def u0(x, x0):
  if x>x0:
    return [p_r, v_r, Ro_r]
  else:
    return [p_l, v_l, Ro_l]

# Initial conditions at t=0
x = domain.variable_dict['x']
u_init0 = np.zeros(x.shape[0])
u_init1 = np.zeros(x.shape[0])
u_init2 = np.zeros(x.shape[0])
j=0
for i in x:
  u_init0[j] = u0(i, x0)[0]
  u_init1[j] = u0(i, x0)[1]
  u_init2[j] = u0(i, x0)[2]
  j +=1

bndval1_0 = torch.from_numpy(u_init0)
bndval1_1 = torch.from_numpy(u_init1)
bndval1_2 = torch.from_numpy(u_init2)

boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_0, var=0)
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_1, var=1)
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_2, var=2)

#  Boundary conditions at x=0
boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=p_l, var=0)
boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=v_l, var=1)
boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=Ro_l, var=2)

# Boundary conditions at x=1
boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=p_r, var=0)
boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=v_r, var=1)
boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=Ro_r, var=2)

'''
gas dynamic system equations:
Eiler's equations system for Sod test in shock tube

'''

equation = Equation()

gas_eq1={
        'dro/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 2
        },
        'v*dro/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [1, 2]
        },
        'ro*dv/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [2, 1]
        }
     }
gas_eq2 = {
        'ro*dv/dt':
        {
            'coeff': 1,
            'term': [[None], [1]],
            'pow': [1, 1],
            'var': [2, 1]
        },
        'ro*v*dv/dx':
        {
            'coeff': 1,
            'term': [[None],[None], [0]],
            'pow': [1, 1, 1],
            'var': [2, 1, 1]
        },
        'dp/dx':
        {
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
     }
gas_eq3 =  {
        'dp/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        },
        'gam*p*dv/dx':
        {
            'coeff': gam_l,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 1]
        },
        'v*dp/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [1, 0]
        }

     }

equation.add(gas_eq1)
equation.add(gas_eq2)
equation.add(gas_eq3)

net = torch.nn.Sequential(
        torch.nn.Linear(2, 200),
        torch.nn.Tanh(),
        torch.nn.Linear(200, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 200),
        torch.nn.Tanh(),
        torch.nn.Linear(200, 3)
    )
start = time.time()

model =  Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=1000, h=h)

img_dir=os.path.join(os.path.dirname( __file__ ), 'SOD_NN_img')

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=100,
                                    no_improvement_patience=500,
                                    patience=5,
                                    randomize_parameter=1e-6,
                                    info_string_every=1000)

cb_plots = plot.Plots(save_every=1000, print_every= None, img_dir=img_dir)

optimizer = Optimizer('PSO', {'pop_size': 40,
                              'b': 0.5,
                              'c2': 0.05,
                              'variance': 5e-3,
                              'c_decrease': True,
                              'lr': 0})

model.train(optimizer, 10000, info_string_every=10, save_model=False, callbacks=[cb_es, cb_plots])

end = time.time()

print('Time taken = {}'.format(end - start))

solver_device('cpu')
device = device_type()

net = net.to(device)
grid = domain.build('NN')
grid = grid.to(device)

def exact(point):
  N = 100
  Pl = 1
  Pr = 0.1
  Rg = 519.4
  Gl = 1.4
  Gr = 1.4
  Tl = 273
  Tr = 248
  Rol = 1
  Ror = 0.125
  
  Cr = (Gr*Pr/Ror)**(1/2)
  Cl = (Gl*Pl/Rol)**(1/2)
  vl = 0
  vr = 0
  t = float(point[-1])
  x = float(point[0])
  x0 = 0
  x1 = 1
  xk = 0.5
      

  eps = 1e-5
  Pc1 = Pl/2
  vc1 = 0.2
  u = 1
  while u >= eps:
      Pc = Pc1
      vc = vc1
      f = vl + 2/(Gl-1)*Cl*(-(Pc/Pl)**((Gl-1)/(2*Gl))+1)-vc
      g = vr + (Pc-Pr)/(Ror*Cr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))**(1/2))-vc
      fp = -2/(Gl-1)*Cl*(1/Pl)**((Gl-1)/2/Gl)*(Gl-1)/2/Gl*Pc**((Gl-1)/(2*Gl)-1)
      gp = (1-(Pc-Pr)*(Gr+1)/(4*Gr*Pr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))))/(Ror*Cr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))**(1/2))
      fu = -1
      gu = -1
      Pc1 = Pc - (fu*g-gu*f)/(fu*gp-gu*fp)
      vc1 = vc - (f*gp-g*fp)/(fu-gp-gu*fp)
      u1 = abs((Pc-Pc1)/Pc)
      u2 = abs((vc-vc1)/vc)
      u = max(u1, u2)

  Pc = Pc1
  vc = vc1

  if x <= xk - Cl*t:
      p = Pl
      v = vl
      T = Tl
      Ro = Rol
  Roc = Rol/(Pl/Pc)**(1/Gl)
  if xk - Cl*t < x <= xk + (vc-(Gl*Pc/Roc)**(1/2))*t:
      Ca = (vl + 2 * Cl / (Gl - 1) + (xk - x) / t) / (1 + 2 / (Gl - 1))
      va = Ca - (xk - x) / t
      p = Pl*(Ca/Cl)**(2*Gl/(Gl-1))
      v = va
      Ro = Rol/(Pl/p)**(1/Gl)
      T = p/Rg/Ro
  if xk + (vc - (Gl * Pc / Roc) ** (1 / 2)) * t < x <= xk + vc * t:
      p = Pc
      Ro = Roc
      v = vc
      T = p / Rg / Ro
  D = vr + Cr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))**(1/2)
  if xk + vc * t < x <= xk+D*t:
      p = Pc
      v = vc
      Ro = Ror*((Gr+1)*Pc+(Gr-1)*Pr)/((Gr+1)*Pr+(Gr-1)*Pc)
      T = p/ Rg / Ro
  if xk+D*t < x:
      p = Pr
      v = vr
      Ro = Ror
      T = p / Rg / Ro
  return p, v, Ro

u_exact = np.zeros((grid.shape[0],3))
j=0
for i in grid:
  u_exact[j] = exact(i)
  j +=1

u_exact = torch.tensor(u_exact).reshape(-1)

u_pred = torch.tensor(net(grid)).reshape(-1)

error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

model.dict_of_learning['RMSE_pso'] = error_rmse.item()

save_data_and_graph.save_data_and_graph(model.dict_of_learning, "SOD_eq")

print('RMSE_pso= ', error_rmse.item()) 