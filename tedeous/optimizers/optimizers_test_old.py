"""Module for custom optimizers"""
from copy import copy
from typing import Tuple
import math
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.device import device_type
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.init as init

class PSO():
    """Custom PSO optimizer.
    """
    def __init__(self,
                 pop_size: int = 30,
                 b: float = 0.9,
                 c1: float = 8e-2,
                 c2: float = 5e-1,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8):
        """The Particle Swarm Optimizer class.

        Args:
            pop_size (int, optional): Population of the PSO swarm. Defaults to 30.
            b (float, optional): Inertia of the particles. Defaults to 0.99.
            c1 (float, optional): The *p-best* coeficient. Defaults to 0.08.
            c2 (float, optional): The *g-best* coeficient. Defaults to 0.5.
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.00,
                so there will not be any gradient-based optimization.
            betas (tuple(float, float), optional): same coeff in Adam algorithm. Defaults to (0.99, 0.999).
            c_decrease (bool, optional): Flag for update_pso_params method. Defautls to False.
            variance (float, optional): Variance parameter for swarm creation
                based on model. Defaults to 1.
            epsilon (float, optional): some add to gradient descent like in Adam optimizer.
                Defaults to 1e-8.
        """
        self.pop_size = pop_size
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.c_decrease = c_decrease
        self.epsilon = epsilon
        self.beta1, self.beta2 = betas
        self.lr = lr * np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.use_grad = True if self.lr != 0 else False
        self.variance = variance
        self.name = "PSO"

        """other parameters are determined in param_init method"""
        self.model_shape = None
        self.sln_cls = None
        self.vec_shape = None
        self.swarm = None
        self.loss_swarm, self.grads_swarm = None, None
        self.p, self.f_p = None, None
        self.g_best = None
        self.v = None
        self.m1 = None
        self.m2 = None
        self.n_iter = None
        self.evolve_cond = None
        self.t = None
        self.V_max = 1.0 #Max velz
        self.C_max = 1.0
        self.C_min = 0

    def params_to_vec(self) -> torch.Tensor:
        """ Method for converting model parameters *NN and autograd*
           or model values *mat* to vector.

        Returns:
            torch.Tensor: model parameters/model values vector.
        """
        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                # реинициализация весов линейных слоев
                m.reset_parameters()
                # init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                # if m.bias is not None:
                #     fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                #     init.uniform_(m.bias, -bound, bound)

        if self.sln_cls.mode != 'mat':
            # self.sln_cls.model.apply(weights_init)
            vec = parameters_to_vector(self.sln_cls.model.parameters())
        else:
            self.model_shape = self.sln_cls.model.shape
            vec = self.sln_cls.model.reshape(-1)

        return vec

    def vec_to_params(self, vec: torch.Tensor) -> None:
        """Method for converting vector to model parameters (NN, autograd)
           or model values (mat)

        Args:
            vec (torch.Tensor): The particle of swarm. 
        """
        if self.sln_cls.mode != 'mat':
            vector_to_parameters(vec, self.sln_cls.model.parameters())
        else:
            self.sln_cls.model.data = vec.reshape(self.model_shape).data

    def param_init(self, sln_cls, tmax) -> None:
        """Method for additional class objects initializing.

        Args:
            sln_cls (Solution): Solution class object to get model and method for loss calculation.
        """
        self.sln_cls = sln_cls
        self.sln_cls.model.requires_grad_()
        vec_shape = self.params_to_vec().shape
        self.vec_shape = list(vec_shape)[0]

        self.swarm = self.build_swarm() 

        self.loss_swarm, self.grads_swarm = self.fitness_fn()

        self.p, self.f_p = copy(self.swarm).detach(), copy(self.loss_swarm).detach()    

        self.g_best = self.p[torch.argmin(self.f_p)]
        self.evolve_cond = ((self.f_p - torch.min(self.f_p)) / self.f_p).reshape(self.pop_size, 1)
        
        self.v = self.start_velocities()
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)
        self.n_iter = tmax
        self.t = 0
        self.update_pso_params()

    def build_swarm(self):
        """Creates the swarm based on solution class model.

        Returns:
            torch.Tensor: The PSO swarm population.
            Each particle represents a neural network (NN, autograd) or model values (mat).
        """

        matrix = []
        vec = self.params_to_vec()
        for _ in range(self.pop_size):
            matrix.append(vec.reshape(1,-1))
        matrix = torch.cat(matrix)
        variance = torch.FloatTensor(self.pop_size, self.vec_shape).uniform_(
                                            -self.variance, self.variance).to(device_type())
        swarm = (matrix + variance).clone().detach().requires_grad_(True)
        # swarm = (matrix).clone().detach().requires_grad_(True)
        return swarm

    def update_pso_params(self) -> None:
        """Method for updating pso parameters if c_decrease=True.
        """
        self.c1 = torch.where(self.evolve_cond <= 0.5, \
                 (self.C_max - self.C_min) * (self.n_iter - self.t) / self.n_iter + self.C_min, \
                 self.C_max - (self.C_max - self.C_min) * (self.n_iter - self.t) / self.n_iter).reshape(self.pop_size, 1)

        self.c2 = torch.where(self.evolve_cond > 0.5, \
                 (self.C_max - self.C_min) * (self.n_iter - self.t) / self.n_iter + self.C_min, \
                 self.C_max - (self.C_max - self.C_min) * (self.n_iter - self.t) / self.n_iter).reshape(self.pop_size, 1)

    def start_velocities(self) -> torch.Tensor:
        """Start the velocities of each particle in the population (swarm) as `0`.

        Returns:
            torch.Tensor: The starting velocities.
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """ Calculation of loss gradient by model parameters (NN, autograd)
            or model values (mat).

        Args:
            loss (torch.Tensor): result of loss calculation.

        Returns:
            torch.Tensor: calculated gradient vector.
        """
        if self.sln_cls.mode != 'mat':
            dl_dparam = torch.autograd.grad(loss, self.sln_cls.model.parameters())
        else:
            dl_dparam = torch.autograd.grad(loss, self.sln_cls.model)

        grads = parameters_to_vector(dl_dparam)

        return grads
    
    def loss_grads(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method for loss and gradient calculaation.
            It uses sln_cls.evaluate method for loss calc-n and
            gradient method for 'gradient' calc-n.

        Returns:
            tuple(torch.Tensor, torch.Tensor): calculated loss and gradient
        """
        loss, _ = self.sln_cls.evaluate()
        if self.use_grad:
            grads = self.gradient(loss)
            grads = torch.where(grads==float('nan'), torch.zeros_like(grads), grads)
        else:
            grads = torch.tensor([0.])
        
        return loss, grads

    def fitness_fn(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fitness function for the whole swarm.

        Returns:
            tuple(torch.Tensor, torch.Tensor): the losses and gradients for all particles.
        """
        loss_swarm = []
        grads_swarm = []
        for particle in self.swarm:
            self.vec_to_params(particle)
            loss_particle, grads = self.loss_grads()
            loss_swarm.append(loss_particle)
            grads_swarm.append(grads.reshape(1,-1))

        losses = torch.stack(loss_swarm).reshape(-1)
        gradients = torch.vstack(grads_swarm)
        return losses, gradients

    def update_evolve_cond(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the *evolve_cond* for each particles."""
        self.evolve_cond = ((self.f_p - torch.min(self.f_p)) / self.loss_swarm.detach()).reshape(self.pop_size, 1)
        
    def update_zero_v(self) -> Tuple[torch.Tensor, torch.Tensor]:

        idx = torch.where(abs(self.v) < 1e-10)
        self.idx = idx
        if len(idx[0]) > 0: 
            rand = torch.rand_like(self.v[idx])  # Генерируем случайные значения той же формы, что и self.v
            self.v[idx] = torch.where(rand > 0.5, rand * (-self.V_max), rand * self.V_max)

    def get_randoms(self) -> torch.Tensor:
        """Generate random values to update the particles' positions.

        Returns:
            torch.Tensor: random tensor
        """
        # r1 = torch.rand(self.pop_size, 1)
        # r2 = torch.rand(self.pop_size, 1)
        r1 = torch.rand_like(self.swarm.detach())
        r2 = torch.rand_like(self.swarm.detach())
        return r1, r2

    def update_p_best(self) -> None:
        """Updates the *p-best* positions."""

        idx = torch.where(self.loss_swarm.detach() < self.f_p)

        self.p[idx] = self.swarm[idx].detach()   
        self.f_p[idx] = self.loss_swarm[idx].detach()     

    def update_g_best(self) -> None:
        """Update the *g-best* position."""
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self) -> torch.Tensor:
        """ Gradiend descent based on Adam algorithm.

        Returns:
            torch.Tensor: gradient term in velocities vector.
        """
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.grads_swarm
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * torch.square(
            self.grads_swarm)
        return self.lr * self.m1 / torch.sqrt(self.m2) + self.epsilon
    
    def step(self) -> torch.Tensor:
        """ It runs ONE step on the particle swarm optimization.

        Returns:
            torch.Tensor: loss value for best particle of thw swarm.
        """
        r1, r2 = self.get_randoms()
        #sqrt = self.t**(1)
        # prev_v = self.v
        self.v = (self.evolve_cond) * self.v + \
            self.c1 * r1* (self.p - self.swarm) + \
            self.c2 * r2 * (self.g_best - self.swarm) #- (1 - self.evolve_cond)**sqrt * (self.g_best - self.p)

        # self.v =  self.c2 * r2 * (self.g_best - self.swarm) #- (1 - self.evolve_cond)**sqrt * (self.g_best - self.p)
        
        # proc_v = (prev_v*self.evolve_cond)[0][0]
        # proc_c1 = (self.c1 * r1* (self.p - self.swarm))[0][0]
        # proc_c2 = (self.c2 * r2 * (self.g_best - self.swarm))[0][0]
        # proc_dis = ((1 - self.evolve_cond)**sqrt * (self.g_best - self.p))[0][0]

        new_swarm = self.swarm + self.v 

        if self.use_grad: 
            self.swarm = torch.where((new_swarm < -1.0) | (new_swarm > 1.0), self.swarm, new_swarm) - self.gradient_descent()
        else:
            self.swarm = torch.where((new_swarm < -1.0) | (new_swarm > 1.0), self.swarm, new_swarm)
        

        # #Корректируем новое положение частиц в соответствии с условием ограничения

        
        # corrected_swarm = torch.where(new_swarm < -1, -1, new_swarm)  # Если значение меньше -1, устанавливаем -1
        # self.swarm = torch.where(corrected_swarm > 1, 1, corrected_swarm)

        self.loss_swarm, self.grads_swarm = self.fitness_fn()
        self.update_p_best()
        self.update_g_best()

        self.update_zero_v()
        self.vec_to_params(self.g_best)
        self.update_evolve_cond()
        self.update_pso_params()
        
        min_loss =  torch.min(self.f_p)
        self.t += 1

        # if self.t % 100 == 0 or self.t == 1:
        print("---------------------------------------------------")
        print(self.f_p)
        print(self.loss_swarm.detach())
        print(min_loss)
        print(self.evolve_cond[0])
        # print(self.v[0][0].item(), proc_v.item(), proc_c1.item(), proc_c2.item(), proc_dis.item())
        # # print(self.swarm[0])
        # # print(self.v[0])

        
        return min_loss, self.f_p
