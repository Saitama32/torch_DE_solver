from typing import Tuple
import torch
from copy import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.device import device_type


class PSO(torch.optim.Optimizer):

    """Custom PSO optimizer.
    """

    def __init__(self,
                 params,
                 pop_size: int = 30,
                 b: float = 0.9,
                 c1: float = 8e-2,
                 c2: float = 5e-1,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8,
                 n_iter: int = 2000):
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
        defaults = {'pop_size': pop_size,
                    'b': b, 'c1': c1, 'c2': c2,
                    'lr': lr, 'betas': betas,
                    'c_decrease': c_decrease,
                    'variance': variance,
                    'epsilon': epsilon}
        super(PSO, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
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
        self.n_iter = n_iter
        self.t = 0
        self.V_max = 1.0 #Max velz
        self.C_max = 1.0
        self.C_min = 0

        vec_shape = self.params_to_vec().shape
        self.vec_shape = list(vec_shape)[0]

        self.swarm = self.build_swarm()

        self.p = copy(self.swarm).detach()

        self.v = self.start_velocities()
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)

        self.indicator = True

    def params_to_vec(self) -> torch.Tensor:
        """ Method for converting model parameters *NN and autograd*
           or model values *mat* to vector.

        Returns:
            torch.Tensor: model parameters/model values vector.
        """
        if not isinstance(self.params, torch.Tensor):
            vec = parameters_to_vector(self.params)
        else:
            self.model_shape = self.params.shape
            vec = self.params.reshape(-1)

        return vec

    def vec_to_params(self, vec: torch.Tensor) -> None:
        """Method for converting vector to model parameters (NN, autograd)
           or model values (mat)

        Args:
            vec (torch.Tensor): The particle of swarm. 
        """
        if not isinstance(self.params, torch.Tensor):
            vector_to_parameters(vec, self.params)
        else:
            self.params.data = vec.reshape(self.params).data


    def build_swarm(self):
        """Creates the swarm based on solution class model.

        Returns:
            torch.Tensor: The PSO swarm population.
            Each particle represents a neural network (NN, autograd) or model values (mat).
        """
        vector = self.params_to_vec()
        matrix = []
        for _ in range(self.pop_size):
            matrix.append(vector.reshape(1, -1))
        matrix = torch.cat(matrix)
        variance = torch.FloatTensor(self.pop_size, self.vec_shape).uniform_(
            -self.variance, self.variance).to(device_type())
        swarm = (matrix + variance).clone().detach().requires_grad_(True)
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
        dl_dparam = torch.autograd.grad(loss, self.params)

        grads = parameters_to_vector(dl_dparam)

        return grads
    
    

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

        idx = torch.where(self.loss_swarm < self.f_p)

        self.p[idx] = self.swarm[idx]
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

    def step(self, closure=None) -> torch.Tensor:
        """ It runs ONE step on the particle swarm optimization.

        Returns:
            torch.Tensor: loss value for best particle of thw swarm.
        """

        self.loss_swarm, self.grads_swarm = closure()
        if self.indicator:
            self.f_p = copy(self.loss_swarm).detach()
            self.g_best = self.p[torch.argmin(self.f_p)]
            self.evolve_cond = ((self.f_p - torch.min(self.f_p)) / self.f_p).reshape(self.pop_size, 1)
            self.indicator = False

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

        self.loss_swarm, self.grads_swarm = closure()
        self.update_p_best()
        self.update_g_best()

        #self.update_zero_v()
        self.vec_to_params(self.g_best)
        self.update_evolve_cond()
        self.update_pso_params()
        
        min_loss =  torch.min(self.f_p)
        self.t += 1

        # if self.t % 100 == 0 or self.t == 1:
        # print("---------------------------------------------------")
        # print(self.f_p)
        # print(self.loss_swarm.detach())
        # print(min_loss)
        # print(self.evolve_cond[0])
        # print(self.v[0][0].item()Ы, proc_v.item(), proc_c1.item(), proc_c2.item(), proc_dis.item())
        # # print(self.swarm[0])
        # # print(self.v[0])

        
        return min_loss
