"""Module of derivative calculations.
"""

from typing import Any, Union, List, Tuple, Callable, Optional, Dict
import numpy as np
from scipy import linalg
import torch
from torch.func import functional_call, vmap, jacrev,  grad, jvp


class DerivativeInt():
    """Interface class
    """
    def take_derivative(self, value):
        """Method that should be built in every child class"""
        raise NotImplementedError


class Derivative_NN(DerivativeInt):
    """
    Taking numerical derivative for 'NN' method.
    """

    def __init__(self, model: Any):
        """
        Args:
            model: neural network.
        """
        self.model = model

    def take_derivative(self, term: Union[list, int, torch.Tensor], *args, **kwargs) -> torch.Tensor:

        """ Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term (Union[list, int, torch.Tensor]): differential operator in conventional form.
        Returns:
            torch.Tensor: resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        if isinstance(term['coeff'], tuple):
            coeff = term['coeff'][0](term['coeff'][1]).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, scheme in enumerate(term[dif_dir][0]):
            grid_sum = 0.
            for k, grid in enumerate(scheme):
                grid_sum += self.model(grid)[:, term['var'][j]].reshape(-1, 1)\
                    * term[dif_dir][1][j][k]
            if isinstance(term['pow'][j], (int, float)):
                der_term = der_term * grid_sum ** term['pow'][j]
            elif isinstance(term['pow'][j], Callable):
                der_term = term['pow'][j](der_term * grid_sum)
        der_term = coeff * der_term

        return der_term
    
class Derivative_func(DerivativeInt):
    """
    Derivatives via torch.func (vmap + jacrev), cached per context.
    Supports derivatives up to 2nd order: u, du/dx_i, d2u/dx_i dx_j.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

        # context
        self._points = None
        self._points_ptr = None
        self._create_graph = True

        # parameter/buffer snapshot for functional_call (rebuilt per context)
        self._pb = None

        # caches
        self._u = None          # [N, out]
        self._J = None          # [N, out, dim]
        self._H = None          # [N, out, dim, dim]  (not used currently)


        self._param_versions = None

    def _get_param_versions(self):
        return tuple(p._version for p in self.model.parameters())

    def set_context(self, points: torch.Tensor, create_graph: bool = True, u_cache: Optional[torch.Tensor] = None,) -> None:
        self._points = points
        self._points_ptr = points.data_ptr()
        self._create_graph = create_graph

        # rebuild params/buffers dict each context bind (params change during training)
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        self._pb = {**params, **buffers}

        # reset caches
        self._u = None
        self._J = None
        self._H = None

        # build u and J eagerly (H lazily)
        self._build_u_J()
        # self._param_versions = self._get_param_versions()

    def _ensure_context(self, points: torch.Tensor, create_graph: bool) -> None:
        ptr = points.data_ptr()
        # cur_versions = self._get_param_versions()

        if (
            self._points is None
            or self._points_ptr != ptr
            or self._create_graph != create_graph
            # or self._param_versions != cur_versions   # ✅ веса обновились -> пересобрать
        ):
            self.set_context(points, create_graph=create_graph)

    def _f_single(self, x_single: torch.Tensor) -> torch.Tensor:
        # x_single: [dim] -> returns [out]
        y = functional_call(self.model, self._pb, (x_single.unsqueeze(0),))  # [1, out]
        return y.squeeze(0)  # [out]

    def _build_u_J(self) -> None:
        X = self._points  # [N, dim]
        # u: [N, out]
        self._u = vmap(self._f_single)(X)
        # J: [N, out, dim]
        self._J = vmap(jacrev(self._f_single))(X)

    def _build_H(self) -> None:
        if self._H is not None:
            return
        X = self._points
        # H: [N, out, dim, dim]
        self._H = vmap(jacrev(jacrev(self._f_single)))(X)

    def _get_derivative(self, var: int, axis: Tuple[int, ...]) -> torch.Tensor:
        # returns [N]
        if not self._create_graph and len(axis) > 0:
            return torch.zeros((self._points.shape[0],), device=self._points.device, dtype=self._points.dtype)

        if len(axis) == 0:
            return self._u[:, var]
        if len(axis) == 1:
            return self._J[:, var, axis[0]]
        if len(axis) == 2:
            self._build_H()
            return self._H[:, var, axis[0], axis[1]]
        raise NotImplementedError("Derivative_func supports derivatives up to 2nd order only.")

    def take_derivative(
        self,
        term: dict,
        grid_points: torch.Tensor,
        create_graph: bool = True,
        **kwargs
    ) -> torch.Tensor:
        # bind context to current grid_points
        self._ensure_context(grid_points, create_graph=create_graph)

        dif_dir = list(term.keys())[1]

        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.0
        for j, derivative in enumerate(term[dif_dir]):
            v = int(term['var'][j])

            if derivative == [None]:
                d = self._get_derivative(v, ())
            else:
                d = self._get_derivative(v, tuple(derivative))

            d = d.reshape(-1, 1)

            p = term['pow'][j]
            if isinstance(p, (int, float)):
                der_term = der_term * (d ** p)
            elif isinstance(p, Callable):
                der_term = p(der_term * d)
            else:
                der_term = der_term * (d ** p)

        return coeff * der_term

class Derivative_autograd(DerivativeInt):
    """
    Taking numerical derivative for 'autograd' method.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model (torch.nn.Module): model of *autograd* mode.
        """
        self.model = model
        # context (set per points-batch)
        self._points = None
        self._points_ptr = None
        self._u_cache = None
        self._create_graph = True

        # caches
        self._d_cache = {}      # (var, axis_tuple) -> tensor [N]
        self._grad_cache = {}   # var -> tensor [N, dim]  (stores full grad for first derivatives)

    def set_context(
        self,
        points: torch.Tensor,
        u_cache: Optional[torch.Tensor] = None,
        create_graph: bool = True,
    ) -> None:
        """
        Bind caches to a specific points tensor (and optional precomputed u_cache).
        Must be called when points/u_cache changes (Operator will call for PDE).
        """
        self._points = points
        self._points_ptr = points.data_ptr()
        self._u_cache = u_cache
        self._create_graph = create_graph

        self._d_cache.clear()
        self._grad_cache.clear()

    def _ensure_context(
        self,
        points: torch.Tensor,
        u_cache: Optional[torch.Tensor],
        create_graph: bool,
    ) -> None:
        """
        If context differs, rebind it. If u_cache is not provided, will be computed lazily.
        """
        # ptr = points.data_ptr()
        if (
            self._points is None
            or self._points is not points
            or (u_cache is not None and self._u_cache is not u_cache)
            or self._create_graph != create_graph
        ):
            # bind new context; u_cache may be None (lazy)
            self.set_context(points, u_cache=u_cache, create_graph=create_graph)
        else:
            # same points; if caller provided u_cache and we don't have it yet, keep it
            if u_cache is not None and self._u_cache is None:
                self._u_cache = u_cache

    def _u(self) -> torch.Tensor:
        """
        Returns u(points) and caches it in context.
        """
        if self._u_cache is None:
            self._u_cache = self.model(self._points)
        return self._u_cache


    def _nn_autograd(self,
                     var: int,
                     axis: Tuple[int] ) -> torch.Tensor:
        """ Computes derivative on the grid using autograd method.

        Args:
            model (torch.nn.Module): torch neural network.
            points (torch.Tensor): points, where numerical derivative is calculated.
            var (int): number of dependent variables (for single equation is *0*)
            axis (list, optional): term of differentiation, example [0,0]->d2/dx2
                                   if grid_points(x,y). Defaults to [0].

        Returns:
            gradient_full (torch.Tensor): the result of desired function differentiation
                in corresponding axis.
        """
        axis = tuple(axis)
        key = (var, axis)
        if key in self._d_cache:
            return self._d_cache[key]

        # if graph is not needed (e.g. PSO) keep old behavior: derivatives -> zeros
        if not self._create_graph and len(axis) > 0:
            out = torch.zeros((self._points.shape[0],), device=self._points.device, dtype=self._points.dtype)
            self._d_cache[key] = out
            return out
        # points.requires_grad_(True)

                # u
        if len(axis) == 0:
            gradient_full = self._u()[:, var]
            self._d_cache[key] = gradient_full
            return gradient_full

                # first derivative: cache full grad(u_var) once
        if len(axis) == 1:
            ax0 = axis[0]
            if var not in self._grad_cache:
                g, = torch.autograd.grad(
                    self._u()[:, var].sum(),
                    self._points,
                    create_graph=self._create_graph
                )
                self._grad_cache[var] = g  # [N, dim]
            gradient_full = self._grad_cache[var][:, ax0]
            self._d_cache[key] = gradient_full
            return gradient_full

        # higher order: d/d axis[-1] of previous derivative
        prev = self._nn_autograd(var, axis[:-1])  # [N]
        g, = torch.autograd.grad(
            prev.sum(),
            self._points,
            create_graph=self._create_graph
        )
        gradient_full = g[:, axis[-1]]
        self._d_cache[key] = gradient_full
        return gradient_full

    def take_derivative(self, term: dict, grid_points:  torch.Tensor, create_graph: bool = True, u_cache: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """ Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term (dict): differential operator in conventional form.
            grid_points (torch.Tensor): points, where numerical derivative is calculated.

        Returns:
            der_term (torch.Tensor): resulting field, computed on a grid.
        """
        # bind context (lazy u if u_cache is None)
        self._ensure_context(grid_points, u_cache=u_cache, create_graph=create_graph)

        dif_dir = list(term.keys())[1]
        # it is may be int, function of grid or torch.Tensor
        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, derivative in enumerate(term[dif_dir]):
            v = int(term['var'][j])
            if derivative == [None]:
                d = self._nn_autograd(v, ())
            else:
                d = self._nn_autograd(v, tuple(derivative))

            d = d.reshape(-1, 1)

            p = term['pow'][j]
            if isinstance(p, (int, float)):
                der_term = der_term * (d ** p)
            elif isinstance(p, Callable):
                der_term = p(der_term * d)
            else:
                der_term = der_term * (d ** p)

        return coeff * der_term


class Derivative_mat(DerivativeInt):
    """
    Taking numerical derivative for 'mat' method.
    """
    def __init__(self, model: torch.Tensor, derivative_points: int):
        """
        Args:
            model (torch.Tensor): model of *mat* mode.
            derivative_points (int): points number for derivative calculation.
        """
        self.model = model
        self.backward, self.farward = Derivative_mat._labels(derivative_points)

        self.alpha_backward = Derivative_mat._linear_system(self.backward)
        self.alpha_farward = Derivative_mat._linear_system(self.farward)

        num_points = int(len(self.backward) - 1)

        self.back = [int(0 - i) for i in range(1, num_points + 1)]

        self.farw = [int(i) for i in range(num_points)]

    @staticmethod
    def _labels(derivative_points: int) -> Tuple[List, List]:
        """ Determine which points are used in derivative calc-n.
            If derivative_points = 2, it return ([-1, 0], [0, 1])

        Args:
            derivative_points (int): points number for derivative calculation.

        Returns:
            labels_backward (list): points labels for backward scheme.
            labels_forward (list): points labels for forward scheme.
        """
        labels_backward = list(i for i in range(-derivative_points + 1, 1))
        labels_farward = list(i for i in range(derivative_points))
        return labels_backward, labels_farward

    @staticmethod
    def _linear_system(labels: list) -> np.ndarray:
        """ To caclulate coeeficints in numerical scheme,
            we have to solve the linear system of algebraic equations.
            A*alpha=b

        Args:
            labels (list): points labels for backward/foraward scheme.

        Returns:
            alpha (np.ndarray): coefficints for numerical scheme.
        """
        points_num = len(labels) # num_points=number of equations
        labels = np.array(labels)
        A = []
        for i in range(points_num):
            A.append(labels**i)
        A = np.array(A)

        b = np.zeros_like(labels)
        b[1] = 1

        alpha = linalg.solve(A, b)

        return alpha

    def _derivative_1d(self, u_tensor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Computes derivative in one dimension for matrix method.

        Args:
            u_tensor (torch.Tensor): dependenet varible of equation,
                                     some part of model.
            h (torch.Tensor): increment of numerical scheme.

        Returns:
            du (torch.Tensor): computed derivative along one dimension.
        """

        shape = u_tensor.shape
        u_tensor = u_tensor.reshape(-1)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)
        du[self.back] = du_back[self.back] / h
        du[self.farw] = du_farw[self.farw] / h

        du = du.reshape(shape)

        return du

    def _step_h(self, h_tensor: torch.Tensor) -> list[torch.Tensor]:
        """ Calculate increment along each axis of the grid.

        Args:
            h_tensor (torch.Tensor): grid of *mat* mode.

        Returns:
            h (list[torch.Tensor]): lsit with increment
                                    along each axis of the grid.
        """
        h = []

        nn_grid = torch.vstack([h_tensor[i].reshape(-1) for i in \
                                range(h_tensor.shape[0])]).T.float()

        for i in range(nn_grid.shape[-1]):
            axis_points = torch.unique(nn_grid[:,i])
            h.append(abs(axis_points[1]-axis_points[0]))
        return h

    def _derivative(self,
                    u_tensor: torch.Tensor,
                    h: torch.Tensor,
                    axis: int) -> torch.Tensor:
        """ Computing derivative for 'mat' method.

        Args:
            u_tensor (torch.Tensor): dependenet varible of equation,
                                     some part of model.
            h (torch.Tensor): increment of numerical scheme.
            axis (int): axis along which the derivative is calculated.

        Returns:
            du (torch.Tensor): computed derivative.
        """

        if len(u_tensor.shape)==1 or u_tensor.shape[0]==1:
            du = self._derivative_1d(u_tensor, h)
            return du

        pos = len(u_tensor.shape) - 1

        u_tensor = torch.transpose(u_tensor, pos, axis)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)

        if pos == 1:
            du[:,self.back] = du_back[:,self.back] / h
            du[:, self.farw] = du_farw[:, self.farw] / h
        elif pos == 2:
            du[:,:, self.back] = du_back[:,:, self.back] / h
            du[:,:, self.farw] = du_farw[:,:, self.farw] / h

        du = torch.transpose(du, pos, axis)

        return du

    def take_derivative(self, term: torch.Tensor, grid_points: torch.Tensor, **kwargs) -> torch.Tensor:

        """ Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term (torch.Tensor): differential operator in conventional form.
            grid_points (torch.Tensor): grid points.

        Returns:
            der_term (torch.Tensor): resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(term[dif_dir]):
            prod=self.model[term['var'][j]]
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = self._step_h(grid_points)[axis]
                    prod = self._derivative(prod, h, axis)
            if isinstance(term['pow'][j], (int, float)):
                der_term = der_term * prod ** term['pow'][j]
            elif isinstance(term['pow'][j], Callable):
                der_term = term['pow'][j](der_term * prod)
        if callable(term['coeff']) is True:
            der_term = term['coeff'](grid_points) * der_term
        else:
            der_term = term['coeff'] * der_term
        return der_term


class Derivative():
    """
   Interface for taking numerical derivative due to chosen calculation mode.

   """
    def __init__(self,
                 model: Union[torch.nn.Module, torch.Tensor],
                 derivative_points: int):
        """_summary_

        Args:
            model (Union[torch.nn.Module, torch.Tensor]): neural network or
                                        matrix depending on the selected mode.
            derivative_points (int): points number for derivative calculation.
            If derivative_points=2, numerical scheme will be ([-1,0],[0,1]),
            parameter determine number of poins in each forward and backward scheme.
        """

        self.model = model
        self.derivative_points = derivative_points

    def set_strategy(self,
                     strategy: str) -> Union[Derivative_NN, Derivative_autograd, Derivative_mat]:
        """
        Setting the calculation method.
        Args:
            strategy: Calculation method. (i.e., "NN", "autograd", "mat").
        Returns:
            equation in input form for a given calculation method.
        """
        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return Derivative_autograd(self.model)
        
        elif strategy == 'func':
            return Derivative_func(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model, self.derivative_points)











