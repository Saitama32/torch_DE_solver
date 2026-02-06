"""Module for operatoins with operator and boundaru con-ns."""

from typing import Tuple, Union, List, Callable, Optional, Dict
import torch

from tedeous.points_type import Points_type
from tedeous.derivative import Derivative
from tedeous.device import device_type, check_device
from tedeous.utils import PadTransform

from torch.utils.data import DataLoader

from collections import defaultdict

def integration(func: torch.Tensor,
                grid: torch.Tensor,
                power: int = 2) \
                -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """ Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid.

    Args:
        func (torch.Tensor): operator multiplied on test function
        grid (torch.Tensor): array of a n-D points.
        power (int, optional): power of func points. Defults to 2.

    Returns:
        'result' is integration result through one grid axis
        'grid' is initial grid without last column or zero (if grid.shape[N,1])
    """
    if grid.shape[-1] == 1:
        column = -1
    else:
        column = -2
    marker = grid[0][column]
    index = [0]
    result = []
    u = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            u += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** power + func[i - 1] ** power) / 2
        else:
            result.append(u)
            marker = grid[i][column]
            index.append(i)
            u = 0.
    if column == -1:
        return u, 0.
    else:
        result.append(u)
        grid = grid[index, :-1]
        return result, grid


def dict_to_matrix(bval: dict, true_bval: dict)\
    -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    """ Function for bounaries values matrix creation from dictionary.

    Args:
        bval (dict): dictionary with predicted boundaries values,
              where keys are boundaries types.
        true_bval (dict): dictionary with true boundaries values,
                   where keys are boundaries types.

    Returns:
        matrix_bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
        matrix_true_bval (torch.Tensor):matrix, where each column is true
                           boundary values of one boundary type.
        keys (list): boundary types list corresponding matrix_bval columns.
        len_list (list): list of length of each boundary type column.
    """

    keys = list(bval.keys())
    max_len = max([len(i) for i in bval.values()])
    pad = PadTransform(max_len, 0)
    matrix_bval = pad(bval[keys[0]]).reshape(-1,1)
    matrix_true_bval = pad(true_bval[keys[0]]).reshape(-1,1)
    len_list = [len(bval[keys[0]])]
    for key in keys[1:]:
        bval_i = pad(bval[key]).reshape(-1,1)
        true_bval_i = pad(true_bval[key]).reshape(-1,1)
        matrix_bval = torch.hstack((matrix_bval, bval_i))
        matrix_true_bval = torch.hstack((matrix_true_bval, true_bval_i))
        len_list.append(len(bval[key]))

    return matrix_bval, matrix_true_bval, keys, len_list


class Operator():
    """
    Class for differential equation calculation.
    """
    def __init__(self,
                 grid: torch.Tensor,
                 prepared_operator: Union[list,dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int,
                 batch_size: int = None):
        """
        Args:
            grid (torch.Tensor): grid (domain discretization).
            prepared_operator (Union[list,dict]): prepared (after Equation class) operator.
            model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
            mode (str): *mat or NN or autograd*
            weak_form (list[callable]): list with basis functions (if the form is *weak*).
            derivative_points (int): points number for derivative calculation.
                                     For details to Derivative_mat class.
            batch_size (int): size of batch.
        """
        self.grid = check_device(grid)
        self.prepared_operator = prepared_operator
        self.model = model.to(device_type())
        self.mode = mode
        self._vars_used = None
        if self.mode == "autograd":
            self._vars_used = self._collect_vars_used()
        self.weak_form = weak_form
        self.derivative_points = derivative_points
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode in ('autograd', 'mat'):
            self.sorted_grid = self.grid
        self.batch_size = batch_size
        if self.batch_size is not None:
            self.grid_loader =  DataLoader(self.sorted_grid, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=device_type()))
            self.n_batches = len(self.grid_loader)
            # del self.sorted_grid
            # torch.cuda.empty_cache()
            self.init_mini_batches()
            self.current_batch_i = 0
        self.create_graph = True
        # strategy = "func"  # хардкод для теста
        # strategy = "autograd"  # хардкод для теста
        self.derivative_obj = Derivative(self.model, self.derivative_points).set_strategy(self.mode)
        self.derivative = self.derivative_obj.take_derivative

    def init_mini_batches(self):
        """ Initialization of batch iterator.

        """
        self.grid_iter = iter(self.grid_loader)
        self.grid_batch = next(self.grid_iter)

    def _collect_vars_used(self) -> List[int]:
        """
        Collects output variables used by dict-terms in prepared_operator.
        Safe: ignores non-dict terms.
        """
        vars_used = set()
        for op in self.prepared_operator:
            if not isinstance(op, dict):
                continue
            for _, term in op.items():
                if isinstance(term, dict) and ("var" in term):
                    for v in term["var"]:
                        vars_used.add(int(v))
        return sorted(vars_used)
    
    # def _get_derivative(
    #     self,
    #     var: int,
    #     axis: Tuple[int, ...],
    #     points: torch.Tensor,
    #     u_cache: torch.Tensor,
    #     deriv_cache: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor],
    #     grad1_cache: Dict[int, torch.Tensor],
    # ) -> torch.Tensor:
    #     """
    #     Returns derivative d^{|axis|} u_var / d x_axis[0] ... d x_axis[k]
    #     as a flat tensor shape [N]. Uses memoization.
    #     """
    #     key = (var, axis)
    #     if key in deriv_cache:
    #         return deriv_cache[key]  # [N]

    #     # u itself
    #     if len(axis) == 0:
    #         out = u_cache[:, var]  # [N]
    #         deriv_cache[key] = out
    #         return out

    #     # first derivative: use / build grad1_cache[var] = grad(u, points) -> [N, dim]
    #     if len(axis) == 1:
    #         ax0 = axis[0]
    #         if var not in grad1_cache:
    #             g, = torch.autograd.grad(
    #                 u_cache[:, var].sum(),
    #                 points,
    #                 create_graph=self.create_graph
    #             )
    #             grad1_cache[var] = g
    #         out = grad1_cache[var][:, ax0]  # [N]
    #         deriv_cache[key] = out
    #         return out

    #     # higher order: d/d axis[-1] of previous derivative
    #     prev = self._get_derivative(var, axis[:-1], points, u_cache, deriv_cache, grad1_cache)  # [N]
    #     g, = torch.autograd.grad(
    #         prev.sum(),
    #         points,
    #         create_graph=self.create_graph
    #     )
    #     out = g[:, axis[-1]]  # [N]
    #     deriv_cache[key] = out
    #     return out
    

    # def derivative_cached(
    #     self,
    #     term: dict,
    #     points: torch.Tensor,
    #     u_cache: torch.Tensor,
    #     deriv_cache: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor],
    #     grad1_cache: Dict[int, torch.Tensor],
    # ) -> torch.Tensor:
    #     """
    #     Computes one PDE term using cached derivatives (var, axis_tuple).

    #     term format is assumed compatible with your Derivative_autograd:
    #     term['coeff'], term['var'], term['pow'], and one derivative-key like 'd2u/dt2'
    #     where term[dif_key] is list of axis-lists (e.g. [[1,1]] or [[0,0]] or [[None]])
    #     """
    #     # find derivative key safely (instead of list(term.keys())[1])
    #     dif_keys = [k for k in term.keys() if k not in ("coeff", "var", "pow")]
    #     if not dif_keys:
    #         raise ValueError(f"Bad term format (no derivative key): {term}")
    #     dif_key = dif_keys[0]

    #     coeff = term["coeff"]
    #     # keep same semantics as before: coeff may be scalar/tensor/callable
    #     if callable(coeff):
    #         coeff = coeff(points)

    #     der_term = 1.0
    #     for j, ax_list in enumerate(term[dif_key]):
    #         v = int(term["var"][j])

    #         # value or derivative
    #         if ax_list == [None]:
    #             d = self._get_derivative(v, tuple(), points, u_cache, deriv_cache, grad1_cache)  # [N]
    #         else:
    #             d = self._get_derivative(v, tuple(ax_list), points, u_cache, deriv_cache, grad1_cache)  # [N]

    #         d = d.reshape(-1, 1)

    #         p = term["pow"][j]
    #         if isinstance(p, (int, float)):
    #             der_term = der_term * (d ** p)
    #         elif callable(p):
    #             # preserve your existing callable(pow) convention
    #             der_term = p(der_term * d)
    #         else:
    #             # fallback: try tensor pow
    #             der_term = der_term * (d ** p)

    #     return coeff * der_term

    def apply_operator(self,
                       operator: list,
                       grid_points: Union[torch.Tensor, None],
                       ) -> torch.Tensor:
        """ Deciphers equation in a single grid subset to a field.

        Args:
            operator (list): prepared (after Equation class) operator. See
            input_preprocessing.operator_prepare()
            grid_points (Union[torch.Tensor, None]): Points, where numerical
            derivative is calculated. **Uses only in 'autograd' and 'mat' modes.**

        Returns:
            total (torch.Tensor): Decoded operator on a single grid subset.
        """
        # if self.mode == "autograd" and grid_points is not None and not grid_points.requires_grad:
        #     # Boundary/operator calls may pass plain tensors; autograd derivatives need grad tracking.
        #     grid_points = grid_points.detach().requires_grad_(True)

        total = None
        for term_key in operator:
            term = operator[term_key]

            dif = self.derivative(term, grid_points, create_graph=self.create_graph)

            total = dif if total is None else (total + dif)

        return total

    def _pde_compute(self) -> torch.Tensor:
        """ Computes PDE residual.

        Returns:
            torch.Tensor: P/O DE residual.
        """

        if self.batch_size is not None:
            sorted_grid = self.grid_batch
            try:
                self.grid_batch = next(self.grid_iter)
            except: # if no batches left then reinit
                self.init_mini_batches()
                self.current_batch_i = -1
        else:
            sorted_grid = self.sorted_grid

        # N = sorted_grid.shape[0]
        # num_eq = len(self.prepared_operator)
        # device = sorted_grid.device
        # dtype = sorted_grid.dtype
        # return torch.zeros((N, num_eq), device=device, dtype=dtype)

        # autograd: работаем на локальном points, чтобы не мутировать исходный тензор
        points = sorted_grid
        u_cache = None
        if self.mode == "autograd":
            points = sorted_grid.detach().requires_grad_(True)
            u_cache = self.model(points)

            # привязываем контекст и кеши к этому points/u_cache
            # теперь все последующие take_derivative будут переиспользовать u_cache и производные
            # ВАЖНО: set_context есть у Derivative_autograd (Architecture B)
            self.derivative_obj.set_context(points, u_cache=u_cache, create_graph=self.create_graph)

        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(self.prepared_operator[0], points).reshape(-1, 1)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(self.prepared_operator[i], points).reshape(-1, 1))
            op = torch.cat(op_list, 1)

        return op

    def _weak_pde_compute(self) -> torch.Tensor:
        """ Computes PDE residual in weak form.

        Returns:
            torch.Tensor: weak PDE residual.
        """

        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self._pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in self.weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for _ in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list).reshape(1,-1)

    def operator_compute(self) -> torch.Tensor:
        """ Corresponding to form (weak or strong) calculate residual of operator.

        Returns:
            torch.Tensor: operator residual.
        """
        
        if self.weak_form is None or self.weak_form == []:
            return self._pde_compute()
        else:
            return self._weak_pde_compute()


class Bounds():
    """
    Class for boundary and initial conditions calculation.
    """
    def __init__(self,
                 grid: torch.Tensor,
                 prepared_bconds: Union[list, dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int):
        """_summary_

        Args:
            grid (torch.Tensor): grid (domain discretization).
            prepared_bconds (Union[list,dict]): prepared (after Equation class) baund-y con-s.
            model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
            mode (str): *mat or NN or autograd*
            weak_form (list[callable]): list with basis functions (if the form is *weak*).
            derivative_points (int): points number for derivative calculation.
                                     For details to Derivative_mat class.
        """
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.operator = Operator(self.grid, self.prepared_bconds,
                                       self.model, self.mode, weak_form,
                                       derivative_points)
        self.operator.create_graph = True

    def _apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """ Method only for *NN* mode. Calculate boundary conditions with derivatives
            to use them in _apply_neumann method.

        Args:
            operator_set (list): list with prepared (after Equation_NN class) boundary operators.
            For details to Equation_NN.operator_prepare method.

        Returns:
            torch.Tensor: Decoded boundary operator on the whole grid.
        """

        field_part = []
        for operator in operator_set:
            field_part.append(self.operator.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def _apply_dirichlet(self, bnd: torch.Tensor, var: int, u_cache: torch.Tensor = None, sl: slice = None) -> torch.Tensor:
        """ Applies Dirichlet boundary conditions.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            For more deatails to input_preprocessing (bnd_prepare maethos).
            var (int): indicates for which dependent variable it is necessary to apply
            the boundary condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated boundary condition.
        """
        # use cache if provided (for autograd efficiency)
        if u_cache is not None and sl is not None:
            return u_cache[sl, var].reshape(-1, 1)

        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                b_op_val.append(self.model[var][position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def _apply_neumann(self, bnd: torch.Tensor, bop: list) -> torch.Tensor:
        """ Applies boundary conditions with derivative operators.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            bop (list): terms of prepared boundary derivative operator.

        Returns:
            torch.Tensor: calculated boundary condition.
        """

        if self.mode == 'NN':
            b_op_val = self._apply_bconds_set(bop)

        elif self.mode in ('autograd'):
            points = bnd.detach().requires_grad_(True)
            u_cache = self.model(points)
            self.operator.derivative_obj.set_context(points, u_cache=u_cache, create_graph=self.operator.create_graph)
            b_op_val = self.operator.apply_operator(bop, points)
        elif self.mode == 'mat':
            var = bop[list(bop.keys())[0]]['var'][0]
            b_op_val = self.operator.apply_operator(bop, self.grid)
            b_val = []
            for position in bnd:
                b_val.append(b_op_val[var][position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def _apply_periodic(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """ Applies periodic boundary conditions.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            bop (list): terms of prepared boundary derivative operator.
            var (int): indicates for which dependent variable it is necessary to apply
            the boundary condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated boundary condition
        """

        if bop is None:
            b_op_val = self._apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self._apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self._apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self._apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode in ('autograd', 'mat'):
                b_op_val = self._apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self._apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def _apply_robin(self, bnd: torch.Tensor, bop: Union[list, dict], var: int,
                     u_cache: torch.Tensor = None, sl: slice = None) -> torch.Tensor:
        """ Applies Robin boundary conditions.

        Args:
            bnd (torch.Tensor): boundary points of prepared boundary conditions.
            bop (list): prepared boundary derivative operator.
            alpha (float): coefficient for the boundary function value.
            beta (float): coefficient for the derivative term.

        Returns:
            torch.Tensor: calculated Robin boundary condition.
        """

        alpha, *betas = [bop[list(bop.keys())[i]]['coeff'] for i in range(len(bop))]

        value_term = alpha * self._apply_dirichlet(bnd, var, u_cache=u_cache, sl=sl)

        derivative_term = 0
        for beta in betas:
            if self.mode == 'NN':
                if isinstance(beta, (int, float)):
                    derivative_term += beta * self._apply_bconds_set(bop)
                elif isinstance(beta, Callable):
                    derivative_term += beta(bnd) * self._apply_bconds_set(bop)
            else:
                if isinstance(beta, (int, float)):
                    derivative_term += beta * self._apply_neumann(bnd, bop)
                elif isinstance(beta, Callable):
                    derivative_term += beta(bnd) * self._apply_neumann(bnd, bop)

        b_op_val = value_term + derivative_term
        return b_op_val

    def _apply_data(self, bnd: torch.Tensor, bop: list, var: int,
                    u_cache: torch.Tensor = None, 
                    sl: slice = None) -> torch.Tensor:
        """ Method for applying known data about solution.

        Args:
            bnd (torch.Tensor): terms (data points) of prepared boundary conditions.
            bop (list): terms of prepared data derivative operator.
            var (int): indicates for which dependent variable it is necessary to apply
            the data condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated data condition.
        """
        if bop is None:
            b_op_val = self._apply_dirichlet(bnd, var, u_cache=u_cache, sl=sl).reshape(-1, 1)
        else:
            b_op_val = self._apply_neumann(bnd, bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond: dict, u_cache: torch.Tensor = None, sl: slice = None,) -> torch.Tensor:
        """ Auxiliary function. Serves only to choose *type* of the condition and evaluate one.

        Args:
            bcond (dict): terms of prepared boundary conditions
            (see input_preprocessing module -> bnd_prepare method).

        Returns:
            torch.Tensor: calculated operator on the boundary.
        """

        b_op_val = None

        if bcond['type'] == 'dirichlet':
            b_op_val = self._apply_dirichlet(bcond["bnd"], bcond["var"], u_cache=u_cache, sl=sl)
        elif bcond['type'] == 'operator':
            b_op_val = self._apply_neumann(bcond["bnd"], bcond["bop"])
        elif bcond['type'] == 'periodic':
            b_op_val = self._apply_periodic(bcond["bnd"], bcond["bop"], bcond["var"])
        elif bcond['type'] == 'robin':
            b_op_val = self._apply_robin(bcond["bnd"], bcond["bop"], bcond["var"], u_cache=u_cache, sl=sl)
        elif bcond['type'] == 'data':
            b_op_val = self._apply_data(bcond["bnd"], bcond["bop"], bcond["var"],
                                u_cache=u_cache, sl=sl)
        return b_op_val

    def apply_bcs(self) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """ Applies boundary and data conditions for each *type* in prepared_bconds.

        Optimization:
            batch ALL dirichlet conditions into a single forward (for NN/autograd modes)
            operator/derivative conditions are still computed separately (2-forward strategy)

        Returns:
            bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
            true_bval (torch.Tensor):matrix, where each column is true
                            boundary values of one boundary type.
            keys (list): boundary types list corresponding matrix_bval columns.
            bval_length (list): list of length of each boundary type column.
        """
        bval_lists = defaultdict(list)
        true_lists = defaultdict(list)

        # ---------- value-batch (NN/autograd): dirichlet + robin(value) + data(bop None) ----------
        u_cache = None
        points_val = None
        sl_map = {}  # bc_index -> slice in u_cache

        if self.mode in ("NN", "autograd"):
            val_indices = []
            for i, bc in enumerate(self.prepared_bconds):
                btype = bc["type"]

                if btype == "periodic":
                    continue  # periodic отдельно

                if btype == "dirichlet":
                    val_indices.append(i)
                elif btype == "robin":
                    val_indices.append(i)  # alpha*u точно нужен
                elif btype == "data" and not bc.get("bop", None):
                    val_indices.append(i)

            if val_indices:
                bnds = [self.prepared_bconds[i]["bnd"] for i in val_indices]
                lens = [b.shape[0] for b in bnds]

                points_val = torch.cat(bnds, dim=0)      # без requires_grad
                u_cache = self.model(points_val)         # один forward для всех value-only

                off = 0
                for i, n in zip(val_indices, lens):
                    sl_map[i] = slice(off, off + n)
                    off += n

                # dirichlet можно как раньше сразу заполнить в bval_lists, чтобы сохранить структуру
                for i in val_indices:
                    bc = self.prepared_bconds[i]
                    if bc["type"] != "dirichlet":
                        continue
                    sl = sl_map[i]
                    var = bc["var"]

                    pred = u_cache[sl, var].reshape(-1)
                    true = bc["bval"].reshape(-1)

                    bval_lists["dirichlet"].append(pred)
                    true_lists["dirichlet"].append(true)

        # ---------- Other BCs (operator/neumann/data/etc.) ----------
        for i, bc in enumerate(self.prepared_bconds):
            btype = bc["type"]

            # dirichlet already handled above for NN/autograd
            if btype == "dirichlet" and self.mode in ("NN", "autograd"):
                continue

            if u_cache is not None and i in sl_map:
                sl = sl_map[i]
                pred = self.b_op_val_calc(bc, u_cache=u_cache, sl=sl).reshape(-1)
            else:
                pred = self.b_op_val_calc(bc).reshape(-1)

            true = bc["bval"].reshape(-1)

            bval_lists[btype].append(pred)
            true_lists[btype].append(true)

        # ---------- Pack to dict for dict_to_matrix ----------
        bval_dict = {k: torch.cat(v, dim=0) for k, v in bval_lists.items()}
        true_bval_dict = {k: torch.cat(v, dim=0) for k, v in true_lists.items()}

        bval, true_bval, keys, bval_length = dict_to_matrix(bval_dict, true_bval_dict)
        return bval, true_bval, keys, bval_length

