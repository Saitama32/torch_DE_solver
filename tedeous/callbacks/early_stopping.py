import numpy as np
from typing import Union
import torch
import datetime
from tedeous.callbacks.callback import Callback
from tedeous.utils import create_random_fn


class EarlyStopping(Callback):
    """ Class for using adaptive stop criterias at training process.
    """
    def __init__(self,
                 eps: float = 1e-5,
                 loss_window: int = 100,
                 no_improvement_patience: int = 1000,
                 patience: int = 5,
                 abs_loss: Union[float, None] = None,
                 normalized_loss: bool = False,
                 randomize_parameter: float = 1e-5,
                 info_string_every: Union[int, None] = None,
                 verbose: bool = True,
                 save_best: bool = False
                 ):
        """_summary_

        Args:
            eps (float, optional): arbitrarily small number that uses for loss comparison criterion. Defaults to 1e-5.
            loss_window (int, optional): width of losses window which is used for average loss estimation. Defaults to 100.
            no_improvement_patience (int, optional):  number of iterations during which
                    the loss may not improve.. Defaults to 1000.
            patience (int, optional): maximum number of times the stopping criterion
                                      can be satisfied.. Defaults to 5.
            abs_loss (Union[float, None], optional): absolute loss value using in _absloss_check().. Defaults to None.
            normalized_loss (bool, optional): calculate loss with all lambdas=1. Defaults to False.
            randomize_parameter (float, optional): some error for resulting
                                        model weights to to avoid local optima. Defaults to 1e-5.
            info_string_every (Union[int, None], optional): prints the loss state after every *int*
                                                    step. Defaults to None.
            verbose (bool, optional): print or not info about loss and current state of stopping criteria. Defaults to True.
            save_best (bool, optional): model with least loss is saved during the training and returned at the end as a result
        """
        super().__init__()
        self.eps = eps
        self.loss_window = loss_window
        self.no_improvement_patience = no_improvement_patience
        self.patience = patience
        self.abs_loss = abs_loss
        self.normalized_loss = normalized_loss
        self._stop_dings = 0
        self._t_imp_start = 0
        self._r = create_random_fn(randomize_parameter)
        self.info_string_every = info_string_every if info_string_every is not None else np.inf
        self.verbose = verbose
        self.save_best=save_best
        self.best_model=None



    def _line_create(self):
        # 1) окно потерь -> float, выбрасываем неконечные
        y_full = np.asarray(self.last_loss, dtype=float)
        x_full = np.arange(len(y_full), dtype=float)

        mask = np.isfinite(y_full)
        y = y_full[mask]
        x = x_full[mask]

        # 2) мало точек → безопасный фолбэк
        if y.size < 2:
            cur = self.model.cur_loss.item() if isinstance(self.model.cur_loss, torch.Tensor) else float(self.model.cur_loss)
            self._line = (0.0, cur)  # (slope, intercept)
            return

        # 3) центрируем и нормируем диапазон для устойчивости
        x_mean, y_mean = x.mean(), y.mean()
        x0, y0 = x - x_mean, y - y_mean
        xr = np.ptp(x0) or 1.0  # защита от нулевого диапазона
        yr = np.ptp(y0) or 1.0
        x1, y1 = x0 / xr, y0 / yr

        try:
            slope_n, intercept_n = np.polyfit(x1, y1, 1)  # нормализованные коэфф-ты
            # 4) денормализация
            slope = (slope_n * yr) / xr
            intercept = y_mean - slope * x_mean
            self._line = (float(slope), float(intercept))
        except np.linalg.LinAlgError:
            # 5) фолбэк: константа по среднему (наклон 0)
            self._line = (0.0, float(np.mean(y)))


    def _window_check(self):
        """ Stopping criteria. We divide angle coeff of the approximating
        line (line_create()) on current loss value and compare one with *eps*
        """
        if self.t % self.loss_window == 0 and self._check is None:
            self._line_create()
            if abs(self._line[0] / self.model.cur_loss) < self.eps and self.t > 0:
                self._stop_dings += 1
                if self.mode in ('NN', 'autograd'):
                    self.model.net.apply(self._r)
                    self.model.solution_cls._model_change(self.model.net)
                self._check = 'window_check'

    def _patience_check(self):
        """ Stopping criteria. We control the minimum loss and count steps
        when the current loss is bigger then min_loss. If these steps equal to
        no_improvement_patience parameter, the stopping criteria will be achieved.

        """
        if (self.t - self._t_imp_start) == self.no_improvement_patience and self._check is None:
            self._stop_dings += 1
            self._t_imp_start = self.t
            if self.mode in ('NN', 'autograd'):
                if self.save_best:
                    self.model.net=self.best_model
                self.model.net.apply(self._r)
                self.model.solution_cls._model_change(self.model.net)
            self._check = 'patience_check'

    def _absloss_check(self):
        """ Stopping criteria. If current loss absolute value is lower then *abs_loss* param,
        the stopping criteria will be achieved.
        """
        if self.abs_loss is not None and self.model.cur_loss < self.abs_loss and self._check is None:
            self._stop_dings += 1
            self._check = 'absloss_check'

    def verbose_print(self):
        """ print info about loss and stopping criteria.
        """

        if self._check == 'window_check':
            print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
        elif self._check == 'patience_check':
            print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(), self.no_improvement_patience))
        elif self._check == 'absloss_check':
            print('[{}] Absolute value of loss is lower than threshold'.format(
                                                        datetime.datetime.now()))

        if self._check is not None or self.t % self.info_string_every == 0:
            try:
                self._line
            except:
                self._line_create()
            loss = self.model.cur_loss.item() if isinstance(self.model.cur_loss, torch.Tensor) else self.mdoel.cur_loss
            info = '[{}] Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    datetime.datetime.now(), self.t, loss, self._line[0] / loss, self._line[1] / loss, self._stop_dings)
            print(info)

    def on_epoch_end(self, logs=None):
        self._window_check()
        self._patience_check()
        self._absloss_check()

        if self.model.cur_loss < self.model.min_loss:
            self.model.min_loss = self.model.cur_loss
            if self.save_best:
                self.best_model=self.model.net
            self._t_imp_start = self.t

        if self.verbose:
            self.verbose_print()
        if self._stop_dings >= self.patience:
            self.model.stop_training = True
            if self.save_best:
                self.model.net=self.best_model
        self._check = None

    def on_epoch_begin(self, logs=None):
        self.t = self.model.t
        self.mode = self.model.mode
        self._check = self.model._check

        # Инициализация окна, если его ещё нет
        if not hasattr(self, "last_loss"):
            base = self.model.min_loss.item() if isinstance(self.model.min_loss, torch.Tensor) else float(self.model.min_loss)
            self.last_loss = np.full(self.loss_window, float(base), dtype=float)

        # Берём текущий loss как конечный float
        cur = self.model.cur_loss
        if isinstance(cur, torch.Tensor):
            cur = cur.item()
        cur = float(cur)

        # Если не конечный — подменяем минимальным известным
        if not np.isfinite(cur):
            cur = self.model.min_loss.item() if isinstance(self.model.min_loss, torch.Tensor) else float(self.model.min_loss)
            cur = float(cur)

        # Пишем в окно
        self.last_loss[(self.t - 3) % self.loss_window] = cur

