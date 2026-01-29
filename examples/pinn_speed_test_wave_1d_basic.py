# -*- coding: utf-8 -*-
"""
Benchmark script for TEDEouS wave_1d_basic example (NO argparse).

It measures:
- time per optimizer step (Adam.step(closure)) for a fixed number of steps
- forward passes per step (to diagnose extra forward calls)

It intentionally excludes:
- EarlyStopping, plotting, model saving, RMSE evaluation

Edit the CONFIG section below and run:
  python bench_wave_1d_basic_noarg.py
"""

import gc
import os
import sys
import time
from contextlib import redirect_stdout
import io

import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.device import solver_device, device_type
from tedeous.optimizers.optimizer import Optimizer
from tedeous.optimizers.closure import Closure


# =========================
# CONFIG (edit these)
# =========================
CONFIG = dict(
    grid_res=100,
    neurons=100,
    steps=300,          # measured steps
    warmup_steps=50,    # not measured
    repeats=5,
    lr=1e-3,
    lambda_operator=1.0,
    lambda_bound=100.0,
    mixed_precision=False,
    device="auto",      # "auto" | "cpu" | "gpu"
    seed=0,
    silent=True,        # suppress prints inside training
    use_cudagraph=True,  # True для ускорения на CUDA
)
# =========================

def _cuda_mem_mb(x_bytes: int) -> float:
    return x_bytes / (1024 ** 2)

def get_cuda_mem_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": _cuda_mem_mb(torch.cuda.memory_allocated()),
        "reserved_mb": _cuda_mem_mb(torch.cuda.memory_reserved()),
        "max_allocated_mb": _cuda_mem_mb(torch.cuda.max_memory_allocated()),
        "max_reserved_mb": _cuda_mem_mb(torch.cuda.max_memory_reserved()),
    }

def exact_func(grid, a=4):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sin(np.pi * x) * torch.cos(2 * np.pi * t) + \
          0.5 * torch.sin(a * np.pi * x) * torch.cos(2 * a * np.pi * t)
    return sln


class ForwardCounter(torch.nn.Module):
    """Wraps a module and counts forward calls."""
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
        self.n_forwards = 0

    def forward(self, x):
        self.n_forwards += 1
        return self.net(x)

    def reset(self):
        self.n_forwards = 0


class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def build_model(grid_res: int, neurons: int, lambda_operator: float, lambda_bound: float) -> tuple[Model, ForwardCounter]:
    x_min, x_max = 0.0, 1.0
    t_max = 1.0

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0.0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0.0}, value=exact_func)

    # u_t(x, 0) = 0
    bop = {
        'du/dt': {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
    }
    boundaries.operator({'x': [x_min, x_max], 't': 0.0}, operator=bop, value=0.0)

    # Boundary conditions
    boundaries.dirichlet({'x': x_min, 't': [0.0, t_max]}, value=exact_func)
    boundaries.dirichlet({'x': x_max, 't': [0.0, t_max]}, value=exact_func)

    equation = Equation()
    wave_eq = {
        'd2u/dt2**1': {
            'coeff': 1,
            'd2u/dt2': [1, 1],
            'pow': 1
        },
        '-C*d2u/dx2**1': {
            'coeff': -4,
            'd2u/dx2': [0, 0],
            'pow': 1
        }
    }
    equation.add(wave_eq)

    base_net = torch.nn.Sequential(
        torch.nn.Linear(2, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1)
    )

    for m in base_net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    net = ForwardCounter(base_net)
    model = Model(net, domain, equation, boundaries)
    model.compile('autograd', lambda_operator=lambda_operator, lambda_bound=lambda_bound)
    # print(model.solution_cls.prepared_operator)

    return model, net


def bench_once(
    grid_res: int,
    neurons: int,
    steps: int,
    warmup_steps: int,
    lr: float,
    lambda_operator: float,
    lambda_bound: float,
    mixed_precision: bool,
    silent: bool,
) -> tuple[float, float]:
    """
    Returns:
      (seconds_per_step, forwards_per_step) measured over `steps` (excluding warmup_steps).
    """
    model, net = build_model(grid_res, neurons, lambda_operator, lambda_bound)

    # Build optimizer + closure exactly like Model.train would do
    opt_wrap = Optimizer('Adam', {'lr': lr, 'capturable': True})
    torch_opt = opt_wrap.optimizer_choice(model.mode, model.solution_cls.model)
    model.optimizer = torch_opt 
    closure = Closure(mixed_precision, model).get_closure(torch_opt)

    # Suppress internal prints if any
    stdout_buf = io.StringIO()
    ctx = redirect_stdout(stdout_buf) if silent else nullcontext()

    with ctx:
        # Warmup (not measured)
        net.reset()

        # for _ in range(warmup_steps):
        #     torch_opt.zero_grad(set_to_none=True)
        #     torch_opt.step(closure)

        if device_type() == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

                # --- Capture with CUDA Graph (optional) ---
        use_cudagraph = (CONFIG.get("use_cudagraph", False) and device_type() == "cuda")

        if use_cudagraph:
            # Важно: для CUDA graph нужен capturable optimizer + стабильные grad buffers
            # 1) не set_to_none=True
            # 2) заранее выделить p.grad
            for p in model.solution_cls.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            # статический буфер, чтобы можно было читать loss после replay (не обязательно)
            static_loss = torch.zeros(1, device=device_type())

            # прогрев allocator'а под именно этот шаг
            for _ in range(5):
                # for p in model.solution_cls.model.parameters():
                #     p.grad.zero_()
                torch_opt.step(closure)

            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()

            # Критично: во время capture никаких новых CPU-решений.
            # Мы "записываем" один полный train-step как последовательность CUDA kernels.

            # capture_stream = torch.cuda.Stream()
            # capture_stream.wait_stream(torch.cuda.current_stream())

            # with torch.cuda.stream(capture_stream):
            #     op = model.solution_cls.operator
            #     op.prepare_op_buffer()          # аллокация на capture_stream
            #     op._op_buf.zero_()              # touch на capture_stream
            #     torch.cuda.synchronize()        # или capture_stream.synchronize()

            # torch.cuda.current_stream().wait_stream(capture_stream)
            # torch.cuda.synchronize()

            with torch.cuda.graph(g, pool):
                # for p in model.solution_cls.model.parameters():
                #     p.grad.zero_()  # это станет CUDA kernels в графе
                loss = torch_opt.step(closure)  # closure + backward + adam update
                # Adam.step возвращает loss (если closure не None)
                static_loss.copy_(loss)

            torch.cuda.synchronize()


                # Timed section
        net.reset()
        # if device_type() == "cuda":
        #     # torch.cuda.synchronize()

        t0 = time.perf_counter()
        if use_cudagraph:
            for _ in range(steps):
                g.replay()
            torch.cuda.synchronize()
        else:
            for _ in range(steps):
                torch_opt.zero_grad(set_to_none=True)
                torch_opt.step(closure)
            if device_type() == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()

    sec_per_step = (t1 - t0) / steps
    forwards_per_step = net.n_forwards / steps

    # # Cleanup between runs
    # del model, net, closure, torch_opt, opt_wrap
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    mem = {}
    if device_type() == "cuda":
        torch.cuda.synchronize()
        mem = get_cuda_mem_stats()

    return sec_per_step, forwards_per_step, mem


def main():
    # Seeds
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Device
    if CONFIG["device"] == "auto":
        solver_device("gpu" if torch.cuda.is_available() else "cpu")
    else:
        solver_device(CONFIG["device"])

    print(f"device_type={device_type()}, torch.cuda.is_available={torch.cuda.is_available()}")
    print(
        f"grid_res={CONFIG['grid_res']}, neurons={CONFIG['neurons']}, steps={CONFIG['steps']}, "
        f"warmup_steps={CONFIG['warmup_steps']}, repeats={CONFIG['repeats']}, lr={CONFIG['lr']}"
    )

    # Extra global warmup run (kernels, caching)
    _ = bench_once(
        grid_res=CONFIG["grid_res"],
        neurons=CONFIG["neurons"],
        steps=max(10, CONFIG["steps"] // 10),
        warmup_steps=CONFIG["warmup_steps"],
        lr=CONFIG["lr"],
        lambda_operator=CONFIG["lambda_operator"],
        lambda_bound=CONFIG["lambda_bound"],
        mixed_precision=CONFIG["mixed_precision"],
        silent=True,
    )

    sec_steps = []
    fw_steps = []
    for i in range(CONFIG["repeats"]):
        s, f, mem = bench_once(
            grid_res=CONFIG["grid_res"],
            neurons=CONFIG["neurons"],
            steps=CONFIG["steps"],
            warmup_steps=CONFIG["warmup_steps"],
            lr=CONFIG["lr"],
            lambda_operator=CONFIG["lambda_operator"],
            lambda_bound=CONFIG["lambda_bound"],
            mixed_precision=CONFIG["mixed_precision"],
            silent=CONFIG["silent"],
        )
        sec_steps.append(s)
        fw_steps.append(f)
        print(
            f"run {i+1}/{CONFIG['repeats']}: {s*1000:.3f} ms/step, forwards/step={f:.2f}, "
            f"max_alloc={mem.get('max_allocated_mb', float('nan')):.1f} MB, "
            f"max_res={mem.get('max_reserved_mb', float('nan')):.1f} MB"
        )

    sec = torch.tensor(sec_steps)
    fw = torch.tensor(fw_steps)

    print("\nSummary")
    print(
        f"ms/step: mean={sec.mean().item()*1000:.3f}, std={sec.std(unbiased=False).item()*1000:.3f}, "
        f"min={sec.min().item()*1000:.3f}, max={sec.max().item()*1000:.3f}"
    )
    print(f"forwards/step: mean={fw.mean().item():.2f}, std={fw.std(unbiased=False).item():.2f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
