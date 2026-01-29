# -*- coding: utf-8 -*-
"""
Benchmark script for TEDEouS heat_2d_long_time example (NO argparse).

Measures:
- time per optimizer step (Adam.step(closure))
- forward passes per step (diagnostic)

Runs two derivative strategies:
- "autograd"  (your current default)
- "func"      (torch.func-based strategy)

IMPORTANT ABOUT OOM:
Original heat_long_time uses same grid_res for x,y,t => grid_res^3 points. :contentReference[oaicite:1]{index=1}
Here we split resolutions: grid_res_xy and grid_res_t.
Start small (e.g., 30x30x30) and scale.

How to switch strategies:
This script assumes you added a switch in TEDEouS (e.g., env var) to select derivative strategy.
If you haven't: see comment in build_model() where to inject it.
"""

import os
import sys
import time
import io
from contextlib import redirect_stdout

import numpy as np
import torch
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.device import solver_device, device_type
from tedeous.optimizers.optimizer import Optimizer
from tedeous.optimizers.closure import Closure


# =========================
# CONFIG (edit these)
# =========================
CONFIG = dict(
    # Grid: x,y,t
    grid_res_xy=30,      # start small to avoid OOM
    grid_res_t=30,       # time resolution separate from space
    t_max=30.0,         # long-time interval like in the example :contentReference[oaicite:2]{index=2}

    neurons=100,
    layers=6,            # number of hidden Linear+Tanh blocks (keep moderate)
    steps=30,           # measured steps
    warmup_steps=5,     # not measured
    repeats=5,
    lr=1e-3,

    lambda_operator=1.0,
    lambda_bound=100.0,

    mixed_precision=False,   # you can try True later (may help memory/speed)
    device="auto",           # "auto" | "cpu" | "gpu"
    seed=0,
    silent=True,
    
    use_cuda_graph=True,        # <- включай для теста на CUDA
    cuda_graph_warmup=5,         # eager шаги перед capture (инициализация Adam state и кешей)

    # If you keep your "clone() in BC operator points" fix, leave it as is in library.
)

CONFIG.update(dict(
    compile_net=False,          # <- включай для теста
    compile_evaluate=False,     # <- отдельно
    compile_mode="reduce-overhead",  # или "max-autotune"
))

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


class ForwardCounter(torch.nn.Module):
    """Wraps a module and counts forward calls via model(...) invocations."""
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


def build_heat_equation(m1=4, m2=2, k=1):
    """
    From example_heat_2d_long_time:
      du/dt - 0.001*(u_xx + u_yy) + coeff_u(x,y,t)*sin(k*u^2) = 0
    :contentReference[oaicite:3]{index=3}
    """
    equation = Equation()

    def coeff_u(grid):
        x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
        return -5 * (1 + 2 * torch.sin(torch.pi * t / 4)) * \
               torch.sin(m1 * torch.pi * x) * torch.sin(m2 * torch.pi * y)

    heat_LT = {
        'du/dt**1': {
            'coeff': 1,
            'term': [2],      # t axis
            'pow': 1,
            'var': 0
        },
        '-0.001 * d2u/dx2**1': {
            'coeff': -0.001,
            'term': [0, 0],   # x axis twice
            'pow': 1,
            'var': 0
        },
        '-0.001 * d2u/dy2**1': {
            'coeff': -0.001,
            'term': [1, 1],   # y axis twice
            'pow': 1,
            'var': 0
        },
        'coeff_u * sin(k * u ** 2)': {
            'coeff': coeff_u,
            'term': [None],
            'pow': lambda u: torch.sin(k * u ** 2),
            'var': 0
        }
    }

    equation.add(heat_LT)
    return equation


def build_model(
    grid_res_xy: int,
    grid_res_t: int,
    t_max: float,
    neurons: int,
    layers: int,
    lambda_operator: float,
    lambda_bound: float,
) -> tuple[Model, ForwardCounter]:
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    domain = Domain()
    domain.variable("x", [x_min, x_max], grid_res_xy)
    domain.variable("y", [y_min, y_max], grid_res_xy)
    domain.variable("t", [0.0, t_max], grid_res_t)

    boundaries = Conditions()

    # Initial condition u(x,y,0) = sin(4πx) sin(3πy)  :contentReference[oaicite:4]{index=4}
    boundaries.dirichlet(
        {"x": [x_min, x_max], "y": [y_min, y_max], "t": 0.0},
        value=lambda grid: torch.sin(4 * np.pi * grid[:, 0]) * torch.sin(3 * np.pi * grid[:, 1])
    )

    # Boundary conditions u=0 on edges for all t  :contentReference[oaicite:5]{index=5}
    boundaries.dirichlet({"x": x_min, "y": [y_min, y_max], "t": [0.0, t_max]}, value=0.0)
    boundaries.dirichlet({"x": [x_min, x_max], "y": y_min, "t": [0.0, t_max]}, value=0.0)
    boundaries.dirichlet({"x": x_max, "y": [y_min, y_max], "t": [0.0, t_max]}, value=0.0)
    boundaries.dirichlet({"x": [x_min, x_max], "y": y_max, "t": [0.0, t_max]}, value=0.0)

    equation = build_heat_equation()

    # Network: (3 -> ... -> 1)
    mods = [torch.nn.Linear(3, neurons), torch.nn.Tanh()]
    for _ in range(layers - 1):
        mods += [torch.nn.Linear(neurons, neurons), torch.nn.Tanh()]
    mods += [torch.nn.Linear(neurons, 1)]
    base_net = torch.nn.Sequential(*mods)

    for m in base_net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    net = ForwardCounter(base_net)

    # IMPORTANT:
    # - model.compile uses mode="autograd" for TEDEouS pipeline
    # - derivative strategy selection ("autograd" vs "func") should be done in your TEDEouS fork
    #   e.g. via env var read in Operator/Eval: os.getenv("TEDEOUS_DERIV_STRATEGY", "autograd")
    # We set it here so both strategies are benchmarked in one script:

    model = Model(net, domain, equation, boundaries)
    model.compile("autograd", lambda_operator=lambda_operator, lambda_bound=lambda_bound)

    if CONFIG["compile_net"]:
        compiled_net = torch.compile(
            model.solution_cls.model,
            mode=CONFIG["compile_mode"],
            fullgraph=False,
            dynamic=False,
        )
        # ВАЖНО: пересобираем Operator/Bounds, чтобы они ссылались на compiled_net
        model.solution_cls._model_change(compiled_net)


    return model, net


def bench_once() -> tuple[float, float, dict]:
    model, net = build_model(
        grid_res_xy=CONFIG["grid_res_xy"],
        grid_res_t=CONFIG["grid_res_t"],
        t_max=CONFIG["t_max"],
        neurons=CONFIG["neurons"],
        layers=CONFIG["layers"],
        lambda_operator=CONFIG["lambda_operator"],
        lambda_bound=CONFIG["lambda_bound"],
    )

    opt_wrap = Optimizer("AdamW", {"lr": CONFIG["lr"],
                                  "capturable": True })
    torch_opt = opt_wrap.optimizer_choice(model.mode, model.solution_cls.model)
    model.optimizer = torch_opt

    # --- loss function (evaluate) ---
    def loss_fn():
        loss, _ = model.solution_cls.evaluate()
        return loss

    # compile_evaluate у тебя для PINN с autograd.grad(create_graph=True) часто упирается в double backward,
    # так что оставляй False, но код не ломаем — просто сохраняем возможность.
    if CONFIG["compile_evaluate"]:
        loss_fn = torch.compile(
            loss_fn,
            mode=CONFIG["compile_mode"],
            fullgraph=False,
            dynamic=False,
        )

    # --- один "ручной" шаг оптимизации (лучше для CUDA Graph) ---
    def train_step():
        # ВАЖНО: set_to_none=False, чтобы не было новых аллокаций градиентов
        torch_opt.zero_grad(set_to_none=False)
        loss = loss_fn()
        loss.backward()
        torch_opt.step()
        return loss

    stdout_buf = io.StringIO()
    ctx = redirect_stdout(stdout_buf) if CONFIG["silent"] else nullcontext()

    with ctx:
        # --------------------
        # WARMUP (не измеряем)
        # --------------------
        net.reset()

        # eager warmup: прогрев CUDA + создание Adam state
        warmup_total = max(CONFIG["warmup_steps"], CONFIG.get("cuda_graph_warmup", 0))
        for _ in range(warmup_total):
            train_step()

        gc.collect()

        if device_type() == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # --------------------
        # TIMED
        # --------------------
        net.reset()

        use_graph = bool(CONFIG.get("use_cuda_graph", False)) and (device_type() == "cuda")

        if use_graph:
            # 0) прогрев CUDA/cublas до capture
            torch.cuda.synchronize()
            _ = (torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda"))
            torch.cuda.synchronize()

            # 1) заранее материализуем grad буферы (статические адреса)
            for p in model.solution_cls.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            torch_opt.zero_grad(set_to_none=False)

            static_loss = torch.zeros((), device=next(model.solution_cls.model.parameters()).device)

            g = torch.cuda.CUDAGraph()  # keep_graph=False
            pool = torch.cuda.graphs.graph_pool_handle()

            # 2) ещё один eager шаг перед capture — чтобы кеши/ветки уже “устаканились”
            train_step()
            torch.cuda.synchronize()

            # 3) CAPTURE + немедленный REPLAY тест (чтобы падало “там”, где причина)
            torch.cuda.synchronize()
            try:
                with torch.cuda.graph(g, pool=pool):
                    loss = train_step()
                    static_loss.copy_(loss)

                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()

            except BaseException as e:
                # ВАЖНО: не продолжаем в этом процессе. Иначе “липкая” ошибка выстрелит на zero_grad.
                print("CUDA Graph capture/replay failed:", repr(e))
                raise

            # 4) TIMED REPLAY
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(CONFIG["steps"]):
                g.replay()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        else:
            # обычный eager режим
            if device_type() == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(CONFIG["steps"]):
                train_step()
            if device_type() == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

    mem = {}
    if device_type() == "cuda":
        torch.cuda.synchronize()
        mem = get_cuda_mem_stats()

    sec_per_step = (t1 - t0) / CONFIG["steps"]

    # В CUDA Graph режиме forwards/step для ForwardCounter будет ~0 (replay не вызывает forward()).
    forwards_per_step = net.n_forwards / CONFIG["steps"]

    return sec_per_step, forwards_per_step, mem




def main():
    # os.environ['TORCH_USE_CUDA_DSA'] = 'True'
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    if CONFIG["device"] == "auto":
        solver_device("gpu" if torch.cuda.is_available() else "cpu")
    else:
        solver_device(CONFIG["device"])

    print(f"CUDA is available and used." if device_type() == "cuda" else "CPU is used.")
    print(f"device_type={device_type()}, torch.cuda.is_available={torch.cuda.is_available()}")
    print(
        f"grid_res_xy={CONFIG['grid_res_xy']}, grid_res_t={CONFIG['grid_res_t']}, t_max={CONFIG['t_max']}, "
        f"neurons={CONFIG['neurons']}, layers={CONFIG['layers']}, steps={CONFIG['steps']}, "
        f"warmup_steps={CONFIG['warmup_steps']}, repeats={CONFIG['repeats']}, lr={CONFIG['lr']}"
        f", compile_net={CONFIG['compile_net']}, compile_evaluate={CONFIG['compile_evaluate']}"
    )

    # Global warmup (kernels)
    sec_steps, fw_steps, mem_list = [], [], []
    for i in range(CONFIG["repeats"]):
        try:
            s, f, mem = bench_once()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Try: lower grid_res_xy/grid_res_t, lower neurons/layers, or enable mixed_precision.")
            raise

        sec_steps.append(s)
        fw_steps.append(f)
        mem_list.append(mem)
        print(f"run {i+1}/{CONFIG['repeats']}: {s*1000:.3f} ms/step, forwards/step={f:.2f} "
              f"max_alloc={mem.get('max_allocated_mb', float('nan')):.1f} MB, "
              f"max_res={mem.get('max_reserved_mb', float('nan')):.1f} MB"
        )

    sec = torch.tensor(sec_steps)
    fw = torch.tensor(fw_steps)

    max_allocs = torch.tensor([m.get("max_allocated_mb", float("nan")) for m in mem_list])
    max_res = torch.tensor([m.get("max_reserved_mb", float("nan")) for m in mem_list])

    print("Summary")
    print(f"ms/step: mean={sec.mean().item()*1000:.3f}, std={sec.std(unbiased=False).item()*1000:.3f}, "
        f"min={sec.min().item()*1000:.3f}, max={sec.max().item()*1000:.3f}")
    print(f"forwards/step: mean={fw.mean().item():.2f}, std={fw.std(unbiased=False).item():.2f}")
    print(f"max_alloc_MB: mean={max_allocs.mean().item():.1f}, max={max_allocs.max().item():.1f}")
    print(f"max_res_MB:   mean={max_res.mean().item():.1f}, max={max_res.max().item():.1f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
