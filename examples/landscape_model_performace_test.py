#!/usr/bin/env python3
"""
bench_ae_training.py

Benchmark speed per iteration and memory usage for the Autoencoder/VAE-like training loop
used in loss-landscape visualization.

This version uses an in-file CONFIG (no argparse).

Measures:
- step time (ms): mean, median, p95
- throughput (samples/sec)
- GPU peak memory: allocated & reserved (if CUDA)
- CPU RSS (best-effort)

Edit CONFIG and run:
  python bench_ae_training.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, Optional, OrderedDict
from landscape_visualization._aux.trajectories_data import get_trajectory_dataset  # важно: dataset+normalizer
from torch._dynamo.decorators import mark_static_address

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"C:\torchinductor_cache"
# os.environ["TRITON_CACHE_DIR"] = r"C:\triton_cache"

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# =========================
# CONFIG
# =========================
CONFIG = dict(
    # required paths / run meta
    path_to_trajectories= r"C:\Users\Рустам\Documents\GitHub\torch_DE_solver_local\test\landscape_visualization\trajectories\wave\adamw_lbfgs_state\optimizer_AdamW",
    out_dir="./bench_out",
    tag="run",

    # perf knobs
    amp="bf16",                 # "off" | "fp16" | "bf16"
    # amp="off",
    compile=False,             # torch.compile(ae)
    matmul_precision="highest",# "highest" | "high" | "medium"
    warmup_batches=50,
    max_batches=1,
    weight_decay=0.0,

    # your usual parameter blocks
    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "device": 'cuda'
    },

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 1000,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False
    },

    # kept for compatibility with your config style; not used by this benchmark script
    loss_surface_params={
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": None
    },)
# =========================


# --- optional CPU RSS ---
# def _get_cpu_rss_bytes() -> Optional[int]:
#     # Try psutil first (more portable), fallback to resource on Unix.
#     try:
#         import psutil  # type: ignore
#         return psutil.Process(os.getpid()).memory_info().rss
#     except Exception:
#         try:
#             import resource  # type: ignore
#             # ru_maxrss is KB on Linux, bytes on macOS; we assume Linux KB in most cases.
#             rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#             return int(rss_kb) * 1024
#         except Exception:
#             return None


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BenchSummary:
    device: str
    dtype: str
    amp: str
    torch_compile: bool
    matmul_precision: str
    epochs: int
    batch_size: int
    measured_steps: int
    warmup_steps: int
    total_time_ms: Optional[float]
    mean_step_ms: float
    median_step_ms: float
    p95_step_ms: float
    samples_per_sec: float
    peak_cuda_alloc_mb: Optional[float] = None
    peak_cuda_reserved_mb: Optional[float] = None
    cpu_rss_mb: Optional[float] = None

def _load_models_from_pt_files(pt_files, map_location="cpu"):
    models = []
    for fp in pt_files:
        obj = torch.load(fp, map_location=map_location, weights_only=True)

        # на всякий случай: иногда сохраняют не state_dict, а сам модуль
        if hasattr(obj, "state_dict") and not hasattr(obj, "keys"):
            obj = obj.state_dict()
        elif not hasattr(obj, "keys"):
            # fallback на случай странных типов
            obj = obj.state_dict()

        # важно: датасет проверяет именно OrderedDict, чтобы не вызывать torch.load в __getitem__
        models.append(OrderedDict(obj))
    return models


def make_gpu_batcher(X: torch.Tensor, batch_size: int):
    # X: [N, D] на GPU
    N = X.shape[0]
    perm = torch.randperm(N, device=X.device)
    ptr = 0
    while True:
        if ptr + batch_size > N:
            perm = torch.randperm(N, device=X.device)
            ptr = 0
        idx = perm[ptr:ptr + batch_size]
        ptr += batch_size
        yield X.index_select(0, idx)


def _warmup_alloc(model, optim, static_x, rec_loss_fn, rec_weight: float,
                  autocast_enabled: bool, amp_dtype):
    # прогрев: создаём стейты оптимизатора и “разкладываем” kernel-ы
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=amp_dtype):
            x_recon, z = model(static_x)
            loss = rec_loss_fn(x_recon, static_x, z).float() * rec_weight
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

def _warmup_alloc_compile(train_step, model, optim, static_x, rec_loss_fn, rec_weight: float,
                  autocast_enabled: bool, amp_dtype):
    # прогрев: создаём стейты оптимизатора и “разкладываем” kernel-ы
    for _ in range(3):
        train_step(model, optim, static_x, rec_weight)
        # 2) теперь grad точно существует -> фиксируем адреса

    torch.cuda.synchronize()
    


def make_train_cudagraph(
    model,
    optim,
    batch_shape,          # (B, D)
    rec_loss_fn,
    rec_weight: float = 1.0,
    dtype=torch.float32,  # dtype входа static_x (обычно float32)
    autocast_enabled: bool = False,
    amp_dtype=torch.bfloat16,
    compile_mode: str = "max-autotune-no-cudagraphs",
    ):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

        # 1) Компилируем МОДЕЛЬ заранее (до capture!)
    # Важно: dynamic=False, чтобы не было перекомпиляций по форме
    # model = torch.compile(model, mode=compile_mode, dynamic=False)


    B, D = batch_shape

    # статический вход: в него будем copy_() новый батч
    static_x = torch.empty((B, D), device=device, dtype=dtype)

    # статический лосс, чтобы читать после replay
    static_loss = torch.empty((), device=device, dtype=torch.float32)

    # прогрев (до захвата графа!)
    _warmup_alloc(model, optim, static_x, rec_loss_fn, rec_weight, autocast_enabled, amp_dtype)

    def train_step(ae, optim, rec_batch, rec_weight):
        optim.zero_grad(set_to_none=False)
        x_recon, z = ae(rec_batch)
        loss = rec_loss_fn(x_recon, rec_batch, z).float() * rec_weight

        loss.backward()
        optim.step()
        return loss

    # train_step = torch.compile(train_step, mode="max-autotune-no-cudagraphs")

    # _warmup_alloc_compile(train_step, model, optim, static_x, rec_loss_fn, rec_weight, autocast_enabled, amp_dtype)

    g = torch.cuda.CUDAGraph()

    # CAPTURE
    optim.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=amp_dtype):
            x_recon, z = model(static_x)
            loss = rec_loss_fn(x_recon, static_x, z).float() * rec_weight

        static_loss.copy_(loss)   # записали loss
        loss.backward()
        optim.step()
        # loss = train_step(model, optim, static_x, rec_weight)
        # static_loss.copy_(loss) 

    # STEP (replay)
    def step(batch_gpu: torch.Tensor):
        # batch_gpu: cuda tensor, shape [B, D], dtype == static_x.dtype
        static_x.copy_(batch_gpu)
        g.replay()
        return static_loss  # cuda scalar tensor

    return step



def main() -> None:
    cfg = CONFIG
    ae_cfg = cfg["AE_model_params"]
    train_cfg = cfg["AE_train_params"]

    # --- imports from your project: только нужное ---
    from landscape_visualization._aux.AEmodel import UniformAutoencoder
    from landscape_visualization._aux.losses_of_plot import rec_loss_function
    from landscape_visualization._aux.utils import get_files
    from landscape_visualization._aux.trajectories_data import get_trajectory_dataset

    # --- device ---
    device = torch.device(ae_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- load models as OrderedDict state_dicts ---
    pt_files = get_files(
        cfg["path_to_trajectories"],
        num_models=ae_cfg.get("num_models", None),
        prefix=str(ae_cfg.get("prefix", "model-")),
        from_last=bool(ae_cfg.get("from_last", False)),
        every_nth=int(ae_cfg.get("every_nth", 1)),
    )
    models = _load_models_from_pt_files(pt_files, map_location="cpu")
    print(f"Loaded {len(models)} models from {len(pt_files)} files.")

    # --- dataset -> X matrix on GPU once ---
    dataset, _normalizer = get_trajectory_dataset(models, normalize=True)
    X_cpu = torch.stack([dataset[i] for i in range(len(dataset))], dim=0).contiguous()
    X = X_cpu.to(device, non_blocking=True)

    batch_size = int(train_cfg.get("batch_size", 32))
    B, D = batch_size, X.shape[1]
    rec_iter = make_gpu_batcher(X, batch_size)

    # --- model ---
    input_dim = D
    latent_dim = int(ae_cfg.get("d_max_latent", 2))  # или просто 2
    ae = UniformAutoencoder(
        input_dim,
        int(ae_cfg.get("num_of_layers", 3)),
        latent_dim,
        h=ae_cfg["layers_AE"],
    ).to(device)

    if bool(cfg.get("compile", False)):
        ae = torch.compile(ae)

    # --- optimizer (важно: capturable + fused для cudagraph) ---
    lr = float(train_cfg.get("learning_rate", 5e-4))
    # optim = torch.optim.AdamW(
    #     ae.parameters(),
    #     lr=lr,
    #     weight_decay=float(cfg.get("weight_decay", 0.0)),
    #     capturable=(device.type == "cuda"),
    #     fused=(device.type == "cuda"),
    # )

    optim = torch.optim.RMSprop(
        ae.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        capturable=(device.type == "cuda"),
        # fused=(device.type == "cuda"),
    )
    scheduler = CosineAnnealingWarmRestarts(
        optim,
        int(train_cfg.get("first_RL_epoch_AE_params", {}).get("cosine_scheduler_patience", 1)),
    )

    # --- AMP для cudagraph: bf16 обычно лучший выбор ---
    autocast_enabled = (device.type == "cuda") and (str(cfg.get("amp", "bf16")) != "off")
    amp = str(cfg.get("amp", "bf16"))
    amp_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if amp == "fp16" else torch.float32)

    train_step_cg = make_train_cudagraph(
        model=ae,
        optim=optim,
        batch_shape=(B, D),
        rec_loss_fn=rec_loss_function,
        rec_weight=float(ae_cfg.get("rec_weight", 1.0)),
        dtype=X.dtype,
        autocast_enabled=autocast_enabled,
        amp_dtype=amp_dtype,
    )

    epochs = int(train_cfg.get("first_RL_epoch_AE_params", {}).get("epochs", 1))
    every = int(train_cfg.get("every_epoch", 1000))
    max_batches = int(cfg.get("max_batches", 200))

    step_times_ms = []

    for epoch in range(epochs):
        for b in range(max_batches):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            rec_batch = next(rec_iter)          # уже на GPU
            loss = train_step_cg(rec_batch)     # CUDA Graph replay

            scheduler.step(epoch + b / max_batches)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_ms = (t1 - t0) * 1000.0
            step_times_ms.append(step_ms)

            if (epoch % every == 0) and (b == 0):
                print(f"Epoch: {epoch}\tTotal: {float(loss):.4f}")

    # --- summary ---
    mean_ms = float(statistics.mean(step_times_ms))
    med_ms = float(statistics.median(step_times_ms))
    p95_ms = float(sorted(step_times_ms)[int(0.95 * (len(step_times_ms)-1))])
    samples_per_sec = float(batch_size / (mean_ms / 1000.0))

    peak_alloc_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_alloc_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        peak_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024**2))

    summary = BenchSummary(
        device=str(device),
        dtype=str(next(ae.parameters()).dtype),
        amp=amp,
        torch_compile=bool(cfg.get("compile", False)),
        matmul_precision=str(cfg.get("matmul_precision", "highest")),
        epochs=int(epochs),
        batch_size=int(batch_size),
        measured_steps=int(len(step_times_ms)),
        warmup_steps=int(cfg.get("warmup_batches", 20)),
        total_time_ms=sum(step_times_ms),
        mean_step_ms=mean_ms,
        median_step_ms=med_ms,
        p95_step_ms=p95_ms,
        samples_per_sec=samples_per_sec,
        peak_cuda_alloc_mb=peak_alloc_mb,
        peak_cuda_reserved_mb=peak_reserved_mb,
    )

    # (out_dir / f"bench_{tag}.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    # (out_dir / f"step_times_{tag}.txt").write_text("\n".join(f"{x:.4f}" for x in step_times_ms), encoding="utf-8")

    print("\n=== Benchmark summary ===")
    for k, v in asdict(summary).items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    main()
