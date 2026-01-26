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


import torch

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

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # tag = str(cfg.get("tag", "run"))

    # --- imports from your project ---
    from landscape_visualization._aux.AEmodel import UniformAutoencoder
    from landscape_visualization._aux.losses_of_plot import rec_loss_function, loss_anchor, loss_grid_to_trajectory
    from landscape_visualization._aux.trajectories_data import (
        get_trajectory_dataloader,
        get_anchor_dataloader,
        get_predefined_values,
    )
    from landscape_visualization._aux.utils import (
        get_files,
        loss_well_spaced_trajectory,
        get_gridpoint_and_trajectory_datasets,
    )

    # --- device / precision ---
    if ae_cfg.get("device", None) is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(ae_cfg["device"]) if isinstance(ae_cfg["device"], str) else ae_cfg["device"]

    # torch.set_float32_matmul_precision(str(cfg.get("matmul_precision", "highest")))

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # --- data ---
    pt_files = get_files(
        cfg["path_to_trajectories"],
        num_models=ae_cfg.get("num_models", None),
        prefix=str(ae_cfg.get("prefix", "model-")),
        from_last=bool(ae_cfg.get("from_last", False)),
        every_nth=int(ae_cfg.get("every_nth", 1)),
    )

    models = _load_models_from_pt_files(pt_files)


    # ВАЖНО: normalize=True -> будет использоваться тот же нормализатор, что и раньше
    dataset, normalizer = get_trajectory_dataset(models, normalize=True)

    # Собираем матрицу [N, D] на CPU
    X_cpu = torch.stack([dataset[i] for i in range(len(dataset))], dim=0).contiguous()

    # Переносим на GPU один раз
    device = ae_cfg["device"]
    X = X_cpu.to(device, non_blocking=True)

    print(f"Built X on {device}: shape={X.shape}, dtype={X.dtype}")

    batch_size = int(train_cfg.get("batch_size", 32))
    rec_iter = make_gpu_batcher(X, batch_size)

    B = batch_size
    D = X.shape[1]
    
    rec_loader, transform = get_trajectory_dataloader(batch_size, models=models, device=device)
    dataset = rec_loader.dataset
    input_dim = dataset[0].shape[0]

    # --- model ---
    latent_dim = 2
    ae = UniformAutoencoder(
        input_dim,
        int(ae_cfg.get("num_of_layers", 3)),
        latent_dim,
        h=ae_cfg["layers_AE"],
    ).to(device)

    if bool(cfg.get("compile", False)):
        ae = torch.compile(ae)

    # --- losses enabled? ---
    def enabled(w: float) -> bool:
        return float(w) > 0.0

    # --- iterators: cycle like in VisualizationModel.train ---
    def cycle(dataloader) -> Iterator:
        while True:
            for batch in dataloader:
                yield batch

    def train_step(ae, optim, rec_batch):
        optim.zero_grad(set_to_none=True)
        x_recon, z = ae(rec_batch)
        loss = rec_loss_function(x_recon, rec_batch, z).float() * ae_cfg.get("rec_weight", 1.0)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        return loss

    train_step = torch.compile(train_step, mode="max-autotune")

    iters: Dict[str, Iterator] = {"rec": cycle(rec_loader)}
    max_batches = min(len(rec_loader), int(cfg.get("max_batches", 200)))

    lr = float(train_cfg.get("learning_rate", 5e-4))
    optim = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)), capturable=True, fused=True)

    # --- AMP setup ---
    amp = str(cfg.get("amp", "off"))
    if amp == "off":
        autocast_enabled = False
        amp_dtype = torch.float32
        scaler = None
    else:
        autocast_enabled = True
        amp_dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=(use_cuda and amp == "fp16"))
        # scaler = None

    # --- bench ---
    step_times_ms = []
    measured_steps = 0
    warmup_steps = int(cfg.get("warmup_batches", 20))

    epochs = int(train_cfg.get("first_RL_epoch_AE_params", {}).get("epochs", 1))

    every = int(train_cfg.get("every_epoch", 100))

    # AMP режим: для cudagraph лучше bf16 (если есть) или off
    autocast_enabled = True
    amp_dtype = torch.bfloat16   # если GPU реально поддерживает bf16
    # для T4: лучше autocast_enabled=False (fp32) или fp16 без scaler, если стабильно

    train_step_cg = make_train_cudagraph(
        model=ae,
        optim=optim,
        batch_shape=(B, D),
        rec_loss_fn=rec_loss_function,
        rec_weight=float(ae_cfg.get("rec_weight", 1.0)),
        dtype=torch.float32,              # X у тебя float32 — оставь так
        autocast_enabled=autocast_enabled,
        amp_dtype=amp_dtype,
    )

    for epoch in range(epochs):
        for b in range(max_batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rec_batch = next(rec_iter)          # уже на GPU, shape [B, D]
            loss = train_step_cg(rec_batch)     # CUDA Graph replay
            torch.cuda.synchronize()    
            t1 = time.perf_counter()
            step_ms = (t1 - t0) * 1000.0
            step_times_ms.append(step_ms)
            if (epoch % every == 0) and (b == 0):
                printed_string = f"Epoch: {epoch}\t"
                printed_string += f"Total: {loss:.4f}"
                print(printed_string)

            # loss.item() лучше не делать каждый шаг (дорого синхронизирует)

    # ae.train()
    # for epoch in range(epochs):
    #     for b in range(max_batches):
    #         torch.cuda.synchronize()
    #         t0 = time.perf_counter()
    #         optim.zero_grad(set_to_none=True)

    #         loss_total = torch.zeros((), device=device, dtype=torch.float32)

    #         rec_batch = next(rec_iter)        # на GPU сразуы
    #         if rec_batch.dim() == 1:
    #             rec_batch = rec_batch.view(1, -1)

    #         # _maybe_sync(device)
    #         # t0 = time.perf_counter()
    #         # loss_total = torch.zeros((), device=device, dtype=torch.float32)

    #         # t0 = time.perf_counter()
    #         # optim.zero_grad(set_to_none=True)
    #         # t1 = time.perf_counter()

    #         # # rec_batch = next(iters["rec"])          # CPU
    #         # rec_batch = next(rec_iter)        # на GPU сразуы
    #         # t2 = time.perf_counter()

    #         # # rec_batch = rec_batch.to(device, non_blocking=True)  # H2D (асинхронно)
    #         # t3 = time.perf_counter()

    #         # torch.cuda.synchronize()
    #         # t4 = time.perf_counter()

    #         with torch.autocast(device_type=device, enabled=autocast_enabled, dtype=amp_dtype):

    #             loss_total = train_step(ae, optim, rec_batch)
    #             # x_recon, z = ae(rec_batch)

    #             # if enabled(ae_cfg.get("rec_weight", 1.0)):

                    
    #             #     loss_total = loss_total + rec_loss_function(x_recon, rec_batch, z).float() * float(ae_cfg.get("rec_weight", 1.0))

    #             # if enabled(ae_cfg.get("anchor_weight", 0.0)):
    #             #     anchor_batch = next(iters["anchor"]).to(device).float()
    #             #     _, zz = ae(anchor_batch)
    #             #     loss_total = loss_total + loss_anchor(zz, predefined_values).float() * float(ae_cfg.get("anchor_weight", 0.0))

    #             # if enabled(ae_cfg.get("lastzero_weight", 0.0)):
    #             #     last_coordinate = z[-1, :]
    #             #     loss_zero = torch.nn.functional.mse_loss(10 * last_coordinate, torch.zeros_like(last_coordinate))
    #             #     loss_total = loss_total + loss_zero.float() * float(ae_cfg.get("lastzero_weight", 0.0))

    #             # if enabled(ae_cfg.get("polars_weight", 0.0)):
    #             #     last_coordinate = z[-1, :]
    #             #     first_coordinate = z[0, :]
    #             #     loss1 = torch.nn.functional.mse_loss(10 * last_coordinate, 10 * 0.8 * torch.ones_like(last_coordinate))
    #             #     loss2 = torch.nn.functional.mse_loss(10 * first_coordinate, 10 * -0.8 * torch.ones_like(first_coordinate))
    #             #     loss_total = loss_total + (loss1 + loss2).float() * float(ae_cfg.get("polars_weight", 0.0))

    #             # if enabled(ae_cfg.get("wellspacedtrajectory_weight", 0.0)):
    #             #     well_batch = next(iters["wellspacedtrajectory"]).to(device).float()
    #             #     _, z2 = ae(well_batch)
    #             #     loss_total = loss_total + loss_well_spaced_trajectory(z2).float() * float(ae_cfg.get("wellspacedtrajectory_weight", 0.0))

    #             # if enabled(ae_cfg.get("gridscaling_weight", 0.0)):
    #             #     data_grid_latent, data_trajectory = next(iters["gridscaling"])
    #             #     data_grid_latent = data_grid_latent[0].to(device)  # TensorDataset returns tuple
    #             #     data_trajectory = data_trajectory.to(device)
    #             #     loss_total = loss_total + loss_grid_to_trajectory(
    #             #         ae, data_grid_latent, data_trajectory, l_max_inputspace,
    #             #         d_max_latent=float(ae_cfg.get("d_max_latent", 2.0)), epoch=epoch
    #             #     ).float() * float(ae_cfg.get("gridscaling_weight", 0.0))

    #         if (epoch % every == 0) and (b == 0):
    #             printed_string = f"Epoch: {epoch}\t"
    #             printed_string += f"Total: {loss_total:.4f}"

    #             print(printed_string)

    #         # if scaler is not None:
    #         #     scaler.scale(loss_total).backward()
    #         #     scaler.step(optim)
    #         #     scaler.update()
    #         # else:
    #         #     loss_total.backward()
    #         #     optim.step()

    #         torch.cuda.synchronize()
    #         t1 = time.perf_counter()
    #         step_ms = (t1 - t0) * 1000.0

    #         if measured_steps >= warmup_steps:
    #             step_times_ms.append(step_ms)
    #         measured_steps += 1

    #         # torch.cuda.synchronize()
    #         # t5 = time.perf_counter()

    #         # print("zero_grad", (t1-t0)*1e3,
    #         #     "next", (t2-t1)*1e3,
    #         #     "to()", (t3-t2)*1e3,
    #         #     "wait_h2d", (t4-t3)*1e3,
    #         #     "train", (t5-t4)*1e3) 

    if len(step_times_ms) == 0:
        raise RuntimeError("No measured steps collected. Reduce warmup_batches or increase max_batches/epochs.")

    mean_ms = float(statistics.mean(step_times_ms))
    med_ms = float(statistics.median(step_times_ms))
    p95_ms = float(sorted(step_times_ms)[int(0.95 * (len(step_times_ms)-1))])

    samples_per_sec = float(batch_size / (mean_ms / 1000.0))

    peak_alloc_mb = None
    peak_reserved_mb = None
    if use_cuda:
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
        warmup_steps=int(warmup_steps),
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
    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # или "highest"
    main()
