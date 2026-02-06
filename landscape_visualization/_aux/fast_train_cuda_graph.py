# landscape_visualization/_aux/fast_train.py
import torch
from landscape_visualization._aux.losses_of_plot import rec_loss_function

def make_gpu_batcher(X, B):
    N = X.shape[0]
    while True:
        idx = torch.randint(0, N, (B,), device=X.device)
        yield X.index_select(0, idx)

def _warmup_alloc(model, optim, static_x, rec_weight, autocast_enabled, amp_dtype):
    for _ in range(20):
        optim.zero_grad()
        # with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=amp_dtype):
        x_recon, z = model(static_x)
        loss = rec_loss_function(x_recon, static_x, z).float() * float(rec_weight)
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

def make_train_cudagraph(model, optim, batch_shape, rec_weight,
                         dtype=torch.float32, autocast_enabled=False, amp_dtype=torch.bfloat16, device='cuda'):
    model.train()
    # for p in model.parameters():
    #     p.requires_grad_(True)

    B, D = batch_shape
    # static_x = torch.empty((B, D), device=device, dtype=dtype)
    # static_loss = torch.empty((), device=device, dtype=torch.float32)
    static_x = torch.empty((B, D), device=device)
    static_loss = torch.empty((), device=device)

    _warmup_alloc(model, optim, static_x, rec_weight, autocast_enabled, amp_dtype)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        optim.zero_grad()
        # with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=amp_dtype):
        x_recon, z = model(static_x)
        loss = rec_loss_function(x_recon, static_x, z).float() * float(rec_weight)
        static_loss.copy_(loss)
        loss.backward()
        optim.step()

    def step(batch_gpu: torch.Tensor):
        static_x.copy_(batch_gpu)
        g.replay()
        return static_loss

    return step
