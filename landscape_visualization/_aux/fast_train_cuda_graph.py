# landscape_visualization/_aux/fast_train.py
import torch
from landscape_visualization._aux.losses_of_plot import rec_loss_function

def make_gpu_batcher(X, B):
    N = X.shape[0]
    while True:
        idx = torch.randint(0, N, (B,), device=X.device)
        yield X.index_select(0, idx)

def _warmup_alloc(model, static_x, rec_weight):
    # ВАЖНО: static_x должен быть чем-то заполнен, иначе прогрев на мусоре
    static_x.normal_()

    for _ in range(10):
        # optim.zero_grad(set_to_none=True)
        x_recon, z = model(static_x)
        loss = rec_loss_function(x_recon, static_x, z).float() * rec_weight
        loss.backward()
        # optim.step()
    torch.cuda.synchronize()


def make_train_cudagraph(model, batch_shape, rec_weight, device="cuda", dtype=torch.float32):
    model.train()

    B, D = batch_shape
    static_x = torch.empty((B, D), device=device, dtype=dtype)
    static_loss = torch.empty((), device=device, dtype=dtype)
    # static_lr = optim.param_groups[0]["lr"]  # 1 элемент

    _warmup_alloc(model, static_x, rec_weight)

    g = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    torch.cuda.synchronize()

    with torch.cuda.graph(g, pool):
        # optim.zero_grad(set_to_none=True)
        x_recon, z = model(static_x)
        loss = rec_loss_function(x_recon, static_x, z).float() * rec_weight
        loss.backward()
        # print("Capturing:", torch.cuda.is_current_stream_capturing())
        # optim.step()
        static_loss.copy_(loss.detach())

    def step(batch_gpu: torch.Tensor):
        static_x.copy_(batch_gpu, non_blocking=True)
        # static_lr.fill_(lr)          # <-- вот тут меняем lr “для графа”
        g.replay()
        return static_loss

    return step