import gc
import time
import torch
import torch.nn as nn


def sync():
    torch.cuda.synchronize()


def make_net_and_opt(device, lr=1e-3):
    net = nn.Sequential(
        nn.Linear(3, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
    ).to(device)

    # ВАЖНО: capturable=True для CUDA Graph
    opt = torch.optim.Adam(net.parameters(), lr=lr, capturable=True)
    return net, opt


def warmup_simple(net, opt, x, iters=5):
    for _ in range(iters):
        opt.zero_grad(set_to_none=False)
        y = net(x)
        loss = (y ** 2).mean()
        loss.backward()
        opt.step()
    # гарантированно убиваем ссылки на граф
    del y, loss
    gc.collect()
    sync()


def warmup_with_autograd_grad(net, opt, x_req, iters=5):
    for _ in range(iters):
        opt.zero_grad(set_to_none=False)
        y = net(x_req)
        (gx,) = torch.autograd.grad(y.sum(), x_req, create_graph=True)
        loss = (gx ** 2).mean()
        loss.backward()
        opt.step()
    del y, gx, loss
    gc.collect()
    sync()


def capture_test_A(net, opt, x):
    print("\n[TEST A] capture: forward + loss + backward + step (NO autograd.grad)")

    # pre-allocate grads
    for p in net.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    opt.zero_grad(set_to_none=False)
    static_loss = torch.zeros((), device=x.device)

    g = torch.cuda.CUDAGraph()
    sync()
    with torch.cuda.graph(g):
        opt.zero_grad(set_to_none=False)
        y = net(x)
        loss = (y ** 2).mean()
        loss.backward()
        opt.step()
        static_loss.copy_(loss)
    sync()

    # replay
    t0 = time.perf_counter()
    for _ in range(50):
        g.replay()
    sync()
    t1 = time.perf_counter()
    print("OK: captured & replayed. time per replay step:", f"{(t1 - t0)/50*1000:.3f} ms")


def capture_test_B(net, opt, x):
    print("\n[TEST B] capture: includes torch.autograd.grad(create_graph=True)")

    # x_req должен быть leaf и постоянный
    x_req = x.detach().clone().requires_grad_(True)

    # pre-allocate grads
    for p in net.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    opt.zero_grad(set_to_none=False)
    static_loss = torch.zeros((), device=x.device)

    g = torch.cuda.CUDAGraph()
    sync()
    with torch.cuda.graph(g):
        opt.zero_grad(set_to_none=False)
        y = net(x_req)
        (gx,) = torch.autograd.grad(y.sum(), x_req, create_graph=True)
        (gxx,) = torch.autograd.grad(gx.sum(), x_req, create_graph=True)
        loss = (gx ** 2).mean()
        loss.backward()
        opt.step()
        static_loss.copy_(loss)
    sync()

    g.replay()
    sync()
    print("OK: captured & replayed with autograd.grad(create_graph=True)")
    


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    N = 4096
    x = torch.randn(N, 3, device=device)

    print("torch:", torch.__version__)
    print("device:", torch.cuda.get_device_name(0))
    print("N:", N)

    # ---- RUN TEST A in a clean net/opt ----
    net, opt = make_net_and_opt(device)
    warmup_simple(net, opt, x, iters=5)
    capture_test_A(net, opt, x)

    # ---- RUN TEST B in a fresh process ideally ----
    # Если хочешь в одном процессе — лучше пересоздать net/opt:
    net2, opt2 = make_net_and_opt(device)
    warmup_with_autograd_grad(net2, opt2, x.detach().clone().requires_grad_(True), iters=2)
    capture_test_B(net2, opt2, x)


if __name__ == "__main__":
    main()
