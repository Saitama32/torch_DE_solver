import time
import torch

def make_net(neurons=100, device="cuda", dtype=torch.float32):
    net = torch.nn.Sequential(
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
        torch.nn.Linear(neurons, 1),
    ).to(device=device, dtype=dtype)
    return net

def make_points(N, device="cuda", dtype=torch.float32, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
    # равномерно/случайно — не важно для времени, главное размер
    x = torch.linspace(x_min, x_max, N, device=device, dtype=dtype)
    t = torch.linspace(t_min, t_max, N, device=device, dtype=dtype)
    # чтобы было 2D: (x_i, t_i)
    pts = torch.stack([x, t], dim=1)  # [N,2]
    return pts

def du_dt(net, points, create_graph=True):
    # points: [N,2]
    points = points.detach().requires_grad_(True)
    u = net(points)[:, 0]  # [N]
    # du/dt: axis 1 (t)
    g, = torch.autograd.grad(
        u.sum(), points,
        create_graph=create_graph,
        retain_graph=True  # retain, потому что внутри бенча много прогонов; можно и False при пересоздании points
    )
    return g[:, 1]  # [N]

def d2u_dt2(net, points, create_graph=True):
    points = points.detach().requires_grad_(True)
    u = net(points)[:, 0]
    g1, = torch.autograd.grad(u.sum(), points, create_graph=True, retain_graph=True)
    du_dt_val = g1[:, 1]
    g2, = torch.autograd.grad(du_dt_val.sum(), points, create_graph=create_graph, retain_graph=True)
    return g2[:, 1]

def bench(fn, net, pts, iters=200, warmup=50, device="cuda"):
    # прогрев
    for _ in range(warmup):
        out = fn(net, pts, create_graph=True)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(net, pts, create_graph=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (dt / iters) * 1e3  # ms/iter

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    neurons = 100
    net = make_net(neurons=neurons, device=device, dtype=dtype)

    # N_all = все boundary-точки (условно)
    N_all = 60000
    pts_all = make_points(N_all, device=device, dtype=dtype)

    # slice: например BC содержит 1/5 точек
    n_slice = 10000
    pts_slice = pts_all[:n_slice]  # тот же “кусок” точек

    # --- BENCH: du/dt ---
    ms_all = bench(du_dt, net, pts_all, iters=200, warmup=50, device=device)
    ms_sl  = bench(du_dt, net, pts_slice, iters=200, warmup=50, device=device)

    # correctness: du_dt(all)[:n] == du_dt(slice)
    a = du_dt(net, pts_all, create_graph=False)[:n_slice].detach()
    b = du_dt(net, pts_slice, create_graph=False).detach()
    max_abs = (a - b).abs().max().item()

    print(f"[du/dt] N_all={N_all}, N_slice={n_slice}")
    print(f"  all   : {ms_all:.3f} ms/iter")
    print(f"  slice : {ms_sl:.3f} ms/iter")
    print(f"  speedup (all/slice): {ms_all/ms_sl:.2f}x  (>1 => slice faster)")
    print(f"  max|all[:n]-slice| = {max_abs:.3e}")

    # --- OPTIONAL: d2u/dt2 (дороже, но ближе к wave) ---
    ms_all2 = bench(d2u_dt2, net, pts_all, iters=80, warmup=20, device=device)
    ms_sl2  = bench(d2u_dt2, net, pts_slice, iters=80, warmup=20, device=device)

    a2 = d2u_dt2(net, pts_all, create_graph=False)[:n_slice].detach()
    b2 = d2u_dt2(net, pts_slice, create_graph=False).detach()
    max_abs2 = (a2 - b2).abs().max().item()

    print(f"[d2u/dt2] N_all={N_all}, N_slice={n_slice}")
    print(f"  all   : {ms_all2:.3f} ms/iter")
    print(f"  slice : {ms_sl2:.3f} ms/iter")
    print(f"  speedup (all/slice): {ms_all2/ms_sl2:.2f}x  (>1 => slice faster)")
    print(f"  max|all[:n]-slice| = {max_abs2:.3e}")

if __name__ == "__main__":
    main()
