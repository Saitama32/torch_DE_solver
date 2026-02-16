import torch
from tedeous.eval import Bounds

@torch.no_grad()
def _mse_cat(chunks):
    if not chunks:
        return torch.tensor(float("nan"))
    return torch.sqrt(torch.mean(torch.cat([c.reshape(-1) for c in chunks], dim=0)))

def _rmse_cat(chunks):
    if not chunks:
        return torch.tensor(float("nan"))
    return torch.sqrt(torch.mean(torch.cat([c.reshape(-1) for c in chunks], dim=0)))


def boundary_report(
    net,
    grid_for_dtype,
    bconds,
    exact_solution_fn=None,     # callable(points)->tensor (same shape as net(points))
    device=None,
    mode="autograd",
    weak_form=None,
    derivative_points=2,
):
    """
    Возвращает dict с метриками:
      - bc_residual_mse:    MSE по невязке BC (универсально для всех типов)
      - exact_u_mse:        MSE (u_pred - u_exact) по всем границам, где это имеет смысл (dirichlet + periodic sides)
      - periodic_link_mse: MSE по u(left)-u(right) или B[u](left)-B[u](right) (то есть как periodic задан)
      - periodic_exact_mse:MSE (u_pred - u_exact) отдельно по periodic-сторонам (если exact_solution_fn задан)
    """
    net.eval()

    if device is None:
        device = next(net.parameters()).device

    # --- 1) BC residual (нужно grad для operator/robin/periodic-op)
    bounds = Bounds(
        grid=grid_for_dtype,
        prepared_bconds=bconds,
        model=net,
        mode=mode,
        weak_form=weak_form,
        derivative_points=derivative_points,
    )

    bc_residual_sq = []
    periodic_link_sq = []

    with torch.enable_grad():
        for bc in bconds:
            pred = bounds.b_op_val_calc(bc).reshape(-1)   # B[u_theta] или u(left)-u(right) и т.п.
            true = bc["bval"].reshape(-1)                 # обычно g(x), для periodic это 0
            res = pred - true
            bc_residual_sq.append(res ** 2)

            if bc.get("type") == "periodic":
                # pred здесь уже "связь" (u(left)-u(right) или operator(left)-operator(right))
                periodic_link_sq.append(res ** 2)

    # --- 2) Exact on boundary (это уже "качество" решения, а не выполнение BC)
        exact_u_sq = []
    periodic_exact_sq = []

    if exact_solution_fn is not None:
        with torch.no_grad():
            for b in bconds:
                btype = b.get("type", None)

                if btype == "periodic":
                    # b["bnd"] в TEDEouS это список границ (обычно [left,right], но лучше не фиксировать 2)
                    for bnd in b["bnd"]:
                        if not isinstance(bnd, torch.Tensor):
                            continue
                        bnd = bnd.to(device)
                        u_pred = net(bnd)
                        u_ex = exact_solution_fn(bnd).to(device).reshape_as(u_pred)
                        sq = (u_pred - u_ex).reshape(-1) ** 2
                        exact_u_sq.append(sq)
                        periodic_exact_sq.append(sq)

                else:
                    if not isinstance(b.get("bnd", None), torch.Tensor):
                        continue
                    bnd = b["bnd"].to(device)
                    u_pred = net(bnd)
                    u_ex = exact_solution_fn(bnd).to(device).reshape_as(u_pred)
                    exact_u_sq.append((u_pred - u_ex).reshape(-1) ** 2)

                # operator/robin: сравнение u с exact часто НЕ отражает смысл BC,
                # но если тебе нужно — можно расширить и сравнивать оператор с exact-оператором.
                # (Оставил выключенным, потому что нужен эталонный оператор g(x) именно для производной.)

    report = {
        "bc_residual_mse": _mse_cat(bc_residual_sq).to(device),
        "bc_residual_rmse": _rmse_cat(bc_residual_sq).to(device),
        "exact_u_mse": _mse_cat(exact_u_sq).to(device),
        "exact_u_rmse": _rmse_cat(exact_u_sq).to(device),
        "periodic_link_mse": _mse_cat(periodic_link_sq).to(device),
        "periodic_link_rmse": _rmse_cat(periodic_link_sq).to(device),
        "periodic_exact_mse": _mse_cat(periodic_exact_sq).to(device),
        "periodic_exact_rmse": _rmse_cat(periodic_exact_sq).to(device),
    }
    return report
