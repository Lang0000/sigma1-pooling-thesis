import torch, warnings
from ultralytics.nn.modules.custom_pool import WindowPool, _Sigma1Fast  # the installed module, never a stale copy

torch.manual_seed(0)
g = torch.Generator().manual_seed(0)


def cases(n=2000, dtype=torch.float64):
    out = {}
    out["randn"] = torch.randn(n, 3, 3, dtype=dtype, generator=g)
    Q, _ = torch.linalg.qr(torch.randn(n, 3, 3, dtype=dtype, generator=g))
    R, _ = torch.linalg.qr(torch.randn(n, 3, 3, dtype=dtype, generator=g))
    out["s1=s2"] = Q @ torch.diag(torch.tensor([1.0, 1.0, 0.3], dtype=dtype)) @ R.transpose(-1, -2)
    out["s1=s2=s3"] = 2.0 * Q
    out["zero"] = torch.zeros(n, 3, 3, dtype=dtype)
    a = torch.randn(n, 3, 3, dtype=dtype, generator=g); a[:, 0, :] = 0; out["pad_edge"] = a
    a = torch.randn(n, 3, 3, dtype=dtype, generator=g); a[:, 0, :] = 0; a[:, :, 0] = 0; out["pad_corner"] = a
    a = torch.zeros(n, 3, 3, dtype=dtype); a[:, 1, 1] = torch.rand(n, dtype=dtype, generator=g) + 0.5; out["hot_pixel"] = a
    out["plateau"] = (torch.rand(n, 1, 1, dtype=dtype, generator=g) + 0.5) * torch.ones(n, 3, 3, dtype=dtype)
    out["plateau+1%"] = out["plateau"] * (1 + 0.01 * torch.randn(n, 3, 3, dtype=dtype, generator=g))
    out["rank1"] = torch.randn(n, 3, 1, dtype=dtype, generator=g) @ torch.randn(n, 1, 3, dtype=dtype, generator=g)
    out["near_s1=s2(1e-4)"] = Q @ torch.diag(torch.tensor([1.0, 1.0 - 1e-4, 0.3], dtype=dtype)) @ R.transpose(-1, -2)
    out["near(1e-2)"] = Q @ torch.diag(torch.tensor([1.0, 0.99, 0.3], dtype=dtype)) @ R.transpose(-1, -2)
    out["near(1e-6)"] = Q @ torch.diag(torch.tensor([1.0, 1 - 1e-6, 0.3], dtype=dtype)) @ R.transpose(-1, -2)
    out["scale1e-5"] = 1e-5 * torch.randn(n, 3, 3, dtype=dtype, generator=g)
    out["scale1e3"] = 1e3 * torch.randn(n, 3, 3, dtype=dtype, generator=g)
    return out


print(f"{'case':20s} {'dtype':8s} {'max rel err s1':>15s} {'nan':>4s} {'max|g|':>9s} {'subgrad ok':>10s} {'cos vs svd grad':>15s}")
for dtype in (torch.float32, torch.float64):
    for name, A in cases(dtype=torch.float64).items():
        ref = torch.linalg.svdvals(A)[:, 0]
        A_ = A.to(dtype).clone().requires_grad_(True)
        s = _Sigma1Fast.apply(A_)
        s.sum().backward()
        G = A_.grad.double()
        rel = ((s.double() - ref).abs() / ref.clamp_min(1e-300)).where(ref > 0, s.double().abs()).max().item()
        nan = int(torch.isnan(s).sum() + torch.isnan(G).sum())
        # valid subgradient of spectral norm: nuclear norm <= 1 and <G, A> = ||A||_2
        nuc = torch.linalg.matrix_norm(G, "nuc")
        inner = (G * A).sum((-1, -2))
        ok = ((nuc <= 1 + 1e-4) & ((inner - ref).abs() <= 1e-4 * ref.clamp_min(1e-12) + 1e-12)).double().mean().item()
        # compare with autograd through torch.linalg.svd (where gradient is unique)
        B = A.clone().requires_grad_(True)
        torch.linalg.svd(B).S[:, 0].sum().backward()
        cos = torch.nn.functional.cosine_similarity(G.flatten(1), B.grad.flatten(1), dim=1, eps=1e-30)
        print(f"{name:20s} {str(dtype)[6:]:8s} {rel:15.2e} {nan:4d} {G.flatten(1).norm(dim=1).max().item():9.6f} "
              f"{ok:10.1%} {cos.nanmean().item():15.6f}")

# gradcheck on generic matrices (float64)
A = torch.randn(64, 3, 3, dtype=torch.float64, requires_grad=True)
print("gradcheck generic:", torch.autograd.gradcheck(_Sigma1Fast.apply, (A,)))

# module level: fast vs svd on a feature map with zero padding windows
x = torch.randn(2, 8, 40, 40)
outs = {k: WindowPool(k)(x) for k in ("max", "sigma1_svd", "sigma1_fast", "maxabs", "frobenius")}
print({k: tuple(v.shape) for k, v in outs.items()})
print("fast vs svd max rel err:", ((outs["sigma1_fast"] - outs["sigma1_svd"]).abs() / outs["sigma1_svd"]).max().item())
print("bounds maxabs <= s1 <= fro <= 3 maxabs:",
      bool((outs["maxabs"] <= outs["sigma1_svd"] * (1 + 1e-5)).all() and (outs["sigma1_svd"] <= outs["frobenius"] * (1 + 1e-5)).all()
           and (outs["frobenius"] <= 3 * outs["maxabs"] * (1 + 1e-5)).all()))
# fallback behaviour: NaN input -> LinAlgError
p = WindowPool("sigma1_svd", fallback="max")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    p(torch.full((1, 2, 8, 8), float("nan")))
print("fallback counted:", p.fallbacks, "/", p.calls, "| warning:", bool(w))
try:
    WindowPool("sigma1_svd")(torch.randn(1, 2, 8, 8).half())
except RuntimeError as e:
    print("strict half:", type(e).__name__, str(e)[:70])
# gradient through module (fast vs svd) on the map
x1 = x.clone().requires_grad_(True); x2 = x.clone().requires_grad_(True)
r = torch.randn_like(outs["max"])
(WindowPool("sigma1_fast")(x1) * r).sum().backward(); (WindowPool("sigma1_svd")(x2) * r).sum().backward()
print("map grad cos fast vs svd:", torch.nn.functional.cosine_similarity(x1.grad.flatten(), x2.grad.flatten(), dim=0).item(),
      "rel err:", ((x1.grad - x2.grad).norm() / x2.grad.norm()).item())
