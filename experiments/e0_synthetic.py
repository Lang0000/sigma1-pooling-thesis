"""E0: synthetic checks of sigma1 pooling (3x3 windows) and of the rank-1 H x W reconstruction of the article text. CPU.

    python experiments/e0_synthetic.py            -> experiments/results/E0/{e0.json, gap_sweep.png, bounds.png}

1. Gradient of sigma1(A) in 12 scenarios (zero, sigma1 = sigma2, sigma1 = sigma2 = sigma3, padding, hot pixel,
   uniform patch, rank 1, near-degenerate, fp16-quantised, tiny, huge, negative patch): torch.linalg.svdvals autograd
   (fp64, fp32) and _Sigma1Fast vs the analytic u1 v1^T; ||grad||_2, NaN/Inf.
2. Bounds max|a| <= sigma1 <= ||A||_F <= 3 max|a| on 200 000 windows of 4 distributions; hot pixel -> sigma1 = max|a|,
   uniform patch c -> sigma1 = 3c; distribution of sigma1 / max|a|.
3. Rank-1 reconstruction of an H x W map (U[:, :1] S[:1] Vh[:1]): autograd NaN when >= 2 singular values are exactly 0
   (fp32, fp64); gap sweep sigma1 - sigma2 = 1 ... 1e-6: ||grad|| of the reconstruction ~ 1/gap, ||grad sigma1|| = 1.
4. fp16 quantisation of the window: relative error of sigma1 (fp32 svd of the fp16-rounded window vs fp64).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ultralytics.nn.modules.custom_pool import _Sigma1Fast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "E0"


def scenarios(g):
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    Q1, _ = torch.linalg.qr(r(3, 3))
    Q2, _ = torch.linalg.qr(r(3, 3))
    hot = torch.zeros(3, 3, dtype=torch.float64); hot[1, 1] = 5.0
    pad = r(3, 3); pad[0, :] = 0; pad[:, 0] = 0
    near = Q1 @ torch.diag(torch.tensor([1.0, 1.0 - 1e-6, 0.3], dtype=torch.float64)) @ Q2.T
    return {
        "random": r(3, 3),
        "zero": torch.zeros(3, 3, dtype=torch.float64),
        "sigma1=sigma2": Q1 @ torch.diag(torch.tensor([2.0, 2.0, 0.5], dtype=torch.float64)) @ Q2.T,
        "sigma1=sigma2=sigma3": 1.7 * Q1,
        "padding (row+col zero)": pad,
        "hot pixel": hot,
        "uniform patch c=0.8": torch.full((3, 3), 0.8, dtype=torch.float64),
        "rank 1": torch.outer(r(3), r(3)),
        "near-degenerate gap 1e-6": near,
        "fp16-quantised": r(3, 3).half().double(),
        "tiny 1e-6": 1e-6 * r(3, 3),
        "huge 1e4": 1e4 * r(3, 3),
        "negative patch": -torch.full((3, 3), 1.3, dtype=torch.float64) + 0.05 * r(3, 3),
    }


def grad_of(A, how, dtype):
    A = A.to(dtype).clone().requires_grad_()
    s = torch.linalg.svdvals(A)[0] if how == "svdvals" else _Sigma1Fast.apply(A[None])[0]
    (g,) = torch.autograd.grad(s, A)
    return s.detach(), g


def part1(g):
    rows = []
    for name, A in scenarios(g).items():
        U, S, Vh = torch.linalg.svd(A)
        uv = torch.outer(U[:, 0], Vh[0])
        row = dict(scenario=name, sigma=[round(float(x), 8) for x in S])
        for how, dt in (("svdvals", torch.float64), ("svdvals", torch.float32), ("fast", torch.float32)):
            s, gr = grad_of(A, how, dt)
            gr = gr.double()
            finite = bool(torch.isfinite(gr).all())
            key = f"{how}_{str(dt)[6:]}"
            row[key] = dict(nan_inf=not finite,
                            grad_norm2=round(float(torch.linalg.matrix_norm(gr, 2)), 6) if finite else None,
                            # sigma1 = sigma2: u1 v1^T is not unique -> compare only the norm, not the direction
                            max_diff_u1v1T=round(float((gr - uv).abs().max()), 8) if finite and S[0] - S[1] > 1e-3 else None,
                            sigma1_rel_err=round(float(abs(s.double() - S[0]) / max(float(S[0]), 1e-300)), 10))
        rows.append(row)
    return rows


def part2(g, n=50_000):
    dists = {
        "gauss": torch.randn(n, 3, 3, generator=g, dtype=torch.float64),
        "uniform[-1,1]": 2 * torch.rand(n, 3, 3, generator=g, dtype=torch.float64) - 1,
        "sparse (70% zero)": torch.randn(n, 3, 3, generator=g, dtype=torch.float64)
        * (torch.rand(n, 3, 3, generator=g, dtype=torch.float64) > 0.7),
        "rank-1 + noise": torch.einsum("ni,nj->nij", torch.randn(n, 3, generator=g, dtype=torch.float64),
                                       torch.randn(n, 3, generator=g, dtype=torch.float64))
        + 0.1 * torch.randn(n, 3, 3, generator=g, dtype=torch.float64),
    }
    res, ratios = {}, {}
    for name, A in dists.items():
        s1 = torch.linalg.svdvals(A)[:, 0]
        mx = A.abs().amax((1, 2))
        fro = A.flatten(1).norm(dim=1)
        eps = 1e-12
        ok = (mx <= s1 + eps) & (s1 <= fro + eps) & (fro <= 3 * mx + eps)
        r = (s1 / mx.clamp_min(1e-300))[mx > 0]
        ratios[name] = r
        res[name] = dict(n=n, bounds_violations=int((~ok).sum()), ratio_sigma1_maxabs_median=round(float(r.median()), 4),
                         ratio_p95=round(float(r.quantile(0.95)), 4), ratio_max=round(float(r.max()), 4))
    hot = torch.zeros(3, 3, dtype=torch.float64); hot[2, 0] = 7.0
    res["hot pixel 7 -> sigma1"] = float(torch.linalg.svdvals(hot)[0])
    res["uniform patch 0.8 -> sigma1 (3c = 2.4)"] = float(torch.linalg.svdvals(torch.full((3, 3), 0.8, dtype=torch.float64))[0])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, r in ratios.items():
        ax.hist(r.numpy(), bins=120, range=(1, 3), histtype="step", density=True, label=name)
    ax.set_xlabel("σ₁ / max|a|  (межі: 1 … 3)")
    ax.set_ylabel("щільність")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bounds.png", dpi=150)
    plt.close(fig)
    return res


def rank1_grad(M):
    M = M.clone().requires_grad_()
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = S[0] * torch.outer(U[:, 0], Vh[0])
    (g,) = torch.autograd.grad(R.sum(), M)
    return g


def part3(g):
    res = {}
    H, W = 8, 8
    for dt in (torch.float64, torch.float32):
        x, y = torch.randn(H, generator=g, dtype=torch.float64), torch.randn(W, generator=g, dtype=torch.float64)
        e1 = torch.zeros(H, W, dtype=torch.float64); e1[0, 0] = 1.0                      # exactly 7 zero sigma
        e2 = torch.zeros(H, W, dtype=torch.float64); e2[0, 0], e2[1, 1] = 2.0, 1.0      # exactly 6 zero sigma
        z1 = torch.zeros(H, W, dtype=torch.float64); z1[:4, :4] = torch.randn(4, 4, generator=g, dtype=torch.float64)
        for k, M in {"exact rank 1: one hot pixel (7 sigma = 0 exactly)": e1,
                     "exact rank 2: two pixels (6 sigma = 0 exactly)": e2,
                     "zero-padded map: 4x4 block in 8x8 (4 sigma = 0 exactly)": z1,
                     "numerical rank 1: outer(x, y) (sigma_2.. ~ 1e-16)": torch.outer(x, y),
                     "numerical rank 2": torch.outer(x, y) + 0.5 * torch.outer(y, x),
                     "full rank": torch.randn(H, W, generator=g, dtype=torch.float64)}.items():
            gr = rank1_grad(M.to(dt))
            res[f"{k}, {str(dt)[6:]}"] = dict(nan_inf=int((~torch.isfinite(gr)).sum()),
                                              grad_norm=None if not torch.isfinite(gr).all() else round(float(gr.norm()), 4))
    # gap sweep: full-rank H x W map, sigma1 - sigma2 = gap
    gaps = torch.logspace(0, -6, 25, dtype=torch.float64)
    U, _ = torch.linalg.qr(torch.randn(H, H, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(W, W, generator=g, dtype=torch.float64))
    rec, sig = [], []
    for gap in gaps:
        s = torch.linspace(0.9, 0.1, H, dtype=torch.float64)
        s[0] = 1.0
        s[1] = 1.0 - gap
        M = U @ torch.diag(s) @ V.T
        rec.append(float(rank1_grad(M).norm()))
        A = M.clone().requires_grad_()
        (gs,) = torch.autograd.grad(torch.linalg.svdvals(A)[0], A)
        sig.append(float(torch.linalg.matrix_norm(gs, 2)))
    res["gap_sweep"] = dict(gap=[float(x) for x in gaps], grad_norm_rank1_reconstruction=rec, grad_norm2_sigma1=sig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.loglog(gaps.numpy(), rec, "o-", ms=3, label="rank-1 реконструкція H×W: ‖∇‖_F")
    ax.loglog(gaps.numpy(), sig, "s-", ms=3, label="σ₁-пулінг: ‖∇σ₁‖₂")
    ax.loglog(gaps.numpy(), 1 / gaps.numpy(), "k:", lw=1, label="1/gap")
    ax.set_xlabel("розрив σ₁ − σ₂")
    ax.set_ylabel("норма градієнта")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gap_sweep.png", dpi=150)
    plt.close(fig)
    return res


def part4(g, n=200_000):
    A = torch.randn(n, 3, 3, generator=g, dtype=torch.float64) * torch.logspace(-3, 3, n, dtype=torch.float64)[:, None, None]
    ref = torch.linalg.svdvals(A)[:, 0]
    q = torch.linalg.svdvals(A.half().float())[:, 0].double()
    rel = (q - ref).abs() / ref
    return dict(n=n, scale="1e-3 ... 1e3", max_rel=float(rel.max()), mean_rel=float(rel.mean()),
                fp16_unit_roundoff=2 ** -11)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    res = dict(gradients=part1(g), bounds=part2(g), rank1_reconstruction=part3(g), fp16_quantisation=part4(g))
    (OUT / "e0.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res["gradients"]:
        print(f"{r['scenario']:28s} " + "  ".join(
            f"{k}: nan={v['nan_inf']} |g|={v['grad_norm2']} d={v['max_diff_u1v1T']}" for k, v in r.items()
            if isinstance(v, dict)))
    print(json.dumps({k: v for k, v in res["bounds"].items()}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res["rank1_reconstruction"].items() if k != "gap_sweep"}, indent=1))
    gs = res["rank1_reconstruction"]["gap_sweep"]
    print("gap sweep:", [(f"{a:.0e}", round(b, 1), round(c, 3)) for a, b, c in
                         zip(gs["gap"][::6], gs["grad_norm_rank1_reconstruction"][::6], gs["grad_norm2_sigma1"][::6])])
    print("fp16:", res["fp16_quantisation"])


if __name__ == "__main__":
    main()
