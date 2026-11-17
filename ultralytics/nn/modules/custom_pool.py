"""Window pooling 3x3 / stride 2 / pad 1 for the BiFPN bottom-up path.

kind:
  max            - MaxPool2d(3, 2, 1)                                  (model Lite)
  sigma1_svd     - article v1: unfold -> torch.linalg.svd -> S[:, 0]   (model Lite-SVD, as published)
  sigma1_svd_fp32- same, SVD forced to fp32 inside autocast            (R1b)
  sigma1_fast    - sigma1 via eigen-decomposition of A^T A (cyclic Jacobi, fp32), no cuSOLVER (model Lite-SVD-F)
  maxabs         - max |a| in the window                               (R9 control)
  frobenius      - ||A||_F of the window (L2 pooling)                  (R9 control)
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

K, S, P = 3, 2, 1
KINDS = ("max", "sigma1_svd", "sigma1_svd_fp32", "sigma1_fast", "maxabs", "frobenius")


def _windows(x):
    """(B, C, H, W) -> windows (B*C*L, 3, 3) and output spatial size. Zero padding, like the article code."""
    B, C, H, W = x.shape
    Ho, Wo = (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1
    p = F.unfold(x, K, padding=P, stride=S)                      # B, C*9, L
    p = p.view(B, C, K, K, Ho * Wo).permute(0, 1, 4, 2, 3)       # B, C, L, 3, 3
    return p.reshape(-1, K, K), (B, C, Ho, Wo)


JACOBI_SWEEPS = 3  # 3 and 4 sweeps give the same fp32 error (<= 2.5e-7 vs fp64 svdvals on ~12M windows); 2 are not enough


def _sym3_eig_max(M, sweeps=JACOBI_SWEEPS):
    """Largest eigenvalue and its unit eigenvector of symmetric PSD 3x3 matrices M (..., 3, 3), cyclic Jacobi.

    Element-wise ops on the 6 unique entries, in M's dtype (fp32 in training). Unlike the closed form (Smith 1961),
    Jacobi is stable when lam1 ~ lam2: in fp32 the closed-form eigenvalue error (~1e-4) exceeds such gaps and the
    cross-product eigenvector lands on v2 (that is why the closed form needed float64, slow on GeForce).
    """
    a = {(i, j): M[..., i, j] for i in range(3) for j in range(i, 3)}
    one, zero = torch.ones_like(a[0, 0]), torch.zeros_like(a[0, 0])
    V = [[one if i == j else zero for j in range(3)] for i in range(3)]          # V[row][col], columns = eigenvectors
    tiny = torch.finfo(M.dtype).tiny
    for _ in range(sweeps):
        for p, q in ((0, 1), (0, 2), (1, 2)):
            r = 3 - p - q
            rp, rq = (min(r, p), max(r, p)), (min(r, q), max(r, q))
            apq, app, aqq, arp, arq = a[p, q], a[p, p], a[q, q], a[rp], a[rq]
            d = aqq - app
            # t = tan of the rotation angle that zeroes a[p, q] (the smaller root; t = 0 if a[p, q] = 0)
            t = 2 * apq * torch.where(d >= 0, one, -one) / (d.abs() + torch.sqrt(d * d + 4 * apq * apq)).clamp_min(tiny)
            c = torch.rsqrt(t * t + 1)
            s = t * c
            a[p, p], a[q, q], a[p, q] = app - t * apq, aqq + t * apq, zero
            a[rp], a[rq] = c * arp - s * arq, s * arp + c * arq
            for i in range(3):
                V[i][p], V[i][q] = c * V[i][p] - s * V[i][q], s * V[i][p] + c * V[i][q]
    lam = torch.stack([a[0, 0], a[1, 1], a[2, 2]], -1)
    k = lam.argmax(-1)
    V = torch.stack([torch.stack(row, -1) for row in V], -2)
    v = torch.gather(V, -1, k[..., None, None].expand(*k.shape, 3, 1)).squeeze(-1)
    return lam.gather(-1, k[..., None]).squeeze(-1).clamp_min(0), v


class _Sigma1Fast(torch.autograd.Function):
    """sigma1(A) for A (N, 3, 3); backward g * u1 v1^T (the (sub)gradient of the spectral norm)."""

    @staticmethod
    def forward(ctx, A):
        # computed in A's dtype (fp32 in the model): Jacobi needs no float64 (fp64 = 1/64 of fp32 on GeForce)
        M = A.transpose(-1, -2) @ A
        lam, v = _sym3_eig_max(M)
        s = lam.sqrt()
        Av = (A @ v[..., None]).squeeze(-1)
        u = torch.where(s[..., None] > 0, Av / s.clamp_min(torch.finfo(s.dtype).tiny)[..., None], torch.zeros_like(Av))
        ctx.save_for_backward(u, v)
        return s

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        return g[..., None, None] * u[..., :, None] * v[..., None, :]


class WindowPool(nn.Module):
    """3x3 / s2 / p1 downsampling with an explicit operator and an explicit fallback policy.

    fallback: 'strict' - raise on SVD failure; 'max' - fall back to max_pool2d (article behaviour), but loudly.
    """

    def __init__(self, kind="max", fallback="strict"):
        super().__init__()
        assert kind in KINDS, kind
        self.kind, self.fallback = kind, fallback
        self.calls, self.fallbacks = 0, 0

    def forward(self, x):
        self.calls += 1
        k = self.kind
        if k == "max":
            return F.max_pool2d(x, K, S, P)
        if k == "maxabs":
            return F.max_pool2d(x.abs(), K, S, P)
        if k == "frobenius":
            return (F.avg_pool2d(x * x, K, S, P, count_include_pad=True) * K * K + 1e-12).sqrt()
        # fp32 inside, result in the input dtype: a half() model (AMP validation, no autocast) feeds it to fp16 convs
        if k == "sigma1_fast":
            with torch.autocast(x.device.type, enabled=False):
                A, shape = _windows(x.float())
                return _Sigma1Fast.apply(A).view(shape).to(x.dtype)
        # sigma1_svd / sigma1_svd_fp32
        try:
            if k == "sigma1_svd_fp32":
                with torch.autocast(x.device.type, enabled=False):
                    A, shape = _windows(x.float())
                    return torch.linalg.svd(A).S[:, 0].view(shape).to(x.dtype)
            A, shape = _windows(x)
            return torch.linalg.svd(A).S[:, 0].view(shape)          # article v1: full SVD, only S[:, 0] used
        except RuntimeError as e:                                     # NotImplementedError for fp16/bf16, LinAlgError for NaN
            if self.fallback == "strict":
                raise
            self.fallbacks += 1
            warnings.warn(f"WindowPool[{k}] fallback to max_pool2d ({type(e).__name__}: {str(e)[:80]}), "
                          f"input {tuple(x.shape)} {x.dtype}; fallbacks {self.fallbacks}/{self.calls}")
            return F.max_pool2d(x, K, S, P)
