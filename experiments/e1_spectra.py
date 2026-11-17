"""E1 (spectra of real 3x3 windows) + E6 (target/background contrast of the pooled output) on the valid split.

    python experiments/e1_spectra.py --names F2_Lite-SVD-F_AMP R5_Lite-matched A2_Lite-SVD [--max-batches 0]

Every input of the shared WindowPool (6 calls per forward, FP32 eval) is taken with a forward hook and cut into the
3x3/s2/p1 windows that the pooling sees. A window is "object" if its centre lies inside a ground-truth box.
E1: singular values (torch.linalg.svdvals, float64) - histograms, share below the fp16 normal minimum 2^-14, zero and
    padding windows, sigma2/sigma1 > 0.99, coherence sigma1^2/||A||_F^2 (object vs background), Pearson r of sigma1 with
    max|a| and with the signed max, sigma1/max|a|, share of negative activations.
E6: for the same windows four operators (max, max|a|, sigma1, ||A||_F): per (call, channel) the separability
    d' = (mean_obj - mean_bg) / sqrt((var_obj + var_bg) / 2); summary over the 6 x 96 maps.
-> experiments/results/E1/e1_<name>.json, spectra_<name>.png; experiments/results/E6/e6.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.custom_pool import K, P, S, _windows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"
CHUNK = 4_000_000
FP16_MIN = 2.0 ** -14
LOG_LO, LOG_HI, NB = -8.0, 2.0, 200
OPS = ("max", "maxabs", "sigma1", "frobenius")


def window_masks(x, boxes):
    """x (B, C, H, W); boxes: list of (n, 4) normalised xywh per image -> obj (B, L), pad (L,)."""
    B, C, H, W = x.shape
    Ho, Wo = (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1
    cy = (torch.arange(Ho, device=x.device) * S + 0.5) / H
    cx = (torch.arange(Wo, device=x.device) * S + 0.5) / W
    cy, cx = torch.meshgrid(cy, cx, indexing="ij")
    cy, cx = cy.reshape(-1), cx.reshape(-1)
    obj = torch.zeros(B, Ho * Wo, dtype=torch.bool, device=x.device)
    for b, bx in enumerate(boxes):
        if len(bx):
            inside = ((cx[None] - bx[:, :1]).abs() <= bx[:, 2:3] / 2) & ((cy[None] - bx[:, 1:2]).abs() <= bx[:, 3:4] / 2)
            obj[b] = inside.any(0)
    pad = F.unfold(torch.ones(1, 1, H, W, device=x.device), K, padding=P, stride=S).amin(1)[0] < 1
    return obj, pad


class Acc:
    def __init__(self, dev):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.hist = z(3, NB)            # log10 sigma_k of non-zero values
        self.zero = z(3)
        self.below = z(3)               # 0 < sigma < 2^-14
        self.n = 0
        self.n_zero_win, self.n_pad, self.n_near, self.n_obj = 0, 0, 0, 0
        self.neg, self.entries = 0.0, 0
        self.pear = {k: z(5) for k in ("maxabs", "smax")}   # sum x, y, xx, yy, xy
        self.ratio = z(NB)              # sigma1 / max|a| in [1, 3]
        self.coh = {k: z(2) for k in ("obj", "bg")}         # sum, n
        self.s1 = {k: z(2) for k in ("obj", "bg")}
        self.e6 = {}                    # (call, op) -> (C, 2 groups, 3 sums)

    def add(self, call, x, obj, pad):
        B, C = x.shape[:2]
        self.neg += (x < 0).sum().item()
        self.entries += x.numel()
        A, _ = _windows(x.float())                                  # (B*C*L, 3, 3), order b, c, l
        L = obj.shape[1]
        objw = obj[:, None, :].expand(B, C, L).reshape(-1)
        padw = pad[None, None, :].expand(B, C, L).reshape(-1)
        if (call, "max") not in self.e6:
            for op in OPS:
                self.e6[call, op] = torch.zeros(C, 2, 3, dtype=torch.float64, device=x.device)
        for i in range(0, A.shape[0], CHUNK):
            a = A[i:i + CHUNK].double()
            o = objw[i:i + CHUNK]
            sv = torch.linalg.svdvals(a)                            # (n, 3) descending
            s1 = sv[:, 0]
            flat = a.flatten(1)
            mabs = flat.abs().amax(1)
            smax = flat.amax(1)
            fro2 = (flat * flat).sum(1)
            self.n += a.shape[0]
            self.n_obj += o.sum().item()
            self.n_pad += padw[i:i + CHUNK].sum().item()
            zw = fro2 == 0
            self.n_zero_win += zw.sum().item()
            for k in range(3):
                v = sv[:, k]
                nz = v > 0
                self.zero[k] += (~nz).sum()
                self.below[k] += (nz & (v < FP16_MIN)).sum()
                self.hist[k] += torch.histc(v[nz].log10(), NB, LOG_LO, LOG_HI)
            self.n_near += ((sv[:, 1] / s1.clamp_min(1e-300) > 0.99) & ~zw).sum().item()
            for key, y in (("maxabs", mabs), ("smax", smax)):
                self.pear[key] += torch.stack([s1.sum(), y.sum(), (s1 * s1).sum(), (y * y).sum(), (s1 * y).sum()])
            nzw = ~zw
            self.ratio += torch.histc((s1[nzw] / mabs[nzw]), NB, 1.0, 3.0)
            coh = torch.where(nzw, s1 * s1 / fro2.clamp_min(1e-300), torch.zeros_like(s1))
            for key, m in (("obj", o & nzw), ("bg", ~o & nzw)):
                self.coh[key] += torch.stack([coh[m].sum(), m.sum().double()])
                self.s1[key] += torch.stack([s1[m].sum(), m.sum().double()])
            # E6: same windows, four operators; channel index of each window
            ch = (torch.arange(i, i + a.shape[0], device=x.device) // L) % C
            vals = dict(max=smax, maxabs=mabs, sigma1=s1, frobenius=fro2.sqrt())
            for op, y in vals.items():
                acc = self.e6[call, op]
                for gi, m in enumerate((o, ~o)):
                    cm, ym = ch[m], y[m]
                    acc[:, gi, 0] += torch.bincount(cm, minlength=C).double()
                    acc[:, gi, 1] += torch.bincount(cm, ym, minlength=C)
                    acc[:, gi, 2] += torch.bincount(cm, ym * ym, minlength=C)

    def result(self):
        def pearson(s):
            n = self.n
            sx, sy, sxx, syy, sxy = (float(v) for v in s)
            return (sxy - sx * sy / n) / math.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))

        def hq(h, lo, hi, q):
            c = torch.cumsum(h, 0) / h.sum()
            i = int(torch.searchsorted(c, torch.tensor(q, dtype=c.dtype, device=c.device)))
            return lo + (hi - lo) * (i + 0.5) / NB

        centres = torch.linspace(LOG_LO, LOG_HI, NB + 1)[:-1] + (LOG_HI - LOG_LO) / NB / 2
        out = dict(windows=self.n, object_windows=self.n_obj, padding_windows=self.n_pad, zero_windows=self.n_zero_win,
                   negative_share=self.neg / self.entries,
                   near_degenerate_share=self.n_near / self.n,
                   r_sigma1_maxabs=pearson(self.pear["maxabs"]), r_sigma1_signed_max=pearson(self.pear["smax"]),
                   sigma1_over_maxabs=dict(median=hq(self.ratio, 1, 3, 0.5), p95=hq(self.ratio, 1, 3, 0.95)),
                   coherence=dict({k: float(v[0] / v[1]) for k, v in self.coh.items()}),
                   sigma1_mean=dict({k: float(v[0] / v[1]) for k, v in self.s1.items()}))
        for k in range(3):
            h = self.hist[k]
            out[f"sigma{k + 1}"] = dict(median=10 ** hq(h, LOG_LO, LOG_HI, 0.5), p05=10 ** hq(h, LOG_LO, LOG_HI, 0.05),
                                        p95=10 ** hq(h, LOG_LO, LOG_HI, 0.95), zero=int(self.zero[k]),
                                        below_fp16_min=int(self.below[k]))
        out["_hist"] = dict(log10_centres=centres.tolist(), counts=[self.hist[k].tolist() for k in range(3)])
        return out

    def e6_rows(self, name):
        rows = []
        for op in OPS:
            ds = []
            for call in sorted({c for c, _ in self.e6}):
                acc = self.e6[call, op]
                n, s, ss = acc[..., 0], acc[..., 1], acc[..., 2]
                mu = s / n.clamp_min(1)
                var = (ss / n.clamp_min(1) - mu * mu).clamp_min(0)
                d = (mu[:, 0] - mu[:, 1]) / ((var[:, 0] + var[:, 1]) / 2).sqrt().clamp_min(1e-12)
                ds.append(d[(n[:, 0] > 0) & (n[:, 1] > 0)])
            d = torch.cat(ds).abs()
            rows.append(dict(model=name, op=op, maps=d.numel(), median_abs_dprime=round(float(d.median()), 4),
                             mean_abs_dprime=round(float(d.mean()), 4), share_abs_dprime_gt_0_5=round(float((d > 0.5).float().mean()), 4),
                             share_abs_dprime_gt_1=round(float((d > 1).float().mean()), 4)))
        return rows


def plot(name, res, path):
    h = res["_hist"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k in range(3):
        ax.plot(h["log10_centres"], h["counts"][k], label=f"σ{k + 1}")
    ax.axvline(math.log10(FP16_MIN), color="k", ls=":", lw=1, label="fp16 min normal 2⁻¹⁴")
    ax.set_yscale("log")
    ax.set_xlabel("log₁₀ σ (ненульові)")
    ax.set_ylabel("кількість вікон")
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--data", default=str(DATA))
    a = p.parse_args()
    dev = torch.device("cuda")
    cfg = get_cfg(overrides=dict(imgsz=640, batch=a.batch, workers=0))
    data = check_det_dataset(a.data)
    ds = build_yolo_dataset(cfg, data[a.split], a.batch, data, mode="val", stride=32)
    dl = build_dataloader(ds, a.batch, 0, shuffle=False)
    (ROOT / "experiments" / "results" / "E1").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "results" / "E6").mkdir(parents=True, exist_ok=True)
    e6 = []
    for name in a.names:
        model = YOLO(str(ROOT / "runs" / name / "weights" / "best.pt")).model.to(dev).float().eval()
        pool = model.model[1].pool
        acc = Acc(dev)
        state = dict(call=0, boxes=None)

        def hook(m, inp, out):
            x = inp[0].detach()
            obj, pad = window_masks(x, state["boxes"])
            acc.add(state["call"], x, obj, pad)
            state["call"] += 1

        h = pool.register_forward_hook(hook)
        n_img = 0
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                if a.max_batches and bi >= a.max_batches:
                    break
                img = batch["img"].to(dev).float() / 255
                bb, idx = batch["bboxes"].to(dev), batch["batch_idx"].to(dev)
                state["boxes"] = [bb[idx == b] for b in range(img.shape[0])]
                state["call"] = 0
                model(img)
                n_img += img.shape[0]
                if bi % 10 == 0:
                    print(f"{name} batch {bi}: images {n_img}, windows {acc.n:,}", flush=True)
        h.remove()
        res = dict(name=name, pool_kind=pool.kind, images=n_img, calls_per_image=state["call"], **acc.result())
        out = ROOT / "experiments" / "results" / "E1" / f"e1_{name}.json"
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        plot(f"{name} ({pool.kind}), {n_img} зобр., {acc.n / 1e6:.0f} млн вікон", res,
             ROOT / "experiments" / "results" / "E1" / f"spectra_{name}.png")
        print(json.dumps({k: v for k, v in res.items() if k != "_hist"}, indent=1))
        e6 += acc.e6_rows(name)
        del model, acc
        torch.cuda.empty_cache()
    with open(ROOT / "experiments" / "results" / "E6" / "e6.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e6[0].keys())
        w.writeheader()
        w.writerows(e6)
    for r in e6:
        print(r)


if __name__ == "__main__":
    main()
