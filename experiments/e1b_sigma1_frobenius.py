"""E1b: distribution of sigma1 / ||A||_F of real 3x3 windows (object vs background), complement to E1.

E1 kept only the mean coherence sigma1^2/||A||_F^2; this script adds the histogram, so the position of sigma1 between
max|a| and ||A||_F (L2 pooling) can be given as quantiles. Same windows as E1: every input of the shared WindowPool
(6 calls per forward, FP32 eval), 3x3/s2/p1, "object" if the window centre lies inside a ground-truth box.
Runs on CPU (or CUDA if present), does not touch the E1 / E6 results.

    python experiments/e1b_sigma1_frobenius.py --names F2_Lite-SVD-F_AMP R5_Lite-matched --max-batches 7
-> experiments/results/E1/e1b_<name>.json, e1b_sigma1_frobenius.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.custom_pool import _windows

from e1_spectra import DATA, ROOT, window_masks

CHUNK = 2_000_000
LO, NB = 1 / math.sqrt(3), 200          # sigma1/||A||_F lies in [1/sqrt(3); 1] for 3x3 windows


def quantile(h, q):
    c = torch.cumsum(h, 0) / h.sum()
    i = int(torch.searchsorted(c, torch.tensor(q, dtype=c.dtype)))
    return LO + (1 - LO) * (i + 0.5) / NB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg(overrides=dict(imgsz=640, batch=a.batch, workers=0))
    data = check_det_dataset(str(DATA))
    ds = build_yolo_dataset(cfg, data[a.split], a.batch, data, mode="val", stride=32)
    dl = build_dataloader(ds, a.batch, 0, shuffle=False)
    out_dir = ROOT / "experiments" / "results" / "E1"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in a.names:
        w = ROOT / "runs" / name / "weights" / "best.pt"
        if not w.exists():
            w = ROOT / "experiments" / "weights" / name / "best.pt"
        model = YOLO(str(w)).model.to(dev).float().eval()
        pool = model.model[1].pool
        hist = {k: torch.zeros(NB, dtype=torch.float64) for k in ("obj", "bg")}
        kappa = {k: torch.zeros(2, dtype=torch.float64) for k in ("obj", "bg")}   # sum kappa, n
        state = dict(boxes=None)

        def hook(m, inp, out):
            x = inp[0].detach()
            obj, _ = window_masks(x, state["boxes"])
            A, _ = _windows(x.float())
            B, C, L = x.shape[0], x.shape[1], obj.shape[1]
            objw = obj[:, None, :].expand(B, C, L).reshape(-1)
            for i in range(0, A.shape[0], CHUNK):
                aa = A[i:i + CHUNK].double().cpu()
                o = objw[i:i + CHUNK].cpu()
                fro2 = aa.flatten(1).pow(2).sum(1)
                nz = fro2 > 0
                s1 = torch.linalg.svdvals(aa[nz])[:, 0]
                r = s1 / fro2[nz].sqrt()
                for key, m in (("obj", o[nz]), ("bg", ~o[nz])):
                    hist[key] += torch.histc(r[m], NB, LO, 1.0)
                    kappa[key] += torch.stack([(r[m] ** 2).sum(), m.sum().double()])

        h = pool.register_forward_hook(hook)
        n_img = 0
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                if a.max_batches and bi >= a.max_batches:
                    break
                img = batch["img"].to(dev).float() / 255
                bb, idx = batch["bboxes"].to(dev), batch["batch_idx"].to(dev)
                state["boxes"] = [bb[idx == b] for b in range(img.shape[0])]
                model(img)
                n_img += img.shape[0]
                print(f"{name} batch {bi}: images {n_img}, windows {int(sum(v.sum() for v in hist.values())):,}",
                      flush=True)
        h.remove()
        res = dict(name=name, pool_kind=pool.kind, images=n_img)
        for key in ("obj", "bg"):
            hh = hist[key]
            res[key] = dict(windows=int(hh.sum()), kappa_mean=float(kappa[key][0] / kappa[key][1]),
                            ratio_mean=float((hh * (LO + (1 - LO) * (torch.arange(NB) + 0.5) / NB)).sum() / hh.sum()),
                            **{f"p{int(q * 100):02d}": quantile(hh, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                            share_below_0_8=float(hh[:int((0.8 - LO) / (1 - LO) * NB)].sum() / hh.sum()))
        allh = hist["obj"] + hist["bg"]
        res["all"] = {f"p{int(q * 100):02d}": quantile(allh, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        res["_hist"] = dict(lo=LO, hi=1.0, bins=NB, obj=hist["obj"].tolist(), bg=hist["bg"].tolist())
        (out_dir / f"e1b_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "_hist"}, indent=1), flush=True)
        results[name] = res
        del model

    fig, ax = plt.subplots(figsize=(6, 3.5))
    centres = [LO + (1 - LO) * (i + 0.5) / NB for i in range(NB)]
    for name, res in results.items():
        for key, ls in (("obj", "-"), ("bg", "--")):
            hh = torch.tensor(res["_hist"][key])
            ax.plot(centres, (hh / hh.sum()).tolist(), ls, label=f"{name} ({res['pool_kind']}), {key}")
    ax.set_xlabel("σ₁ / ‖A‖_F")
    ax.set_ylabel("частка вікон")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "e1b_sigma1_frobenius.png", dpi=150)


if __name__ == "__main__":
    main()
