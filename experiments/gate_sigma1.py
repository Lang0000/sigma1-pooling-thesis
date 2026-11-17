"""Gates G1 + G2 for sigma1_fast on real activations of the valid split.

    python experiments/gate_sigma1.py --weights runs/A1_Lite/weights/best.pt --kind sigma1_fast --tag A1
    python experiments/gate_sigma1.py --weights runs/F2_Lite-SVD-F_AMP/weights/best.pt --tag F2

The network runs in FP32 eval mode with WindowPool.kind = --kind (the pool is shared by all 6 calls); every pooling input
is taken with a forward hook.
G1: sigma1_fast (fp32) vs torch.linalg.svdvals in float64 on ALL windows; separately padding windows, near-degenerate
    windows (sigma2/sigma1 > 0.99) and zero windows. Pass: max rel. error <= 1e-5, NaN/Inf = 0.
G2: on the first --g2-batches batches the gradient of the real detection loss w.r.t. each pooling output (g) is pulled
    back through (a) sigma1_fast fp32, (b) sigma1_fast with an fp16 input under autocast (what AMP training does),
    (c) reference: svdvals in float64 of the same input (fp32 or fp16-rounded). Per map: cosine and
    ||d||_inf / ||ref||_inf. Pass (a, b): mean cos >= 0.9999, map error <= 1e-3, NaN/Inf = 0.
    fp16_input (not a gate): fp64 gradient of the fp16-rounded vs the fp32 activation, i.e. the cost of AMP itself.
-> experiments/results/F0_gate/g1_g2_<tag>.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.custom_pool import K, P, S, _Sigma1Fast, _windows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"
CHUNK = 2_000_000  # windows per svdvals call


class G1Stats:
    """Running max/mean of the relative error of sigma1 over window groups."""

    GROUPS = ("all", "padding", "near_degenerate", "zero")

    def __init__(self):
        self.s = {g: dict(n=0, max_rel=0.0, sum_rel=0.0, max_abs=0.0) for g in self.GROUPS}
        self.nonfinite = 0

    def add(self, group, rel, abs_):
        st = self.s[group]
        st["n"] += rel.numel()
        if rel.numel():
            st["max_rel"] = max(st["max_rel"], rel.max().item())
            st["sum_rel"] += rel.sum().item()
            st["max_abs"] = max(st["max_abs"], abs_.max().item())

    def result(self):
        out = {g: dict(n=v["n"], max_rel=v["max_rel"], mean_rel=v["sum_rel"] / max(v["n"], 1), max_abs=v["max_abs"])
               for g, v in self.s.items()}
        out["nonfinite"] = self.nonfinite
        return out


def padding_mask(x):
    """(B*C*L,) bool: windows that contain zero padding (same order as _windows)."""
    B, C, H, W = x.shape
    m = F.unfold(torch.ones(1, 1, H, W, device=x.device), K, padding=P, stride=S).amin(1) < 1   # 1, L
    return m.expand(B * C, -1).reshape(-1)


@torch.no_grad()
def g1_update(st, x):
    A, _ = _windows(x.float())
    pad = padding_mask(x)
    for i in range(0, A.shape[0], CHUNK):
        a = A[i:i + CHUNK]
        fast = _Sigma1Fast.apply(a).double()
        sv = torch.linalg.svdvals(a.double())
        ref, s2 = sv[:, 0], sv[:, 1]
        st.nonfinite += (~torch.isfinite(fast)).sum().item()
        err = (fast - ref).abs()
        zero = ref == 0
        rel = torch.where(zero, err, err / ref.clamp_min(1e-300))    # zero windows: absolute error
        st.add("all", rel, err)
        st.add("padding", rel[pad[i:i + CHUNK]], err[pad[i:i + CHUNK]])
        near = ~zero & (s2 / ref.clamp_min(1e-300) > 0.99)
        st.add("near_degenerate", rel[near], err[near])
        st.add("zero", err[zero], err[zero])


def pool_grad(x, g, how):
    """d<g, pool(x)>/dx for one pooling map."""
    if how == "fast_fp32":
        x = x.float().requires_grad_()
        y = _Sigma1Fast.apply(_windows(x)[0])
    elif how == "fast_amp":   # AMP: the activation arrives in fp16, WindowPool upcasts inside autocast(enabled=False)
        x = x.half().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.autocast("cuda", enabled=False):
                y = _Sigma1Fast.apply(_windows(x.float())[0])
    else:                     # reference
        x = x.double().requires_grad_()
        y = torch.linalg.svdvals(_windows(x)[0])[:, 0]
    scale = 2.0 ** 16 if how == "fast_amp" else 1.0  # GradScaler's initial scale: unscaled loss grads underflow in fp16
    (gx,) = torch.autograd.grad(y, x, g.reshape(-1).to(y.dtype) * scale)
    return gx.double() / scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--kind", default="sigma1_fast", help="WindowPool.kind used for the forward pass")
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--g2-batches", type=int, default=20, help="batches for G2 (20 x 8 = 160 images); 0 = skip G2")
    p.add_argument("--max-batches", type=int, default=0, help="limit G1 (0 = whole split)")
    p.add_argument("--data", default=str(DATA))
    a = p.parse_args()

    dev = torch.device("cuda")
    model = YOLO(a.weights).model.to(dev).float().eval()
    pool = model.model[1].pool
    trained_kind = pool.kind
    pool.kind, pool.fallback = a.kind, "strict"
    for q in model.parameters():
        q.requires_grad_(False)
    model.args = get_cfg(overrides={})        # box/cls/dfl gains for the loss (defaults, as in all our runs)

    cfg = get_cfg(overrides=dict(imgsz=640, batch=a.batch, workers=0))
    data = check_det_dataset(a.data)
    ds = build_yolo_dataset(cfg, data[a.split], a.batch, data, mode="val", stride=32)
    dl = build_dataloader(ds, a.batch, 0, shuffle=False)

    inputs, outputs = [], []

    def hook(m, inp, out):
        inputs.append(inp[0].detach())
        outputs.append(out)

    h = pool.register_forward_hook(hook)
    g1 = G1Stats()
    g2 = {k: dict(cos=[], err=[], nonfinite=0) for k in ("fast_fp32", "fast_amp", "fp16_input")}
    n_img = 0
    for bi, batch in enumerate(dl):
        if a.max_batches and bi >= a.max_batches:
            break
        inputs.clear(), outputs.clear()
        batch["img"] = batch["img"].to(dev).float() / 255
        for k in ("cls", "bboxes", "batch_idx"):
            batch[k] = batch[k].to(dev)
        n_img += batch["img"].shape[0]
        do_g2 = bi < a.g2_batches
        with torch.set_grad_enabled(do_g2):
            if do_g2:
                batch["img"].requires_grad_()   # params are frozen; make the graph reach the pooling outputs
                loss, _ = model.loss(batch)
                grads = torch.autograd.grad(loss.sum(), outputs)
            else:
                model(batch["img"])
        for i, x in enumerate(inputs):
            g1_update(g1, x)
            if do_g2:
                # fast_amp is compared with fp64 SVD of the same fp16-rounded activation (the algorithm's error);
                # fp16_input = what the fp16 rounding of the activation alone does to the gradient (for information)
                refs = {"fast_fp32": pool_grad(x, grads[i], "ref"), "fast_amp": pool_grad(x.half(), grads[i], "ref")}
                refs["fp16_input"] = refs["fast_fp32"]
                for how, st in g2.items():
                    ref = refs[how]
                    gx = refs["fast_amp"] if how == "fp16_input" else pool_grad(x, grads[i], how)
                    st["nonfinite"] += (~torch.isfinite(gx)).sum().item()
                    st["cos"].append(F.cosine_similarity(gx.flatten(), ref.flatten(), 0).item())
                    st["err"].append(((gx - ref).abs().max() / ref.abs().max().clamp_min(1e-300)).item())
        if bi % 10 == 0:
            r = g1.result()["all"]
            print(f"batch {bi}: images {n_img}, windows {r['n']:,}, max rel {r['max_rel']:.3e}", flush=True)
    h.remove()

    res = dict(weights=a.weights, trained_kind=trained_kind, kind=a.kind, split=a.split, images=n_img,
               G1=g1.result(), G2={k: dict(maps=len(v["cos"]), mean_cos=sum(v["cos"]) / max(len(v["cos"]), 1),
                                           min_cos=min(v["cos"], default=float("nan")),
                                           max_err=max(v["err"], default=float("nan")), nonfinite=v["nonfinite"])
                                   for k, v in g2.items()})
    r1 = res["G1"]
    res["G1_pass"] = r1["all"]["max_rel"] <= 1e-5 and r1["nonfinite"] == 0
    res["G2_pass"] = a.g2_batches > 0 and all(v["mean_cos"] >= 0.9999 and v["max_err"] <= 1e-3 and v["nonfinite"] == 0
                                              for k, v in res["G2"].items() if k != "fp16_input")
    out = ROOT / "experiments" / "results" / "F0_gate" / f"g1_g2_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
