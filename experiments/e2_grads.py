"""E2: gradient of sigma1 pooling on real windows and the real detection loss, in the precision modes of training.

    python experiments/e2_grads.py --weights runs/F2_Lite-SVD-F_AMP/weights/best.pt --tag F2 [--batches 13]

Network in FP32 eval; for every pooling call the gradient g of the real loss w.r.t. the pooling output is pulled back to
the pooling input through the modes below and compared with the reference: torch.linalg.svdvals in float64 of the fp32
activation with the fp32 g. Per map: cosine, ||d||_inf / ||ref||_inf, NaN/Inf; also the share of upstream gradient
entries that are non-zero but below the fp16 normal minimum 2^-14 (underflow without loss scaling).
  fp32               sigma1_fast, fp32 input, fp32 g
  fp16_in            input rounded to fp16, sigma1_fast in fp32 (what AMP training with our layer does), fp32 g
  bf16_in            input rounded to bf16, sigma1_fast in fp32
  fp16_g_unscaled    fp32 input, g rounded to fp16 without loss scaling
  fp16_g_scaled      fp32 input, g * 2^16 rounded to fp16, then / 2^16 (GradScaler's initial scale)
  maxpool            gradient of MaxPool2d(3, 2, 1) (what the article's AMP run actually trained: silent fallback)
-> experiments/results/E2/e2_<tag>.json
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
MODES = ("fp32", "fp16_in", "bf16_in", "fp16_g_unscaled", "fp16_g_scaled", "maxpool")


def grad_through(x, g, how):
    if how == "maxpool":
        x = x.double().requires_grad_()
        y = F.max_pool2d(x, K, S, P)
        (gx,) = torch.autograd.grad(y, x, g.double())
        return gx
    if how == "ref":
        x = x.double().requires_grad_()
        y = torch.linalg.svdvals(_windows(x)[0])[:, 0]
        (gx,) = torch.autograd.grad(y, x, g.reshape(-1).double())
        return gx
    xin = {"fp16_in": x.half().float(), "bf16_in": x.bfloat16().float()}.get(how, x.float())
    gg = {"fp16_g_unscaled": g.half().float(), "fp16_g_scaled": (g * 2.0 ** 16).half().float() / 2.0 ** 16}.get(how, g.float())
    xin = xin.clone().requires_grad_()
    y = _Sigma1Fast.apply(_windows(xin)[0])
    (gx,) = torch.autograd.grad(y, xin, gg.reshape(-1))
    return gx.double()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--batches", type=int, default=13, help="13 x 8 = 104 val images")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--data", default=str(DATA))
    a = p.parse_args()
    dev = torch.device("cuda")
    model = YOLO(a.weights).model.to(dev).float().eval()
    pool = model.model[1].pool
    for q in model.parameters():
        q.requires_grad_(False)
    model.args = get_cfg(overrides={})
    cfg = get_cfg(overrides=dict(imgsz=640, batch=a.batch, workers=0))
    data = check_det_dataset(a.data)
    ds = build_yolo_dataset(cfg, data["val"], a.batch, data, mode="val", stride=32)
    dl = build_dataloader(ds, a.batch, 0, shuffle=False)
    inputs, outputs = [], []
    def hook(m, inp, out):  # must return None (a returned value would replace the output)
        inputs.append(inp[0].detach())
        outputs.append(out)

    h = pool.register_forward_hook(hook)
    st = {k: dict(cos=[], err=[], nonfinite=0) for k in MODES}
    g_nz, g_under, windows, n_img = 0, 0, 0, 0
    for bi, batch in enumerate(dl):
        if bi >= a.batches:
            break
        inputs.clear(), outputs.clear()
        batch["img"] = batch["img"].to(dev).float().div(255).requires_grad_()
        for k in ("cls", "bboxes", "batch_idx"):
            batch[k] = batch[k].to(dev)
        n_img += batch["img"].shape[0]
        loss, _ = model.loss(batch)
        grads = torch.autograd.grad(loss.sum(), outputs)
        for x, g in zip(inputs, grads):
            windows += g.numel()
            nz = g != 0
            g_nz += nz.sum().item()
            g_under += (nz & (g.abs() < 2.0 ** -14)).sum().item()
            ref = grad_through(x, g, "ref")
            for how in MODES:
                gx = grad_through(x, g, how)
                s = st[how]
                s["nonfinite"] += (~torch.isfinite(gx)).sum().item()
                s["cos"].append(F.cosine_similarity(gx.flatten(), ref.flatten(), 0).item())
                s["err"].append(((gx - ref).abs().max() / ref.abs().max().clamp_min(1e-300)).item())
        print(f"batch {bi}: images {n_img}, windows {windows:,}", flush=True)
    h.remove()
    res = dict(weights=a.weights, pool_kind=pool.kind, images=n_img, windows=windows,
               upstream_grad_nonzero=g_nz, upstream_grad_below_fp16_min_share=g_under / max(g_nz, 1),
               modes={k: dict(maps=len(v["cos"]), mean_cos=sum(v["cos"]) / len(v["cos"]), min_cos=min(v["cos"]),
                              median_rel_err=sorted(v["err"])[len(v["err"]) // 2], max_rel_err=max(v["err"]),
                              nonfinite=v["nonfinite"]) for k, v in st.items()})
    out = ROOT / "experiments" / "results" / "E2" / f"e2_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
