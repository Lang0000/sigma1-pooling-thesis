"""E3: efficiency of Lite (MaxPool), Lite-SVD (full SVD), Lite-SVD-F (analytic sigma1) and the R9 controls.

    python experiments/e3_efficiency.py [--iters 1000] [--repeats 3] [--models Lite Lite-SVD ...]

Protocol: batch 1, 640x640 (a real valid image, letterboxed), CUDA events, warm-up 50,
--repeats x --iters timed forwards; slow models (> 50 ms/forward) get as many iterations as fit in 60 s per repeat
(>= 100). FP32 and FP16 inference (model.half(); Lite-SVD in FP16 uses sigma1_svd_fp32 - torch has no fp16 SVD).
  latency      forward only (fused Conv+BN), median / mean / p95 over all timed forwards, ms
  pool_ms      time of the 6 WindowPool calls per forward (CUDA events in pre/post hooks, separate pass)
  e2e          YOLO.predict on 200 valid images (640x512 letterbox), batch 1: preprocess + inference + NMS, ms and FPS;
               b20 = throughput with batches of 20 images, ms per image
  energy       board power (pynvml, 10 ms sampling) integrated over the timed loop, J/image; also above idle power
  vram_inf     peak torch.cuda.max_memory_allocated during inference, MB
  train step   batch 8 (real batch), fwd + loss + bwd, FP32 and AMP (autocast; Lite-SVD: sigma1_svd_fp32): ms, peak MB
               after 5 warm-up steps (cudnn.benchmark off), 20 timed steps (Lite-SVD FP32: 5)
  GFLOPs       thop (convolutions etc.) + pooling counted analytically (thop does not see unfold/SVD/Jacobi)
  params, s/epoch (from runs/<ID>/results.csv of the 100-epoch run)
-> experiments/results/E3/e3.csv, e3.json
"""
import argparse
import csv
import json
import statistics
import threading
import time
from copy import deepcopy
from pathlib import Path

import pynvml
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"
MODELS = {  # name -> (run with weights, run whose results.csv gives s/epoch, precision of that run)
    "Lite": ("R5_Lite-matched", "R5_Lite-matched", "FP32"),
    "Lite-SVD": ("A2_Lite-SVD", "A2_Lite-SVD", "FP32"),
    "Lite-SVD-F": ("F2_Lite-SVD-F_AMP", "F2_Lite-SVD-F_AMP", "AMP"),
    "Lite-SVD-F (FP32 run)": ("F1_Lite-SVD-F_FP32", "F1_Lite-SVD-F_FP32", "FP32"),
    "Lite-MaxAbs": ("R9_Lite-MaxAbs", "R9_Lite-MaxAbs", "FP32"),
    "Lite-Frobenius": ("R9_Lite-Frobenius", "R9_Lite-Frobenius", "FP32"),
}
# FLOPs per 3x3 window (multiply/add/compare/sqrt = 1): max 8 cmp; maxabs 9 abs + 8 cmp; frobenius 9 mul + 8 add + 1 sqrt;
# sigma1_fast: A^T A (6 unique entries x 5) + 3 Jacobi sweeps x 3 rotations (~45 each incl. eigenvectors) + argmax 2
# + sqrt + A v (15) + u = A v / s (4) = ~440; sigma1_svd: Golub-Kahan SVD of 3x3, ~4 m n^2 + 8 n^3 = 324 (lower bound,
# cuSOLVER's batched Jacobi does more).
POOL_FLOPS = {"max": 8, "maxabs": 17, "frobenius": 18, "sigma1_fast": 440, "sigma1_svd": 324, "sigma1_svd_fp32": 324}


class Power:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples, self.run = [], False

    def _loop(self):
        while self.run:
            self.samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000))
            time.sleep(0.01)

    def start(self):
        self.samples, self.run = [], True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.run = False
        self.t.join()
        s = self.samples
        energy = sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(s, s[1:]))
        return energy, statistics.mean(p for _, p in s) if s else float("nan")


def timed(model, x, n):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        model(x)
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev]


def pool_time(model, x, n):
    pool = model.model[1].pool
    marks = []

    def pre(m, inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        marks.append(e)

    def post(m, inp, out):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        marks.append(e)

    h1, h2 = pool.register_forward_pre_hook(pre), pool.register_forward_hook(post)
    for _ in range(n):
        model(x)
    torch.cuda.synchronize()
    h1.remove(), h2.remove()
    per_call = [a.elapsed_time(b) for a, b in zip(marks[::2], marks[1::2])]
    calls = len(per_call) // n
    return sum(per_call) / n, calls


def pool_windows(model, x):
    pool, n = model.model[1].pool, []
    h = pool.register_forward_hook(lambda m, i, o: n.append(o.numel()) and None)
    model(x)
    h.remove()
    return sum(n)


def end_to_end(w, kind, prec, images):
    """Ultralytics predictor (letterbox, normalise, forward, NMS) on real images.
    batch 1: one predict() call per image (a list passed to predict() is stacked into ONE batch);
    batch 20: throughput, ms per image."""
    y = YOLO(str(w))
    y.model.model[1].pool.kind = kind
    kw = dict(half=prec == "FP16", imgsz=640, verbose=False)
    for im in images[:10]:
        y.predict(im, **kw)                                   # warm-up
    sp = [y.predict(im, **kw)[0].speed for im in images]
    out = {f"e2e_{k}_ms": round(statistics.mean(s[v] for s in sp), 3)
           for k, v in (("pre", "preprocess"), ("inf", "inference"), ("nms", "postprocess"))}
    out["e2e_ms"] = round(sum(out.values()), 3)
    out["fps_e2e"] = round(1000 / out["e2e_ms"], 1)
    y.predict(images[:20], **kw)                              # warm-up of batch 20
    sp = [r.speed for i in range(0, len(images), 20) for r in y.predict(images[i:i + 20], **kw)]
    b20 = sum(statistics.mean(s[k] for s in sp) for k in ("preprocess", "inference", "postprocess"))
    out["b20_ms_per_img"] = round(b20, 3)
    out["fps_b20"] = round(1000 / b20, 1)
    del y
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--e2e-images", type=int, default=200)
    p.add_argument("--e2e-only", action="store_true", help="recompute only the end-to-end columns of e3.csv")
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    data = check_det_dataset(str(DATA))
    if a.e2e_only:
        images = sorted(str(f) for f in Path(data["val"]).glob("*.jpg"))[:a.e2e_images]
        out = ROOT / "experiments" / "results" / "E3"
        rows = list(csv.DictReader(open(out / "e3.csv")))
        for r in rows:
            r.update(end_to_end(ROOT / "runs" / r["run"] / "weights" / "best.pt", r["pool_kind"], r["precision"], images))
            print(r["model"], r["precision"], r["e2e_ms"], r["fps_e2e"], r["fps_b20"], flush=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "e3.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        js = json.loads((out / "e3.json").read_text(encoding="utf-8"))
        js["rows"] = rows
        (out / "e3.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
        return
    cfg = get_cfg(overrides=dict(imgsz=640, batch=8, workers=0))
    ds = build_yolo_dataset(cfg, data["val"], 8, data, mode="val", stride=32)
    batch8 = next(iter(build_dataloader(ds, 8, 0, shuffle=False)))
    x1 = (batch8["img"][:1].to(dev).float() / 255).contiguous()
    images = sorted(str(f) for f in Path(data["val"]).glob("*.jpg"))[:a.e2e_images]
    power = Power()
    time.sleep(2)
    power.start(); time.sleep(5); _, idle_w = power.stop()
    print(f"idle board power {idle_w:.1f} W", flush=True)
    rows = []
    for name in a.models:
        run, run_ep, run_prec = MODELS[name]
        w = ROOT / "runs" / run / "weights" / "best.pt"
        base = YOLO(str(w)).model.to(dev).float().eval()
        kind = base.model[1].pool.kind
        params = sum(q.numel() for q in base.parameters())
        gflops_thop = get_flops(deepcopy(base), 640)
        win = pool_windows(base, x1)
        res_csv = list(csv.DictReader(open(ROOT / "runs" / run_ep / "results.csv")))
        s_ep = float(res_csv[-1]["time"].strip()) / len(res_csv)
        for prec in ("FP32", "FP16"):
            torch.backends.cudnn.benchmark = True
            m = deepcopy(base).fuse(verbose=False)
            x = x1
            if prec == "FP16":
                m, x = m.half(), x1.half()
                if kind == "sigma1_svd":
                    m.model[1].pool.kind = "sigma1_svd_fp32"
            pk = m.model[1].pool.kind
            with torch.inference_mode():
                warm = timed(m, x, 50)
                lat0 = statistics.median(warm[10:])
                n = a.iters if lat0 <= 50 else max(100, int(60_000 / lat0))
                torch.cuda.reset_peak_memory_stats()
                lat = []
                power.start()
                t0 = time.perf_counter()
                for _ in range(a.repeats):
                    lat += timed(m, x, n)
                wall = time.perf_counter() - t0
                energy, mean_w = power.stop()
                vram_inf = torch.cuda.max_memory_allocated() / 2 ** 20
                pms, calls = pool_time(m, x, min(n, 200))
            lat.sort()
            e2e = end_to_end(w, pk, prec, images)
            row = dict(model=name, run=run, pool_kind=pk, precision=prec, params=params,
                       gflops_thop=round(gflops_thop, 3),
                       pool_windows=win, pool_gflops=round(win * POOL_FLOPS[pk] / 1e9, 3),
                       gflops_total=round(gflops_thop + win * POOL_FLOPS[pk] / 1e9, 3),
                       iters=n * a.repeats, lat_median_ms=round(lat[len(lat) // 2], 3),
                       lat_mean_ms=round(statistics.mean(lat), 3), lat_p95_ms=round(lat[int(0.95 * len(lat))], 3),
                       repeat_medians_ms=[round(statistics.median(lat_r), 3) for lat_r in
                                          [lat[i::a.repeats] for i in range(a.repeats)]],
                       pool_ms=round(pms, 3), pool_calls=calls, pool_share=round(pms / statistics.mean(lat), 3),
                       **e2e, fps_forward=round(1000 / statistics.mean(lat), 1),
                       energy_J_per_img=round(energy / (n * a.repeats), 4),
                       energy_above_idle_J_per_img=round((energy - idle_w * wall) / (n * a.repeats), 4),
                       power_mean_W=round(mean_w, 1),
                       idle_W=round(idle_w, 1), vram_inf_MB=round(vram_inf, 1),
                       s_per_epoch=round(s_ep, 1), s_per_epoch_run=f"{run_ep} ({run_prec})")
            print({k: row[k] for k in ("model", "precision", "pool_kind", "lat_median_ms", "pool_ms", "fps_e2e",
                                       "energy_J_per_img", "vram_inf_MB")}, flush=True)
            del m
            torch.cuda.empty_cache()
            rows.append(row)
        # training step: batch 8, fwd + loss + bwd
        for prec in ("FP32", "AMP"):
            m = YOLO(str(w)).model.to(dev).float().train()
            m.args = get_cfg(overrides={})
            if prec == "AMP" and kind == "sigma1_svd":
                m.model[1].pool.kind = "sigma1_svd_fp32"
            for q in m.parameters():
                q.requires_grad_(True)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch8.items()}
            b["img"] = b["img"].float() / 255
            ts = []
            torch.backends.cudnn.benchmark = False   # benchmark's algorithm search allocates large workspaces
            steps = 5 if kind == "sigma1_svd" and prec == "FP32" else 20
            for i in range(5 + steps):
                if i == 5:   # peak memory of the steady state, not of the warm-up
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", enabled=prec == "AMP"):
                    loss, _ = m.loss(b)
                loss.sum().backward()
                m.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i >= 5:
                    ts.append((time.perf_counter() - t0) * 1000)
            for r in rows:
                if r["model"] == name:
                    r[f"train_step_{prec}_ms"] = round(statistics.median(ts), 1)
                    r[f"train_vram_{prec}_MB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            print(f"{name} train {prec}: {statistics.median(ts):.1f} ms/step, "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB", flush=True)
            del m
            torch.cuda.empty_cache()
    out = ROOT / "experiments" / "results" / "E3"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "e3.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    gpu = torch.cuda.get_device_name(0)
    (out / "e3.json").write_text(json.dumps(dict(gpu=gpu, torch=torch.__version__, idle_W=idle_w, rows=rows), indent=1),
                                 encoding="utf-8")
    print("->", out / "e3.csv")


if __name__ == "__main__":
    main()
