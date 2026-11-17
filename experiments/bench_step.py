"""Training-step speed (forward + backward, batch 8x3x640x640) per pooling kind, FP32 and AMP.

    python experiments/bench_step.py [--kinds max sigma1_fast sigma1_svd] [--warmup 10] [--iters 50]

Times the full detector step and, separately, the 6 pooling calls (fwd+bwd on the real pooling input shapes).
CUDA events; result -> experiments/results/00_smoke/bench_step.csv.
"""
import argparse
import csv
from pathlib import Path

import torch

from ultralytics.nn.modules.custom_pool import WindowPool
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
CFG = {"max": "lite", "sigma1_svd": "lite-svd", "sigma1_fast": "lite-svd-f"}

p = argparse.ArgumentParser()
p.add_argument("--kinds", nargs="+", default=["max", "sigma1_fast", "sigma1_svd"])
p.add_argument("--batch", type=int, default=8)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iters", type=int, default=50)
a = p.parse_args()
dev = torch.device("cuda")


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts = torch.tensor(ts)
    return ts.median().item(), ts.quantile(0.9).item()


rows = []
x = torch.randn(a.batch, 3, 640, 640, device=dev)
# pooling inputs in PoolBiFPN (c=96): P2 160x160, P3 80x80, P4 40x40, x2 repeats
pool_in = [torch.randn(a.batch, 96, s, s, device=dev, requires_grad=True) for s in (160, 80, 40) * 2]
for kind in a.kinds:
    m = DetectionModel(str(ROOT / "experiments" / "configs" / f"yolov8-epb-{CFG[kind]}.yaml"), nc=5, verbose=False).to(dev)
    m.model[1].pool.fallback = "strict"            # a silent fallback must not distort the timing
    m.train()
    pool = WindowPool(kind, "strict")
    for amp in (False, True):
        if kind == "sigma1_svd" and amp:
            continue                               # no fp16 SVD kernel: strict mode raises (trap 1)

        def step():
            with torch.autocast("cuda", enabled=amp):
                out = m(x)
                loss = sum(o.float().sum() for o in out)
            loss.backward()

        def pool_step():
            with torch.autocast("cuda", enabled=amp):
                s = sum(pool(t).float().sum() for t in pool_in)
            s.backward()

        torch.cuda.reset_peak_memory_stats()
        med, p90 = timeit(step, a.warmup, a.iters)
        mem = torch.cuda.max_memory_allocated() / 2**30
        pmed, pp90 = timeit(pool_step, a.warmup, a.iters)
        m.zero_grad(set_to_none=True)
        rows.append(dict(kind=kind, amp=amp, step_ms_median=round(med, 1), step_ms_p90=round(p90, 1),
                         pool6_ms_median=round(pmed, 1), pool6_ms_p90=round(pp90, 1), peak_vram_gb=round(mem, 2)))
        print(rows[-1], flush=True)
    del m
    torch.cuda.empty_cache()

out = ROOT / "experiments" / "results" / "00_smoke" / "bench_step.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "->", out)
