"""Environment + model build check: run from the repo root.

Expected: CUDA available; 4 556 695 parameters for all three models, stride [8, 16, 32], kinds max / sigma1_svd / sigma1_fast.
"""
from pathlib import Path

import torch

import ultralytics
from ultralytics.nn.tasks import DetectionModel

if __name__ == "__main__":
    cfg = Path(__file__).resolve().parent / "configs"
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    print("ultralytics", ultralytics.__version__, ultralytics.__file__)
    ok = True
    for k, kind in (("lite", "max"), ("lite-svd", "sigma1_svd"), ("lite-svd-f", "sigma1_fast")):
        m = DetectionModel(str(cfg / f"yolov8-epb-{k}.yaml"), nc=5, verbose=False)
        n = sum(p.numel() for p in m.parameters())
        good = n == 4556695 and m.stride.tolist() == [8, 16, 32] and m.model[1].pool.kind == kind
        ok &= good
        print(f"{k:12s} {n:>9,d} params  stride {m.stride.tolist()}  pool {m.model[1].pool.kind:12s} {'OK' if good else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)
