"""Train one model variant with the exact hyper-parameters of the article runs.

    python experiments/train.py --model lite        --args lite        --name A1_Lite
    python experiments/train.py --model lite-svd    --args svd         --name A2_Lite-SVD
    python experiments/train.py --model lite-svd-f  --args svd         --name F1_Lite-SVD-F_FP32
    python experiments/train.py --model lite-svd-f  --args svd --amp   --name F2_Lite-SVD-F_AMP
    python experiments/train.py --model lite        --args svd         --name R5_Lite-matched

Windows: same commands (PowerShell, `python experiments\\train.py ...`).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "experiments" / "configs"
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"

# args.yaml of YOLOv8-EPB-Lite (article): AMP, custom augmentations, patience 20
LITE_ARGS = dict(amp=True, epochs=100, batch=8, imgsz=640, patience=20, seed=0, deterministic=True,
                 mosaic=0.0, mixup=0.1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=5.0)
# args.yaml of YOLOv8-EPB-Lite-SVD (article): FP32, Ultralytics default augmentations, patience 100
SVD_ARGS = dict(amp=False, epochs=100, batch=8, imgsz=640, patience=100, seed=0, deterministic=True,
                mosaic=1.0, mixup=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0)


def main():
    # the guard below is required on Windows: DataLoader workers are spawned and re-import this module
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="lite | lite-svd | lite-svd-f | any yolov8-epb-<x>.yaml in configs/")
    p.add_argument("--args", required=True, choices=["lite", "svd"])
    p.add_argument("--amp", action="store_true", help="override amp=True")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="override epochs (smoke tests)")
    p.add_argument("--fraction", type=float, default=None, help="fraction of train set (smoke tests)")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default 8); 4 for two parallel runs")
    p.add_argument("--name", required=True)
    p.add_argument("--resume", action="store_true", help="continue runs/<name>/weights/last.pt (args come from it)")
    p.add_argument("--close-mosaic", type=int, default=None,
                   help="resume only: 0 skips the dataloader rebuild at epoch 90 (a no-op for mosaic=0 runs, e.g. --args lite; "
                        "on Windows the rebuild spawns a second set of workers and can exhaust the page file)")
    a = p.parse_args()

    args = dict(LITE_ARGS if a.args == "lite" else SVD_ARGS)
    if a.amp:
        args["amp"] = True
    for k in ("seed", "epochs", "fraction", "workers"):
        if getattr(a, k) is not None:
            args[k] = getattr(a, k)

    # gate G4: non-finite losses and optimizer steps skipped by GradScaler (each skip halves the scale)
    steps = dict(nan_batches=0, amp_skipped_steps=0, amp_final_scale=None)
    prev_scale = [None]

    def on_batch_end(trainer):
        if not torch.isfinite(trainer.loss).all():
            steps["nan_batches"] += 1
        if trainer.amp:
            s = trainer.scaler.get_scale()
            if prev_scale[0] is not None and s < prev_scale[0]:
                steps["amp_skipped_steps"] += 1
            prev_scale[0] = steps["amp_final_scale"] = s

    t0 = time.time()
    if a.resume:  # interrupted run: optimizer, EMA and epoch are restored from last.pt (RNG state is not)
        model = YOLO(str(ROOT / "runs" / a.name / "weights" / "last.pt"))
        model.add_callback("on_train_batch_end", on_batch_end)
        model.train(resume=True, **({} if a.close_mosaic is None else {"close_mosaic": a.close_mosaic}))
    else:
        model = YOLO(str(CFG / f"yolov8-epb-{a.model}.yaml"))
        model.add_callback("on_train_batch_end", on_batch_end)
        model.train(data=a.data, project=str(ROOT / "runs"), name=a.name, exist_ok=False, **args)
    # after train() Ultralytics reloads model.model from best.pt (fresh counters), so read the trainer's own models:
    # train = the model that was optimised, ema = the copy used for validation and saved to best.pt
    pools = {"train": unwrap_model(model.trainer.model).model[1].pool, "ema": model.trainer.ema.ema.model[1].pool}
    log = dict(name=a.name, model=a.model, args=args, hours=round((time.time() - t0) / 3600, 3), resumed=a.resume,
               pool_kind=pools["train"].kind, pool_fallbacks=sum(q.fallbacks for q in pools.values()),
               pool_calls={k: q.calls for k, q in pools.items()},
               pool_fallbacks_by_model={k: q.fallbacks for k, q in pools.items()}, **steps)
    with open(ROOT / "experiments" / "train_log.jsonl", "a") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    print(log)


if __name__ == "__main__":
    main()
