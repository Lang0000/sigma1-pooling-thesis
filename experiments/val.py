"""Validate runs/<name>/weights/best.pt in FP32 (half=False, trap 1) on valid and test -> experiments/results/<name>/metrics.csv.

    python experiments/val.py --name A1_Lite [--weights best|last] [--splits val test]
    python experiments/val.py --name F2_Lite-SVD-F_AMP --kind sigma1_svd     # gate G3: same weights, full SVD

One row per split x class (plus 'all'): P, R, mAP50, mAP50-95. Also checks that the pooling did not fall back.
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--weights", default="best", choices=["best", "last"])
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--workers", type=int, default=4)  # Windows: each worker is a process with torch (page file)
    p.add_argument("--kind", default=None, help="override WindowPool.kind, e.g. sigma1_svd for gate G3 -> metrics_<kind>.csv")
    a = p.parse_args()

    model = YOLO(str(ROOT / "runs" / a.name / "weights" / f"{a.weights}.pt"))
    pool = model.model.model[1].pool
    if a.kind:
        pool.kind, pool.fallback = a.kind, "strict"
    rows = []
    for split in a.splits:
        m = model.val(data=a.data, split=split, half=False, batch=8, imgsz=640, workers=a.workers, plots=True,
                      project=str(ROOT / "runs" / a.name), name=f"val_{split}" + (f"_{a.kind}" if a.kind else ""),
                      exist_ok=True)
        b = m.box
        rows.append(dict(split=split, cls="all", P=b.mp, R=b.mr, mAP50=b.map50, mAP50_95=b.map))
        for i, c in enumerate(m.ap_class_index):
            rows.append(dict(split=split, cls=m.names[int(c)], P=b.p[i], R=b.r[i], mAP50=b.ap50[i], mAP50_95=b.ap[i]))
    out = ROOT / "experiments" / "results" / a.name / (f"metrics_{a.kind}.csv" if a.kind else "metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows({k: (round(float(v), 4) if isinstance(v, float) or hasattr(v, "item") else v) for k, v in r.items()}
                    for r in rows)
    print(f"pool {pool.kind}: fallbacks {pool.fallbacks} / calls {pool.calls}")
    for r in rows:
        print(f"{r['split']:5s} {r['cls']:14s} P {r['P']:.4f}  R {r['R']:.4f}  mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50_95']:.4f}")
    print("->", out)


if __name__ == "__main__":
    main()
