"""E4: paired bootstrap over images of the difference of P, R, mAP50, mAP50-95 between two models (val and test).

    python experiments/e4_bootstrap.py dump --name R5_Lite-matched          # per-image stats of best.pt, FP32
    python experiments/e4_bootstrap.py dump --name F2_Lite-SVD-F_AMP
    python experiments/e4_bootstrap.py boot --pairs F2_Lite-SVD-F_AMP:R5_Lite-matched F1_Lite-SVD-F_FP32:R5_Lite-matched
    python experiments/e4_bootstrap.py boot --tag seeds --pairs F2+F2_s1+F2_s2:R5+R5_s1+R5_s2   # mean over seeds

dump: the standard Ultralytics validation (half=False, conf 0.001, iou 0.7, as val.py); the per-image statistics that
      DetMetrics accumulates (tp at 10 IoU thresholds, conf, pred_cls, target_cls) are saved, in dataloader order (no
      shuffle -> the same image order for every model; checked through target_cls).
      -> experiments/results/E4/stats_<name>_<split>.pkl
boot: the same B resamples of image indices for both models of a pair; metrics are recomputed with ap_per_class,
      exactly as Ultralytics does (P, R at the max-F1 confidence, averaged over classes).
      Point estimate = metrics on all images (equals val.py). 95% percentile CI; two-sided p = share of resamples whose
      difference has the other sign, x2.
      -> experiments/results/E4/e4_<split>.csv
"""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"
OUT = ROOT / "experiments" / "results" / "E4"
METRICS = ("P", "R", "mAP50", "mAP50_95")


def dump(a):
    from ultralytics import YOLO
    from ultralytics.utils.metrics import DetMetrics

    per_image = []
    orig = DetMetrics.update_stats

    def update_stats(self, stat):
        per_image.append({k: np.asarray(v) for k, v in stat.items()})
        orig(self, stat)

    DetMetrics.update_stats = update_stats
    model = YOLO(str(ROOT / "runs" / a.name / "weights" / "best.pt"))
    pool = model.model.model[1].pool
    OUT.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        per_image.clear()
        m = model.val(data=a.data, split=split, half=False, batch=8, imgsz=640, workers=a.workers, plots=False,
                      project=str(ROOT / "runs" / a.name), name=f"e4_{split}", exist_ok=True)
        b = m.box
        point = dict(P=b.mp, R=b.mr, mAP50=b.map50, mAP50_95=b.map)
        with open(OUT / f"stats_{a.name}_{split}.pkl", "wb") as f:
            pickle.dump(dict(name=a.name, split=split, point=point, images=list(per_image)), f)
        print(split, len(per_image), "images", {k: round(float(v), 4) for k, v in point.items()})
    print(f"pool {pool.kind}: fallbacks {pool.fallbacks} / calls {pool.calls}")


def metrics(imgs, idx, nc):
    from ultralytics.utils.metrics import Metric, ap_per_class

    cat = {k: np.concatenate([imgs[i][k] for i in idx], 0) for k in ("tp", "conf", "pred_cls", "target_cls")}
    m = Metric()
    m.nc = nc
    m.update(ap_per_class(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[2:])
    return np.array([m.mp, m.mr, m.map50, m.map])


def boot(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for split in a.splits:
        # a side of a pair may be a group "X+Y+Z" (e.g. 3 seeds): its metric = mean over members on the same resample
        names = sorted({n for p in a.pairs for side in p.split(":") for n in side.split("+")})
        st = {n: pickle.load(open(OUT / f"stats_{n}_{split}.pkl", "rb")) for n in names}
        imgs = {n: s["images"] for n, s in st.items()}
        n_img = len(imgs[names[0]])
        for n in names:   # same images in the same order for every model
            assert len(imgs[n]) == n_img and all(np.array_equal(x["target_cls"], y["target_cls"])
                                                 for x, y in zip(imgs[n], imgs[names[0]])), n
        nc = 5
        full = {n: metrics(imgs[n], range(n_img), nc) for n in names}
        idx = rng.integers(0, n_img, size=(a.B, n_img))
        res = {n: np.empty((a.B, 4)) for n in names}
        for b in range(a.B):
            for n in names:
                res[n][b] = metrics(imgs[n], idx[b], nc)
            if b % 1000 == 0:
                print(f"{split}: resample {b}/{a.B}", flush=True)
        for side in {side for p in a.pairs for side in p.split(":") if "+" in side}:
            full[side] = np.mean([full[n] for n in side.split("+")], 0)
            res[side] = np.mean([res[n] for n in side.split("+")], 0)
        rows = []
        for p in a.pairs:
            x, y = p.split(":")
            d = res[x] - res[y]
            for j, k in enumerate(METRICS):
                lo, hi = np.percentile(d[:, j], [2.5, 97.5])
                delta = full[x][j] - full[y][j]
                pv = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
                rows.append(dict(split=split, a=x, b=y, metric=k, a_value=round(full[x][j], 4),
                                 b_value=round(full[y][j], 4), delta=round(delta, 4), ci95_lo=round(lo, 4),
                                 ci95_hi=round(hi, 4), p=round(pv, 4), B=a.B))
        with open(OUT / (f"e4_{a.tag}_{split}.csv" if a.tag else f"e4_{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"{split:4s} {r['a']} - {r['b']} {r['metric']:9s} {r['delta']:+.4f} "
                  f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] p={r['p']:.4f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--name", required=True)
    d.add_argument("--data", default=str(DATA))
    d.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("boot")
    b.add_argument("--pairs", nargs="+", required=True, help="A:B -> A - B")
    b.add_argument("--B", type=int, default=10000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tag", default="", help="-> e4_<tag>_<split>.csv (separate tables, e.g. seeds, R6)")
    for s in (d, b):
        s.add_argument("--splits", nargs="+", default=["val", "test"])
    a = p.parse_args()
    dump(a) if a.cmd == "dump" else boot(a)


if __name__ == "__main__":
    main()
