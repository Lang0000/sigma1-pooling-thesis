"""E5: error structure - TP / FN / FP by class and object size, background -> class false positives.

    python experiments/e5_errors.py --names R5_Lite-matched F2_Lite-SVD-F_AMP ... [--data ...] [--tag orig]

best.pt, FP32, conf 0.25 (Ultralytics confusion-matrix threshold), NMS IoU 0.7, val and test. Per image predictions are
matched greedily by confidence at IoU >= 0.5:
  TP          same class, unmatched GT
  FP_class    no same-class GT left, but a GT of another class at IoU >= 0.5 (class confusion)
  FP_bg       no GT at IoU >= 0.5 (background -> class, "false alarm on background")
  FN_missed   GT without any prediction at IoU >= 0.5;  FN_class   GT covered only by a wrong-class prediction
Size of a GT: area in pixels of the original image: small < 16^2, medium < 32^2, large >= 32^2 (thermal UAV objects are
tiny; COCO's 32^2 / 96^2 would put almost all in "small").
-> experiments/results/E5/e5_<tag>.csv (one row per model x split x class|size bucket)
"""
import argparse
import csv
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import box_iou

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "huv-termal.v2i.yolov8" / "data.yaml"
SIZES = (("small", 16 ** 2), ("medium", 32 ** 2), ("large", float("inf")))


def xywhn2xyxy(b, w, h):
    x, y, bw, bh = b.unbind(-1)
    return torch.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], -1)


def size_of(area):
    return next(n for n, t in SIZES if area < t)


def analyse(model, images, names):
    keys = list(names.values()) + [n for n, _ in SIZES] + ["all"]
    c = {k: dict(gt=0, TP=0, FN_missed=0, FN_class=0, pred=0, FP_class=0, FP_bg=0) for k in keys}
    results = (r for i in range(0, len(images), 8)   # chunks of 8: a long list is otherwise stacked into one batch
               for r in model.predict(images[i:i + 8], conf=0.25, iou=0.7, half=False, imgsz=640, verbose=False))
    for r in results:
        h, w = r.orig_shape
        lb = Path(r.path).parents[1] / "labels" / (Path(r.path).stem + ".txt")
        rows = [list(map(float, l.split())) for l in lb.read_text().splitlines() if l.strip()] if lb.exists() else []
        gcls = torch.tensor([int(r_[0]) for r_ in rows], dtype=torch.long)
        gbox = xywhn2xyxy(torch.tensor([r_[1:5] for r_ in rows]).reshape(-1, 4), w, h)
        gsize = [size_of(float((b[2] - b[0]) * (b[3] - b[1]))) for b in gbox]
        pbox, pcls, pconf = r.boxes.xyxy.cpu(), r.boxes.cls.cpu().long(), r.boxes.conf.cpu()
        order = pconf.argsort(descending=True)
        pbox, pcls = pbox[order], pcls[order]
        iou = box_iou(pbox, gbox) if len(pbox) and len(gbox) else torch.zeros(len(pbox), len(gbox))
        matched = torch.zeros(len(gbox), dtype=torch.bool)
        wrong_cov = torch.zeros(len(gbox), dtype=torch.bool)
        for i in range(len(pbox)):
            pc = names[int(pcls[i])]
            for k in (pc, "all"):
                c[k]["pred"] += 1
            same = (gcls == pcls[i]) & ~matched & (iou[i] >= 0.5) if len(gbox) else torch.zeros(0, dtype=torch.bool)
            if same.any():
                j = int(torch.where(same, iou[i], torch.full_like(iou[i], -1)).argmax())
                matched[j] = True
            elif len(gbox) and (iou[i] >= 0.5).any():
                wrong_cov |= (iou[i] >= 0.5) & (gcls != pcls[i])
                for k in (pc, "all"):
                    c[k]["FP_class"] += 1
            else:
                for k in (pc, "all"):
                    c[k]["FP_bg"] += 1
        for j in range(len(gbox)):
            for k in (names[int(gcls[j])], gsize[j], "all"):
                c[k]["gt"] += 1
                if matched[j]:
                    c[k]["TP"] += 1
                elif wrong_cov[j]:
                    c[k]["FN_class"] += 1
                else:
                    c[k]["FN_missed"] += 1
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--tag", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    a = p.parse_args()
    data = check_det_dataset(a.data)
    out_rows = []
    for name in a.names:
        model = YOLO(str(ROOT / "runs" / name / "weights" / "best.pt"))
        for split in a.splits:
            images = sorted(str(f) for f in Path(data[split]).glob("*.jpg"))
            c = analyse(model, images, model.names)
            for k, v in c.items():
                out_rows.append(dict(model=name, split=split, group=k, **v,
                                     recall=round(v["TP"] / v["gt"], 4) if v["gt"] else None))
            al = c["all"]
            print(f"{name} {split}: GT {al['gt']} TP {al['TP']} FN missed {al['FN_missed']} class {al['FN_class']} | "
                  f"pred {al['pred']} FP bg {al['FP_bg']} class {al['FP_class']}", flush=True)
    out = ROOT / "experiments" / "results" / "E5" / f"e5_{a.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    print("->", out)


if __name__ == "__main__":
    main()
