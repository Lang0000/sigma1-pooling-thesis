"""R6: re-split HIT-UAV by video sequence (no neighbouring frames across train/val/test), ~70/20/10 of images.

    python experiments/r6_split.py            # -> data/huv-termal-r6/ (hard links, outside git) + results/R6_split/

File name: <f0>_<altitude>_<angle>_<f3>_<frame>_jpg.rf.<hash>.jpg; sequence = (f0, altitude, angle, f3).
Sequences are shuffled and assigned greedily to the split furthest below its target share of images; out of N_SEEDS
shuffles the one with the smallest max deviation of image and per-class object shares from 70/20/10 is kept.
The split lists go to experiments/results/R6_split/split.csv (in git); images/labels are hard links to the original.
"""
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "huv-termal.v2i.yolov8"
DST = ROOT / "data" / "huv-termal-r6"
OUT = ROOT / "experiments" / "results" / "R6_split"
TARGET = {"train": 0.7, "valid": 0.2, "test": 0.1}
NAMES = ["Bicycle", "Car", "DontCare", "OtherVehicle", "Person"]
N_SEEDS = 20000


def main():
    seqs = defaultdict(list)  # sequence -> [(orig split, image path)]
    for split in ("train", "valid", "test"):
        for im in sorted((SRC / split / "images").glob("*.jpg")):
            seqs[tuple(im.name.split("_")[:4])].append((split, im))
    n = sum(len(v) for v in seqs.values())
    # per sequence: [images, objects of each class]
    vec = {}
    for k, items in seqs.items():
        v = [len(items)] + [0] * len(NAMES)
        for orig, im in items:
            lb = SRC / orig / "labels" / (im.stem + ".txt")
            if lb.exists():
                for l in lb.read_text().splitlines():
                    if l.strip():
                        v[1 + int(l.split()[0])] += 1
        vec[k] = v
    tot = [sum(v[i] for v in vec.values()) for i in range(1 + len(NAMES))]

    def greedy(seed):  # shuffled sequences -> split furthest below its target share of images
        keys = sorted(seqs)
        random.Random(seed).shuffle(keys)
        acc = {s: [0] * len(tot) for s in TARGET}
        assign = {}
        for k in keys:
            s = min(TARGET, key=lambda s: acc[s][0] / n - TARGET[s])
            assign[k] = s
            acc[s] = [a + b for a, b in zip(acc[s], vec[k])]
        dev = max(abs(acc[s][i] / tot[i] - TARGET[s]) for s in TARGET for i in range(len(tot)))
        return dev, seed, assign, acc

    # stratification: the seed whose split has the smallest max deviation of image AND per-class shares from 70/20/10
    dev, seed, assign, acc = min(greedy(s) for s in range(N_SEEDS))
    count = {s: acc[s][0] for s in TARGET}
    print(f"best seed {seed} of {N_SEEDS}: max share deviation {dev:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, cls = [], {s: Counter() for s in TARGET}
    for k, items in seqs.items():
        s = assign[k]
        for orig, im in items:
            lb = SRC / orig / "labels" / (im.stem + ".txt")
            for d, f in (("images", im), ("labels", lb)):
                t = DST / s / d / f.name
                t.parent.mkdir(parents=True, exist_ok=True)
                if f.exists() and not t.exists():
                    os.link(f, t)
            if lb.exists():
                cls[s].update(int(l.split()[0]) for l in lb.read_text().splitlines() if l.strip())
            rows.append(dict(image=im.name, sequence="_".join(k), orig_split=orig, r6_split=s))
    rows.sort(key=lambda r: (r["r6_split"], r["image"]))
    with open(OUT / "split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (DST / "data.yaml").write_text(
        "# R6 split by video sequence (experiments/r6_split.py); paths relative to this file\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\nnc: 5\n"
        f"names: {NAMES}\n")

    seq_n = Counter(assign.values())
    lines = ["| split | images | share | sequences | " + " | ".join(NAMES) + " |", "|---" * (4 + len(NAMES)) + "|"]
    for s in TARGET:
        lines.append(f"| {s} | {count[s]} | {count[s] / n:.1%} | {seq_n[s]} | "
                     + " | ".join(str(cls[s][i]) for i in range(len(NAMES))) + " |")
    head = f"{n} images, {len(seqs)} sequences; split seed {seed} of {N_SEEDS}, max share deviation {dev:.3f}"
    print(head)
    print("\n".join(lines))
    (OUT / "split_stats.md").write_text(head + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
