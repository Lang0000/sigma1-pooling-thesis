# σ₁-пулінг у BiFPN для детекції об'єктів на теплових знімках з БПЛА

Програмна частина магістерської роботи: детектор YOLOv8-EPB-Lite (бекбон EfficientNet-B0 + BiFPN), у якому
пулінг вікна 3×3 у BiFPN замінено на найбільше сингулярне число σ₁ матриці вікна. Тут є код, конфігурації моделей,
скрипти експериментів, набір даних, ваги навчених моделей і результати, на які спирається текст роботи.

Основа — форк [Ultralytics](https://github.com/ultralytics/ultralytics) **8.3.204**. Власний код займає два модулі
й один рядок імпорту, решта бібліотеки не змінювалась.

## Структура

```
ultralytics/                      бібліотека Ultralytics 8.3.204
├── nn/modules/custom_pool.py     WindowPool 3×3/s2/p1: max | sigma1_svd | sigma1_svd_fp32 | sigma1_fast | maxabs | frobenius
├── nn/modules/custom.py          EfficientNetBackbone + PoolBiFPN (вид пулінгу задається в YAML)
└── nn/tasks.py                   +1 рядок: імпорт власних модулів для парсера YAML
data/huv-termal.v2i.yolov8/       набір даних HIT-UAV (Roboflow dersi/huv-termal v2), формат YOLO
experiments/
├── configs/                      YAML моделей: lite, lite-svd, lite-svd-f, lite-maxabs, lite-frobenius
├── train.py, val.py              навчання і валідація (гіперпараметри як у статті)
├── check_build.py, test_pool.py  перевірка середовища, збірки моделей і σ₁/градієнтів
├── e0_…e5_*.py, gate_sigma1.py   експерименти E0–E6 і ворота G1–G3
├── bench_step.py                 швидкість навчального кроку
├── r6_split.py                   розбиття за відеопослідовностями (без витоку сусідніх кадрів)
├── original/                     оригінальний код співавтора і скрипт порівняння з відновленим
├── windows/                      setup.ps1 і run_min_set.ps1 для Windows
├── weights/<ID>/{best,last}.pt   ваги навчених моделей (Git LFS)
└── results/<ID>/                 метрики, таблиці й графіки кожного прогону та експерименту
```

## Швидкий старт

Потрібні Python 3.10+, NVIDIA GPU з CUDA 12.8 і Git LFS.

```bash
git clone https://github.com/Lang0000/sigma1-pooling-thesis.git
cd sigma1-pooling-thesis
git lfs install && git lfs pull                       # ваги моделей
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e . timm
python experiments/check_build.py                     # CUDA, 3 моделі по 4 556 695 параметрів, stride 8/16/32
cd experiments && python test_pool.py                 # σ₁ fast проти SVD, градієнти
```

На Windows те саме робить `experiments/windows/setup.ps1`: створює conda-середовище `thesis`, ставить PyTorch
і пакет, проганяє перевірки. Варто ввімкнути довгі шляхи (`git config --global core.longpaths true`).

### Детекція на нових кадрах

```python
from ultralytics import YOLO

model = YOLO("experiments/weights/F2_Lite-SVD-F_AMP/best.pt")
results = model.predict("frames/", conf=0.25, imgsz=640, save=True)
for r in results:
    print(r.path, r.boxes.cls.tolist(), r.boxes.conf.tolist())
```

### Навчання і валідація

```bash
python experiments/train.py --model lite       --args svd        --name R5_Lite-matched     # MaxPool
python experiments/train.py --model lite-svd-f --args svd --amp  --name F2_Lite-SVD-F_AMP   # σ₁ аналітично
python experiments/val.py --name F2_Lite-SVD-F_AMP                                          # → results/<ID>/metrics.csv
```

Навчання пише ваги в `runs/<ID>/weights/` (поза git). Скрипти валідації та експериментів беруть ваги звідти ж,
тому для збережених моделей їх треба скопіювати: `cp -r experiments/weights/<ID> runs/<ID>/weights`.
Додаткові параметри: `--seed`, `--data <data.yaml>` (інший набір), `--resume`.

## Моделі

`--args lite` / `svd` — набір гіперпараметрів Lite або Lite-SVD зі статті. Метрика — mAP50-95 на valid / test.

| ID | Модель | mAP50-95 valid / test |
|---|---|---|
| A1_Lite | Lite (MaxPool), args Lite | 0,548 / 0,500 |
| R5_Lite-matched (+ `_s1`, `_s2`) | Lite (MaxPool), args Lite-SVD, seed 0 / 1 / 2 | 0,543 / 0,508 · 0,557 / 0,524 · 0,547 / 0,515 |
| A2_Lite-SVD | σ₁ через повний SVD, FP32 (як у статті) | 0,552 / 0,524 |
| F1_Lite-SVD-F_FP32 | σ₁ аналітично (метод Якобі), FP32 | 0,557 / 0,515 |
| F2_Lite-SVD-F_AMP (+ `_s1`, `_s2`) | σ₁ аналітично, AMP, seed 0 / 1 / 2 | 0,557 / 0,523 · 0,557 / 0,522 · 0,551 / 0,515 |
| R9_Lite-MaxAbs | контроль: пулінг max\|a\| | 0,551 / 0,535 |
| R9_Lite-Frobenius | контроль: пулінг ‖A‖_F (L2) | 0,552 / 0,498 |
| R6_Lite-matched (+ `_s1`, `_s2`) | MaxPool на розбитті за послідовностями | 0,347 / 0,313 · 0,358 / 0,369 · 0,349 / 0,348 |
| R6_Lite-SVD-F_AMP (+ `_s1`, `_s2`) | σ₁ на розбитті за послідовностями | 0,353 / 0,391 · 0,347 / 0,326 · 0,368 / 0,366 |
| G4_lite_amp, G4_lite-svd-f_amp | 2-епохові прогони для воріт G4 | — |

У `results/<ID>/` для кожної моделі лежать `args.yaml`, `results.csv` (по епохах), `metrics.csv` (valid/test, FP32)
і матриці помилок.

## Експерименти

| ID | Що перевіряє | Скрипт | Результати |
|---|---|---|---|
| E0 | σ₁ на синтетичних вікнах, межі max\|a\| ≤ σ₁ ≤ ‖A‖_F | `e0_synthetic.py` | `results/E0/` |
| E1, E6 | спектри реальних вікон; контраст ціль/фон після пулінгу | `e1_spectra.py`, `e1b_sigma1_frobenius.py` | `results/E1/`, `results/E6/` |
| E2 | градієнти σ₁-пулінгу в режимах FP32 / AMP | `e2_grads.py` | `results/E2/` |
| E3 | латентність, FPS, частка пулінгу, FLOPs, VRAM, енергія | `e3_efficiency.py`, `bench_step.py` | `results/E3/` |
| E4 | парний bootstrap різниці метрик двох моделей | `e4_bootstrap.py` | `results/E4/` |
| E5 | структура помилок за класами й розмірами об'єктів | `e5_errors.py` | `results/E5/` |
| G1–G3 | збіг σ₁ fast з SVD на реальних активаціях | `gate_sigma1.py`, `val.py --kind sigma1_svd` | `results/F0_gate/` |
| R6 | розбиття за відеопослідовностями | `r6_split.py` | `results/R6_split/` |

Версії середовища, на якому отримано результати (ПК з RTX 5070 Ti і Google Colab), записано в `results/00_env/`.

## Набір даних

[HIT-UAV](https://universe.roboflow.com/dersi/huv-termal/dataset/2) у реекспорті Roboflow `dersi/huv-termal` v2
(ліцензія CC BY 4.0): 2866 теплових кадрів, класи Bicycle, Car, DontCare, OtherVehicle, Person. У `data.yaml` немає
`path:`, тому шляхи до частин набору рахуються від самого файлу і працюють на будь-якій машині.

## Ліцензія

Код Ultralytics і похідний код поширюються за ліцензією AGPL-3.0 (див. `LICENSE`). Набір даних — CC BY 4.0.
