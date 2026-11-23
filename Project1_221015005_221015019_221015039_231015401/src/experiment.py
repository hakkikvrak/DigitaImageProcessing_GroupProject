"""Runs the full experiment: 2 detectors x 4 image versions x 28 images.

Outputs (results/ folder):
  per_image.csv               TP, FP, FN, counting error, time per image / detector / method
  summary.csv                 detector x method: TP, FP, FN, Precision, Recall, F1, MAE, time
  condition_summary.csv       detector x method x condition (normal / low light / overexposed)
  image_quality.csv           each method's effect on the image (mean brightness, contrast, entropy)
  parameters.json             fixed parameters and versions
  charts/*.png                charts for the report
  samples/*.jpg               sample images with the boxes drawn

Run:  python src/experiment.py
"""
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors as D             # noqa: E402
import enhancement as I           # noqa: E402
import metrics as M                # noqa: E402
import dataset as V                # noqa: E402

ROOT = V.ROOT
RESULTS_DIR = os.path.join(ROOT, "results")
METHODS = list(I.METHODS)                         # ham, parlaklik_kontrast, ...
CONDITIONS = list(V.CONDITION_NAMES)                       # normal, dusuk_isik, asiri_parlak

# Colors: detector identity (categorical, fixed order)
COLORS = {"Haar Cascade": "#2a78d6", "YOLOv8n-face": "#eb6834"}
GT_COLOR_BGR = (0, 200, 0)                             # ground-truth boxes in green
# English tags used only in sample image file names (method keys stay as in the CSV files)
FILE_TAGS = {"ham": "raw", "parlaklik_kontrast": "brightness_contrast", "histogram_esitleme": "hist_eq", "clahe": "clahe"}


def draw_boxes(img, truth, pred, matches):
    """Green = ground truth, blue = correct detection (TP), red = false detection (FP)."""
    out = img.copy()
    thickness = max(2, img.shape[1] // 400)
    for g in truth:
        cv2.rectangle(out, g[:2], g[2:4], GT_COLOR_BGR, thickness)
    tp_idx = {i for i, _, _ in matches}
    for i, t in enumerate(pred):
        color = (255, 120, 0) if i in tp_idx else (0, 0, 255)
        cv2.rectangle(out, t[:2], t[2:4], color, thickness)
    return out


def entropy(gray):
    h = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def verify_histogram():
    """Compares our own histogram equalization with cv2.equalizeHist (verification only)."""
    record = V.read_conditions()[0]
    gray = cv2.cvtColor(V.read_image(os.path.join(V.IMAGE_DIR, record["dosya"])),
                       cv2.COLOR_BGR2GRAY)
    diff = np.abs(I.equalize_hist_gray(gray).astype(int) - cv2.equalizeHist(gray).astype(int))
    return int(diff.max()), float(diff.mean())


def run(progress=None, haar_param=None, yolo_param=None):
    """Runs the experiment. progress(text, ratio) is used for a progress bar."""
    os.makedirs(os.path.join(RESULTS_DIR, "charts"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "samples"), exist_ok=True)

    labels = V.read_labels()
    records = V.read_conditions()
    detectors = [D.HaarDetector(**(haar_param or {})), D.YoloDetector(**(yolo_param or {}))]

    # YOLO warm-up (the first call includes model loading; keep it out of the timing)
    detectors[1].detect(np.zeros((640, 640, 3), np.uint8))

    rows, quality = [], []
    total_tasks = len(records) * len(METHODS)
    task_no = 0
    for k in records:
        raw = V.read_image(os.path.join(V.IMAGE_DIR, k["dosya"]))
        truth = labels[k["dosya"]]
        for y in METHODS:
            task_no += 1
            if progress:
                progress(f"Image {k['no']} / {I.METHODS[y][0]}", task_no / total_tasks)
            t0 = time.perf_counter()
            img = I.apply_method(raw, y)
            enhance_ms = (time.perf_counter() - t0) * 1000

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            quality.append(dict(no=k["no"], kosul=k["kosul"], yontem=y,
                               parlaklik=float(gray.mean()), kontrast=float(gray.std()),
                               entropi=entropy(gray)))

            for d in detectors:
                t1 = time.perf_counter()
                pred = d.detect(img)
                infer_ms = (time.perf_counter() - t1) * 1000
                matches, tp, fp, fn = M.match(pred, truth)
                rows.append(dict(
                    no=int(k["no"]), dosya=k["dosya"], kosul=k["kosul"], yontem=y, dedektor=d.name,
                    gercek_sayi=len(truth), tahmin_sayi=len(pred), tp=tp, fp=fp, fn=fn,
                    sayma_hatasi=abs(len(pred) - len(truth)),
                    cikarim_ms=round(infer_ms, 2),
                    uctan_uca_ms=round(enhance_ms + infer_ms, 2)))
                # Save some sample images (one from each condition)
                if int(k["no"]) in (4, 5, 6, 18, 25):
                    name = f"{int(k['no']):02d}_{FILE_TAGS[y]}_{d.name.split()[0].split('v')[0]}.jpg"
                    V.write_image(os.path.join(RESULTS_DIR, "samples", name),
                                  cv2.resize(draw_boxes(img, truth, pred, matches), None, fx=0.5, fy=0.5))

    write_csv(os.path.join(RESULTS_DIR, "per_image.csv"), rows)
    write_csv(os.path.join(RESULTS_DIR, "image_quality.csv"), quality)
    summary = summarize(rows, ["dedektor", "yontem"])
    condition_summary = summarize(rows, ["dedektor", "yontem", "kosul"])
    write_csv(os.path.join(RESULTS_DIR, "summary.csv"), summary)
    write_csv(os.path.join(RESULTS_DIR, "condition_summary.csv"), condition_summary)
    quality_summary = summarize_quality(quality)
    write_csv(os.path.join(RESULTS_DIR, "image_quality_summary.csv"), quality_summary)

    mx, mean_diff = verify_histogram()
    param = dict(
        haar=dict(detectors[0].param, model="haarcascade_frontalface_default.xml"),
        yolo=dict(detectors[1].param, model="yolov8n-face.pt (akanametov/yolo-face 1.0.0)", device="cpu"),
        iyilestirme=dict(parlaklik_kontrast=dict(alfa=I.BC_ALPHA, beta=I.BC_BETA),
                         histogram_esitleme="elle yazıldı, YCrCb Y kanalı",
                         clahe=dict(clipLimit=I.CLAHE_CLIP, tileGridSize=I.CLAHE_GRID, kanal="LAB L")),
        iou_esigi=M.IOU_THRESHOLD, eslestirme="bire bir, en yüksek IoU öncelikli (greedy)",
        sure_olcumu="cikarim_ms = sadece dedektör çağrısı; uctan_uca_ms = iyileştirme + dedektör "
                    "(dosya okuma hariç). YOLO bir kez ısındırıldıktan sonra ölçüldü.",
        kendi_HE_vs_cv2_equalizeHist=dict(max_fark=mx, ort_fark=round(mean_diff, 4)),
        surumler=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                      ultralytics=__import__("ultralytics").__version__,
                      torch=__import__("torch").__version__, platform=platform.platform()),
    )
    with open(os.path.join(RESULTS_DIR, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(param, f, ensure_ascii=False, indent=2)

    plot_charts(summary, condition_summary, rows, quality_summary)
    if progress:
        progress("Done", 1.0)
    return summary, condition_summary


def summarize(rows, keys):
    groups = defaultdict(list)
    for s in rows:
        groups[tuple(s[a] for a in keys)].append(s)
    order = {"dedektor": list(COLORS), "yontem": METHODS, "kosul": CONDITIONS}
    output = []
    for group_key in sorted(groups, key=lambda t: [order[a].index(v) for a, v in zip(keys, t)]):
        g = groups[group_key]
        tp, fp, fn = (sum(s[x] for s in g) for x in ("tp", "fp", "fn"))
        p, r, f1 = M.summary(tp, fp, fn)
        output.append({**dict(zip(keys, group_key)),
                      "goruntu": len(g), "gercek_yuz": sum(s["gercek_sayi"] for s in g),
                      "tahmin": sum(s["tahmin_sayi"] for s in g),
                      "TP": tp, "FP": fp, "FN": fn,
                      "precision": round(p, 3), "recall": round(r, 3), "F1": round(f1, 3),
                      "MAE": round(np.mean([s["sayma_hatasi"] for s in g]), 3),
                      "ort_cikarim_ms": round(np.mean([s["cikarim_ms"] for s in g]), 1),
                      "ort_uctan_uca_ms": round(np.mean([s["uctan_uca_ms"] for s in g]), 1)})
    return output


def summarize_quality(quality):
    output = []
    for condition in CONDITIONS + ["tumu"]:
        for y in METHODS:
            g = [q for q in quality if q["yontem"] == y and (condition == "tumu" or q["kosul"] == condition)]
            output.append(dict(kosul=condition, yontem=y,
                              ort_parlaklik=round(np.mean([q["parlaklik"] for q in g]), 1),
                              ort_kontrast=round(np.mean([q["kontrast"] for q in g]), 1),
                              ort_entropi=round(np.mean([q["entropi"] for q in g]), 3)))
    return output


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ charts
def _clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors="#444444")
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, values_dict, labels, fmt="{:.2f}"):
    n = len(values_dict)
    bar_w = 0.8 / n
    x = np.arange(len(labels))
    for i, (name, values) in enumerate(values_dict.items()):
        pos = x - 0.4 + bar_w * (i + 0.5)
        bars = ax.bar(pos, values, bar_w * 0.92, color=COLORS[name], label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _clean_axes(ax)


def plot_charts(summary, condition_summary, rows, quality_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(RESULTS_DIR, "charts")
    method_labels = [I.METHODS[y][0] for y in METHODS]

    def get_values(table, field, **filters):
        return {d: [next(s[field] for s in table if s["dedektor"] == d and s["yontem"] == y
                         and all(s[k] == v for k, v in filters.items())) for y in METHODS]
                for d in COLORS}

    # 1) F1 and MAE - full dataset
    for field, title_text, filename, fmt in [
            ("F1", "F1 score (IoU ≥ 0.50) – full dataset", "f1_overall.png", "{:.2f}"),
            ("MAE", "Mean absolute counting error (MAE) – lower is better", "mae_overall.png", "{:.2f}"),
            ("ort_cikarim_ms", "Mean inference time (ms / image)", "inference_time.png", "{:.0f}")]:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        _bars(ax, get_values(summary, field), method_labels, fmt)
        if field == "F1":
            ax.set_ylim(0, 1.05)
        ax.set_title(title_text, loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)

    # 2) F1 by condition (small multiples)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, condition in zip(axs, CONDITIONS):
        _bars(ax, get_values(condition_summary, "F1", kosul=condition), ["Raw", "B/C", "HE", "CLAHE"])
        ax.set_title(V.CONDITION_NAMES[condition], fontsize=11)
        ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("F1")
    axs[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("F1 score by capture condition", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_by_condition.png"))
    plt.close(fig)

    # 3) Recall by condition
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, condition in zip(axs, CONDITIONS):
        _bars(ax, get_values(condition_summary, "recall", kosul=condition), ["Raw", "B/C", "HE", "CLAHE"])
        ax.set_title(V.CONDITION_NAMES[condition], fontsize=11)
        ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("Recall")
    axs[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Recall by capture condition (share of faces found)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recall_by_condition.png"))
    plt.close(fig)

    # 4) Research question: does enhancement increase F1 compared with the raw image?
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, condition in zip(axs, CONDITIONS):
        f1 = get_values(condition_summary, "F1", kosul=condition)
        diff = {d: [v - f1[d][0] for v in f1[d][1:]] for d in COLORS}   # difference from raw
        _bars(ax, diff, ["B/C", "HE", "CLAHE"], "{:+.2f}")
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_title(V.CONDITION_NAMES[condition], fontsize=11)
    axs[0].set_ylabel("F1 difference (enhanced − raw)")
    axs[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("How much did enhancement change F1 vs. the raw image? (below 0 = worse)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_change.png"))
    plt.close(fig)


if __name__ == "__main__":
    def show_progress(msg, ratio):
        print(f"\r[{ratio * 100:5.1f}%] {msg:<50}", end="", flush=True)

    summary, _ = run(show_progress)
    print("\n\nSUMMARY (full dataset)")
    print(f"{'Detector':<14}{'Method':<22}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>7}{'R':>7}{'F1':>7}{'MAE':>7}{'ms':>8}")
    for s in summary:
        print(f"{s['dedektor']:<14}{I.METHODS[s['yontem']][0]:<22}{s['TP']:>4}{s['FP']:>4}{s['FN']:>4}"
              f"{s['precision']:>7.3f}{s['recall']:>7.3f}{s['F1']:>7.3f}{s['MAE']:>7.2f}{s['ort_cikarim_ms']:>8.1f}")
    print("\nResults:", RESULTS_DIR)
