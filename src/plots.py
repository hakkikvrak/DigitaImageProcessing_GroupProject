"""Rapor için grafikler ve karşılaştırma görselleri."""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enhance import VARIANT_NAMES

# Sabit sıralı kategorik palet (sürüm -> renk her grafikte aynı)
VAR_COLORS = {"raw": "#2a78d6", "bc": "#eb6834", "he": "#1baf7a", "clahe": "#eda100"}
DET_NAMES = {"haar": "Haar", "hog": "HOG", "yolo": "YOLO-face"}
COND_NAMES = {"ALL": "Tümü", "normal": "Normal ışık", "low_light": "Düşük ışık", "challenge": "Zorlayıcı"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _grouped(ax, summary, cond, metric, dets, variants, ymax=None, fmt=None):
    g = summary[summary.condition == cond]
    n = len(variants)
    w = 0.8 / n
    x = np.arange(len(dets))
    for i, v in enumerate(variants):
        vals = [g[(g.detector == d) & (g.variant == v)][metric].mean() for d in dets]
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, w * 0.92, color=VAR_COLORS[v],
                      label=VARIANT_NAMES[v], edgecolor="white", linewidth=1)
        for b, val in zip(bars, vals):
            if fmt and not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(val),
                        ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x, [DET_NAMES[d] for d in dets])
    ax.set_title(COND_NAMES.get(cond, cond), color=INK)
    if ymax:
        ax.set_ylim(0, ymax)


def _small_multiples(summary, metric, ylabel, path, dets, variants, ymax=None, fmt=None):
    conds = ["ALL"] + [c for c in ["normal", "low_light", "challenge"] if c in set(summary.condition)]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    if ymax is None:
        ymax = summary[metric].max() * 1.18 + 1e-9
    for ax, c in zip(axes, conds):
        _grouped(ax, summary, c, metric, dets, variants, ymax, fmt)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(variants), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.savefig(path)
    plt.close(fig)


def all_figures(summary, stats, rq, outdir, dets, variants):
    _small_multiples(summary, "F1", "F1 skoru (IoU ≥ 0.50)", outdir / "f1_by_condition.png", dets, variants, 1.1)
    _small_multiples(summary, "recall", "Recall", outdir / "recall_by_condition.png", dets, variants, 1.1)
    _small_multiples(summary, "precision", "Precision", outdir / "precision_by_condition.png", dets, variants, 1.1)
    _small_multiples(summary, "MAE", "MAE (yüz sayma hatası)", outdir / "mae_by_condition.png", dets, variants)
    _small_multiples(summary[summary.condition == "ALL"], "e2e_ms", "Uçtan uca süre (ms/görüntü)",
                     outdir / "time_e2e.png", dets, variants, fmt="{:.0f}")

    # Görüntü istatistikleri: her koşulda sürümlerin ortalama parlaklık / kontrast değerleri
    st = stats.groupby(["condition", "variant"])[["mean_brightness", "rms_contrast"]].mean().reset_index()
    st["detector"] = "x"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, lab in [(axes[0], "mean_brightness", "Ortalama parlaklık (0-255)"),
                         (axes[1], "rms_contrast", "RMS kontrast (std)")]:
        conds = [c for c in ["normal", "low_light", "challenge"] if c in set(st.condition)]
        x = np.arange(len(conds)); w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            vals = [st[(st.condition == c) & (st.variant == v)][col].mean() for c in conds]
            ax.bar(x - 0.4 + w * (i + 0.5), vals, w * 0.92, color=VAR_COLORS[v], label=VARIANT_NAMES[v],
                   edgecolor="white", linewidth=1)
        ax.set_xticks(x, [COND_NAMES[c] for c in conds]); ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(variants), frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.savefig(outdir / "image_quality_stats.png"); plt.close(fig)

    # Araştırma sorusu: kontrast artışı (x) ile F1 değişimi (y)
    if not rq.empty:
        markers = {"normal": "o", "low_light": "s", "challenge": "^"}
        fig, axes = plt.subplots(1, len(dets), figsize=(4 * len(dets), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, d in zip(axes, dets):
            g = rq[rq.detector == d]
            ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
            for _, r in g.iterrows():
                ax.scatter(r.d_contrast, r.d_F1, s=70, color=VAR_COLORS[r.variant], marker=markers[r.condition],
                           edgecolor="white", linewidth=1.5, zorder=3)
            ax.set_title(DET_NAMES[d]); ax.set_xlabel("Δ RMS kontrast (ham'a göre)")
            ax.grid(True, axis="both")
            ax.text(0.98, 0.03, "kontrast ↑ ama F1 ↓", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
        axes[0].set_ylabel("Δ F1 (ham'a göre)")
        from matplotlib.lines import Line2D
        hv = [Line2D([], [], marker="o", ls="", color=VAR_COLORS[v], label=VARIANT_NAMES[v]) for v in variants if v != "raw"]
        hc = [Line2D([], [], marker=markers[c], ls="", color=MUTED, label=COND_NAMES[c]) for c in markers]
        fig.legend(handles=hv + hc, loc="upper center", ncol=len(hv + hc), frameon=False, bbox_to_anchor=(0.5, 1.08))
        fig.savefig(outdir / "research_contrast_vs_f1.png"); plt.close(fig)


# ---------------- Karşılaştırma görseli ----------------
C_GT, C_TP, C_FP = (0, 200, 0), (255, 140, 0), (0, 0, 230)   # BGR: yeşil, mavi, kırmızı


def draw(img, gt, preds, m, scale):
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    t = max(2, round(3 * scale * 2))
    S = lambda b: tuple(int(round(v * scale)) for v in b[:4])
    if gt is not None:
        for b in gt:
            x1, y1, x2, y2 = S(b); cv2.rectangle(out, (x1, y1), (x2, y2), C_GT, t)
    fp = set(m["_fp"]) if m else set()
    for i, b in enumerate(preds):
        x1, y1, x2, y2 = S(b)
        cv2.rectangle(out, (x1, y1), (x2, y2), C_FP if i in fp else C_TP, t)
    return out


def comparison_grid(name, gt, enhanced, store, dets, variants, path, cell_w=420):
    img = make_grid(name, gt, enhanced, store, dets, variants, cell_w)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def make_grid(name, gt, enhanced, store, dets, variants, cell_w=420):
    """Dedektör × sürüm karşılaştırma ızgarasını BGR dizi olarak döndürür."""
    rows = []
    for d in dets:
        cells = []
        for v in variants:
            img = enhanced[(name, v)][0]
            preds, m = store[(name, d, v)]
            sc = cell_w / img.shape[1]
            c = draw(img, gt, preds, m, sc)
            txt = f"{DET_NAMES[d]} | {v}"
            txt += f" | TP{m['tp']} FP{m['fp']} FN{m['fn']}" if m else f" | n={len(preds)}"
            cv2.rectangle(c, (0, 0), (c.shape[1], 26), (255, 255, 255), -1)
            cv2.putText(c, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            cells.append(cv2.copyMakeBorder(c, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    legend = np.full((30, grid.shape[1], 3), 255, np.uint8)
    for i, (lab, col) in enumerate([("GT (referans)", C_GT), ("Dogru tespit (TP)", C_TP), ("Yanlis tespit (FP)", C_FP)]):
        x = 10 + i * 220
        cv2.rectangle(legend, (x, 8), (x + 22, 22), col, -1)
        cv2.putText(legend, lab, (x + 30, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([legend, grid])
