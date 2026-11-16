"""Projenin akış şemasını üretir -> report/akis_semasi.png   (python src/make_flowchart.py)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import ROOT

INK, MUTED = "#0b0b0b", "#52514e"
FILL = {"data": "#cde2fb", "proc": "#f3f2ee", "det": "#fde0d3", "eval": "#cdf0e2", "out": "#fdecc4"}


def box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[kind], ec=MUTED, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))


def main():
    fig, ax = plt.subplots(figsize=(10, 12.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")
    cx = 5
    box(ax, cx, 12.3, 6.4, 0.8, "1. Görüntü toplama (onaylı, ekip üyeleri)\nnormal ışık | düşük ışık | zorlayıcı koşul (≥5'er)", "data")
    box(ax, cx, 11.0, 6.4, 0.8, "Ön işleme (preprocess.py)\nEXIF yön düzeltme → uzun kenar 1280 px → EXIF silme", "proc")
    box(ax, cx, 9.7, 6.4, 0.8, "Etiketleme (annotate.py)\nher yüz için GT kutusu (YOLO txt) + gerçek yüz sayısı", "data")
    box(ax, cx, 8.4, 6.4, 0.8, "3. Görüntü iyileştirme (enhance.py) — 4 sürüm\nHam | Parlaklık/Kontrast | Histogram Eşitleme (elle) | CLAHE", "proc")
    xs = [1.9, 5, 8.1]
    labels = ["Haar Cascade\n(Viola–Jones)\nscale 1.1, minN 5", "Dlib HOG + SVM\n(ek klasik)\nupsample 1",
              "YOLOv11n-face\nconf 0.25, NMS 0.45\nimgsz 1280"]
    for x, l in zip(xs, labels):
        box(ax, x, 6.8, 2.7, 1.1, l, "det")
        arrow(ax, cx, 8.0, x, 7.37)
    ax.text(cx, 7.75, "2/4. Kontrollü karşılaştırma: aynı görüntüler, sabit dedektör ayarları", ha="center",
            fontsize=8, color=MUTED, bbox=dict(fc="white", ec="none", pad=1))
    box(ax, cx, 5.2, 6.4, 0.8, "Tahmin kutuları + yüz sayısı + süre (infer_ms, e2e_ms)", "proc")
    for x in xs:
        arrow(ax, x, 6.24, cx, 5.62)
    box(ax, cx, 3.95, 6.4, 0.9, "5. Değerlendirme (evaluate.py)\nIoU ≥ 0.50 bire bir eşleştirme (Macar alg.) → TP, FP, FN\nPrecision, Recall, F1 | |sayma hatası|, MAE", "eval")
    box(ax, cx, 2.6, 6.4, 0.8, "Analiz: koşul × sürüm × dedektör tabloları\nkontrast/entropi değişimi ↔ F1 değişimi (araştırma sorusu)", "eval")
    box(ax, 2.6, 1.1, 4.0, 0.9, "Çıktılar: CSV tablolar, grafikler,\nkarşılaştırma görselleri", "out")
    box(ax, 7.4, 1.1, 4.0, 0.9, "Word raporu + canlı demo (demo.py)\nyeni görüntü / parametre değişikliği", "out")
    for (y1, y2) in [(11.9, 11.42), (10.6, 10.12), (9.3, 8.82), (4.8, 4.42), (3.5, 3.02)]:
        arrow(ax, cx, y1, cx, y2)
    arrow(ax, 3.8, 2.2, 2.8, 1.57); arrow(ax, 6.2, 2.2, 7.2, 1.57)
    ax.set_title("Zorlu Koşullarda Yüz Tespiti — Proje Akış Şeması", fontsize=13, color=INK, pad=4)
    out = ROOT / "report" / "akis_semasi.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
