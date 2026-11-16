"""
ADIM 2 + 4 + 5 - Tüm deneyleri tek komutla çalıştırır.

  python src/run_experiments.py                       # haar, hog, yolo × raw, bc, he, clahe
  python src/run_experiments.py --detectors haar yolo # yalnızca zorunlu iki dedektör
  python src/run_experiments.py --minNeighbors 3      # (savunma) parametre değişikliği

Kontrollü karşılaştırma: veri seti ve dedektör ayarları tüm sürümlerde AYNI kalır;
yalnızca girdi görüntüsü (raw / bc / he / clahe) değişir.

Süre tanımları (ms, görüntü başına):
  infer_ms : yalnızca dedektör çağrısı (Haar/HOG için gri dönüşüm dahil, YOLO için
             ön işleme + model + NMS dahil). Diskten okuma HARİÇ.
  enh_ms   : yalnızca iyileştirme işlemi süresi
  e2e_ms   : enh_ms + infer_ms  (uçtan uca, diskten okuma hariç)
Ölçümden önce her dedektör TIMING_WARMUP kez ısınma çalıştırması yapar.

Çıktılar results/ altında: per_image.csv, summary.csv, summary_pivot.csv, image_stats.csv,
research_question.csv, settings.json, figures/*.png, vis/*_grid.jpg
"""
import argparse
import json
import platform
import time
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import config as C
import enhance as E
import detectors as D
import evaluate as EV
from dataset import load_dataset, write_gt_counts
import plots


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detectors", nargs="+", default=["haar", "hog", "yolo"], choices=list(D.DETECTORS))
    ap.add_argument("--variants", nargs="+", default=E.VARIANTS, choices=E.VARIANTS)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    # parametre değişiklikleri (savunma için) - verilmezse config.py değerleri
    ap.add_argument("--scaleFactor", type=float)
    ap.add_argument("--minNeighbors", type=int)
    ap.add_argument("--upsample", type=int)
    ap.add_argument("--conf", type=float)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--no-vis", action="store_true", help="karşılaştırma görsellerini üretme")
    return ap.parse_args()


def overrides(a, name):
    if name == "haar":
        return dict(scale_factor=a.scaleFactor, min_neighbors=a.minNeighbors)
    if name == "hog":
        return dict(upsample=a.upsample)
    if name == "yolo":
        return dict(conf=a.conf, imgsz=a.imgsz)
    return {}


def summarize(df):
    rows = []
    ev = df[df.gt_count.notna()]
    for (det, var), g in ev.groupby(["detector", "variant"], sort=False):
        for cond, gg in [("ALL", g)] + list(g.groupby("condition", sort=False)):
            tp, fp, fn = int(gg.tp.sum()), int(gg.fp.sum()), int(gg.fn.sum())
            p, r, f = EV.prf(tp, fp, fn)
            rows.append(dict(detector=det, variant=var, condition=cond, n_images=len(gg),
                             gt_faces=int(gg.gt_count.sum()), pred_faces=int(gg.pred_count.sum()),
                             TP=tp, FP=fp, FN=fn, precision=round(p, 4), recall=round(r, 4), F1=round(f, 4),
                             MAE=round(gg.abs_count_err.mean(), 4),
                             mean_iou_tp=round(gg.mean_iou_tp.mean(), 4),
                             infer_ms=round(gg.infer_ms.mean(), 2), e2e_ms=round(gg.e2e_ms.mean(), 2)))
    return pd.DataFrame(rows)


def research_table(summary, stats):
    """'Görsel olarak daha iyi görünen görüntü her zaman daha iyi tespit sağlar mı?'
    Her koşul × sürüm için: kontrast/entropi değişimi (ham'a göre) ve her dedektörde F1 değişimi."""
    st = stats.groupby(["condition", "variant"])[["mean_brightness", "rms_contrast", "entropy"]].mean()
    rows = []
    for (det, cond), g in summary[summary.condition != "ALL"].groupby(["detector", "condition"]):
        base = g[g.variant == "raw"]
        if base.empty:
            continue
        f1_raw, mae_raw = base.F1.iloc[0], base.MAE.iloc[0]
        s_raw = st.loc[(cond, "raw")]
        for _, r in g[g.variant != "raw"].iterrows():
            s = st.loc[(cond, r.variant)]
            d_con = s.rms_contrast - s_raw.rms_contrast
            rows.append(dict(detector=det, condition=cond, variant=r.variant,
                             d_brightness=round(s.mean_brightness - s_raw.mean_brightness, 2),
                             d_contrast=round(d_con, 2), d_entropy=round(s.entropy - s_raw.entropy, 3),
                             F1_raw=f1_raw, F1=r.F1, d_F1=round(r.F1 - f1_raw, 4),
                             MAE_raw=mae_raw, MAE=r.MAE, d_MAE=round(r.MAE - mae_raw, 4),
                             visually_better_but_worse=bool(d_con > 0 and r.F1 < f1_raw)))
    return pd.DataFrame(rows)


def main():
    a = parse()
    random.seed(C.SEED); np.random.seed(C.SEED)
    out = C.Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "vis").mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    if not data:
        raise SystemExit("data/images altında görüntü yok. Önce: python src/preprocess.py")
    n_lab = sum(gt is not None for *_, gt in data)
    print(f"{len(data)} görüntü, {n_lab} tanesi etiketli.")
    if n_lab < len(data):
        print("UYARI: etiketsiz görüntüler metriklere dahil edilmeyecek (python src/annotate.py)")
    write_gt_counts(data, out / "gt_counts.csv")

    # 1) Tüm sürümleri bir kez üret (iyileştirme süresini de ölç)
    enhanced, stats_rows = {}, []
    for cond, p, img, gt in data:
        for v in a.variants:
            t0 = time.perf_counter()
            e = E.apply(img, v)
            enhanced[(p.name, v)] = (e, (time.perf_counter() - t0) * 1000)
            stats_rows.append(dict(image=p.name, condition=cond, variant=v, **E.image_stats(e)))
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(out / "image_stats.csv", index=False)

    # 2) Dedektörler
    settings = {"date": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(), "opencv": cv2.__version__,
                "platform": platform.platform(), "seed": C.SEED,
                "iou_threshold": C.IOU_THRESHOLD,
                "enhancement": {"bc": {"alpha": C.BC_ALPHA, "beta": C.BC_BETA},
                                "he": "kendi uygulamamız, YCrCb-Y kanalında",
                                "clahe": {"clipLimit": C.CLAHE_CLIP, "tileGridSize": C.CLAHE_TILE,
                                          "channel": "YCrCb-Y"}},
                "timing": "infer_ms = yalnızca dedektör; e2e_ms = iyileştirme + dedektör; disk okuma hariç; "
                          f"{C.TIMING_WARMUP} ısınma çalıştırması",
                "detectors": {}}
    rows, boxes_store = [], {}
    for name in a.detectors:
        det = D.build(name, **overrides(a, name))
        settings["detectors"][name] = det.describe()
        print(f"\n== {det.label}: {det.describe()}")
        for _ in range(C.TIMING_WARMUP):
            det.detect(data[0][2])
        for cond, p, img, gt in data:
            for v in a.variants:
                e, enh_ms = enhanced[(p.name, v)]
                t0 = time.perf_counter()
                preds = det.detect(e)
                infer_ms = (time.perf_counter() - t0) * 1000
                row = dict(image=p.name, condition=cond, detector=name, variant=v,
                           infer_ms=infer_ms, enh_ms=enh_ms, e2e_ms=infer_ms + enh_ms,
                           pred_count=len(preds), boxes=json.dumps([list(b[:4]) + [round(b[4], 3)] for b in preds]))
                if gt is not None:
                    m = EV.image_metrics(preds, gt)
                    boxes_store[(p.name, name, v)] = (preds, m)
                    row.update({k: v_ for k, v_ in m.items() if not k.startswith("_")})
                else:
                    boxes_store[(p.name, name, v)] = (preds, None)
                    row.update(gt_count=np.nan)
                rows.append(row)
            print(f"  {p.name}: " + " ".join(
                f"{v}={boxes_store[(p.name, name, v)][0].__len__()}" for v in a.variants)
                  + (f"  (GT={len(gt)})" if gt is not None else ""))

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    (out / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False, default=str),
                                       encoding="utf-8")

    if n_lab == 0:
        print("\nEtiket olmadığı için metrik hesaplanmadı. Tahminler per_image.csv içinde.")
        return
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    pivot = summary.pivot_table(index=["detector", "condition"], columns="variant",
                                values=["F1", "MAE"], sort=False)
    pivot.to_csv(out / "summary_pivot.csv")
    rq = research_table(summary, stats)
    rq.to_csv(out / "research_question.csv", index=False)

    # 3) Grafikler ve görseller
    plots.all_figures(summary, stats, rq, out / "figures", a.detectors, a.variants)
    if not a.no_vis:
        for cond, p, img, gt in data:
            plots.comparison_grid(p.name, gt, enhanced, boxes_store, a.detectors, a.variants,
                                  out / "vis" / f"{p.stem}_grid.jpg")

    pd.set_option("display.width", 200)
    print("\n=== ÖZET (tüm görüntüler) ===")
    print(summary[summary.condition == "ALL"][["detector", "variant", "TP", "FP", "FN", "precision",
                                               "recall", "F1", "MAE", "infer_ms", "e2e_ms"]].to_string(index=False))
    print(f"\nTüm çıktılar: {out}")


if __name__ == "__main__":
    main()
