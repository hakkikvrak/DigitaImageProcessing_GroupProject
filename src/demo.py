"""
CANLI GÖSTERİM - herhangi bir görüntüde (hocanın getirdiği yeni görüntü dahil) çalışır.

  python src/demo.py --image yeni.jpg                          # 3 dedektör × 4 sürüm ızgarası
  python src/demo.py --image yeni.jpg --detectors haar yolo --variants raw clahe
  python src/demo.py --image yeni.jpg --minNeighbors 3 --conf 0.4   # parametre değişikliği
  python src/demo.py --image data/images/low_light/low_light_01.jpg # etiketliyse TP/FP/FN de yazar
  python src/demo.py --camera                                       # web kamerası (q ile çıkış)

Sonuç results/demo/ içine kaydedilir; --show ile pencerede de açılır.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config as C
import enhance as E
import detectors as D
import evaluate as EV
from dataset import load_gt, label_path
from run_experiments import overrides
from plots import comparison_grid, draw


def prepare(img):
    """Ön işlemeyle aynı ölçek: uzun kenar en fazla MAX_SIDE."""
    s = C.MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--detectors", nargs="+", default=["haar", "hog", "yolo"], choices=list(D.DETECTORS))
    ap.add_argument("--variants", nargs="+", default=E.VARIANTS, choices=E.VARIANTS)
    ap.add_argument("--scaleFactor", type=float)
    ap.add_argument("--minNeighbors", type=int)
    ap.add_argument("--upsample", type=int)
    ap.add_argument("--conf", type=float)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    dets = {n: D.build(n, **overrides(a, n)) for n in a.detectors}
    for n, d in dets.items():
        print(f"{d.label}: {d.describe()}")

    if a.camera:
        return camera(dets, a.variants[-1])
    if not a.image:
        ap.error("--image veya --camera gerekli")

    p = Path(a.image)
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"Görüntü okunamadı: {p}")
    img = prepare(img)
    gt = load_gt(p, img.shape[1], img.shape[0]) if label_path(p).exists() else None
    print(f"\n{p.name} {img.shape[1]}x{img.shape[0]}" + (f"  GT={len(gt)} yüz" if gt is not None else "  (etiket yok)"))

    enhanced, store = {}, {}
    for v in a.variants:
        enhanced[(p.name, v)] = (E.apply(img, v), 0)
    for n, d in dets.items():
        d.detect(img)  # ısınma
        for v in a.variants:
            e = enhanced[(p.name, v)][0]
            t0 = time.perf_counter()
            preds = d.detect(e)
            ms = (time.perf_counter() - t0) * 1000
            m = EV.image_metrics(preds, gt) if gt is not None else None
            store[(p.name, n, v)] = (preds, m)
            extra = f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} |hata|={m['abs_count_err']}" if m else ""
            print(f"  {n:5s} {v:6s}: {len(preds)} yüz  {ms:7.1f} ms{extra}")

    out = C.RESULTS_DIR / "demo"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{p.stem}_demo.jpg"
    comparison_grid(p.name, gt, enhanced, store, a.detectors, a.variants, path)
    print(f"\nKaydedildi: {path}")
    if a.show:
        g = cv2.imread(str(path))
        s = min(1.0, 1600 / g.shape[1])
        cv2.imshow("demo", cv2.resize(g, None, fx=s, fy=s))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def camera(dets, variant):
    cap = cv2.VideoCapture(0)
    print(f"Kamera açık, sürüm: {variant}. Tuşlar: 1=raw 2=bc 3=he 4=clahe, q=çıkış")
    keys = {ord("1"): "raw", ord("2"): "bc", ord("3"): "he", ord("4"): "clahe"}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        e = E.apply(frame, variant)
        panels = []
        for n, d in dets.items():
            t0 = time.perf_counter()
            preds = d.detect(e)
            ms = (time.perf_counter() - t0) * 1000
            c = draw(e, None, preds, None, 1.0)
            cv2.putText(c, f"{n} | {variant} | {len(preds)} yuz | {ms:.0f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(c)
        cv2.imshow("canli", np.hstack(panels))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        variant = keys.get(k, variant)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
