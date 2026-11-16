"""
ADIM 5 - Performans ölçümü

IoU(A,B) = |A ∩ B| / |A ∪ B|
Bire bir eşleştirme: Macar algoritması (scipy linear_sum_assignment).
  - IoU < 0.50 olan çiftler eşleşemez.
  - Ağırlık = 1 + IoU  -> önce eşleşme SAYISI, eşitlikte toplam IoU en büyüklenir.
    (Böylece greedy yöntemdeki "iyi bir kutuyu yanlış yüze harcama" hatası olmaz.)
TP = eşleşen tahminler, FP = eşleşmeyen tahminler, FN = eşleşmeyen referans kutular
Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)
Sayma hatası (görüntü başına) = |tahmin sayısı - gerçek sayı|,  MAE = ortalaması
Veri seti geneli P/R/F1, toplam TP/FP/FN üzerinden hesaplanır (mikro ortalama).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import IOU_THRESHOLD


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr=IOU_THRESHOLD):
    """preds: [(x1,y1,x2,y2,score)], gts: [(x1,y1,x2,y2)]
    Dönüş: eşleşmeler [(pred_i, gt_j, iou)], eşleşmeyen pred indeksleri, eşleşmeyen gt indeksleri"""
    P, G = len(preds), len(gts)
    if P == 0 or G == 0:
        return [], list(range(P)), list(range(G))
    M = np.array([[iou(p[:4], g) for g in gts] for p in preds])
    W = np.where(M >= thr, 1.0 + M, 0.0)
    rows, cols = linear_sum_assignment(W, maximize=True)
    pairs = [(int(r), int(c), float(M[r, c])) for r, c in zip(rows, cols) if M[r, c] >= thr]
    mp = {r for r, _, _ in pairs}
    mg = {c for _, c, _ in pairs}
    return pairs, [i for i in range(P) if i not in mp], [j for j in range(G) if j not in mg]


def image_metrics(preds, gts, thr=IOU_THRESHOLD):
    pairs, fp, fn = match(preds, gts, thr)
    return {
        "tp": len(pairs), "fp": len(fp), "fn": len(fn),
        "pred_count": len(preds), "gt_count": len(gts),
        "abs_count_err": abs(len(preds) - len(gts)),
        "mean_iou_tp": float(np.mean([p[2] for p in pairs])) if pairs else np.nan,
        "_pairs": pairs, "_fp": fp, "_fn": fn,
    }


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


if __name__ == "__main__":
    # Küçük birim testleri
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9
    gts = [(0, 0, 10, 10), (20, 0, 30, 10)]
    preds = [(1, 0, 11, 10, .9), (21, 0, 31, 10, .8), (100, 100, 110, 110, .5)]
    m = image_metrics(preds, gts)
    assert (m["tp"], m["fp"], m["fn"], m["abs_count_err"]) == (2, 1, 0, 1), m
    assert image_metrics([], gts)["fn"] == 2 and image_metrics(preds, [])["fp"] == 3
    # Bire bir: aynı yüze iki kutu -> 1 TP + 1 FP
    m = image_metrics([(0, 0, 10, 10, .9), (0, 0, 10, 11, .8)], [(0, 0, 10, 10)])
    assert (m["tp"], m["fp"]) == (1, 1)
    print("evaluate.py testleri geçti. P/R/F1(2,1,0) =", prf(2, 1, 0))
