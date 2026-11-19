"""IoU, bire bir eşleştirme ve performans ölçütleri (TP, FP, FN, P, R, F1, MAE)."""

IOU_ESIK = 0.50


def iou(a, b):
    """İki kutunun kesişim / birleşim oranı. Kutu: (x1, y1, x2, y2, ...)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    kesisim = max(0, x2 - x1) * max(0, y2 - y1)
    alan_a = (a[2] - a[0]) * (a[3] - a[1])
    alan_b = (b[2] - b[0]) * (b[3] - b[1])
    birlesim = alan_a + alan_b - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def eslestir(tahminler, gercekler, esik=IOU_ESIK):
    """Bire bir (greedy) eşleştirme: en yüksek IoU'lu çiftten başlayarak,
    her tahmin ve her gerçek kutu en fazla bir kez eşleşir; IoU >= eşik olmalı.

    Döndürür: (eslesmeler[(tahmin_i, gercek_j, iou)], TP, FP, FN)
    """
    ciftler = []
    for i, t in enumerate(tahminler):
        for j, g in enumerate(gercekler):
            v = iou(t, g)
            if v >= esik:
                ciftler.append((v, i, j))
    ciftler.sort(reverse=True)

    kullanilan_t, kullanilan_g, eslesmeler = set(), set(), []
    for v, i, j in ciftler:
        if i not in kullanilan_t and j not in kullanilan_g:
            kullanilan_t.add(i)
            kullanilan_g.add(j)
            eslesmeler.append((i, j, v))

    tp = len(eslesmeler)
    fp = len(tahminler) - tp
    fn = len(gercekler) - tp
    return eslesmeler, tp, fp, fn


def ozet(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
