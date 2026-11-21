"""IoU, bire bir eşleştirme ve performans ölçütleri (TP, FP, FN, P, R, F1, MAE)."""

IOU_THRESHOLD = 0.50


def iou(a, b):
    """İki kutunun kesişim / birleşim oranı. Kutu: (x1, y1, x2, y2, ...)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match(preds, truths, threshold=IOU_THRESHOLD):
    """Bire bir (greedy) eşleştirme: en yüksek IoU'lu çiftten başlayarak,
    her tahmin ve her gerçek kutu en fazla bir kez eşleşir; IoU >= eşik olmalı.

    Döndürür: (eslesmeler[(tahmin_i, gercek_j, iou)], TP, FP, FN)
    """
    pairs = []
    for i, t in enumerate(preds):
        for j, g in enumerate(truths):
            v = iou(t, g)
            if v >= threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    used_t, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if i not in used_t and j not in used_g:
            used_t.add(i)
            used_g.add(j)
            matches.append((i, j, v))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(truths) - tp
    return matches, tp, fp, fn


def summary(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
