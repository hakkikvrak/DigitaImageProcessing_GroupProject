"""
Veri seti okuma.
Görüntü : data/images/<koşul>/<ad>.jpg
Etiket  : data/labels/<koşul>/<ad>.txt   (YOLO formatı, her satır bir yüz)
          <sınıf> <cx> <cy> <w> <h>       (0-1 arası normalize; sınıf her zaman 0 = face)
Boş .txt dosyası = görüntüde yüz yok (geçerli etiket).
makesense.ai / LabelImg / CVAT "YOLO" çıktısı ve annotate.py bu formatı üretir.
"""
import csv
import cv2
from config import IMG_DIR, LABEL_DIR, CONDITIONS, IMG_EXTS


def yolo_to_xyxy(line, W, H):
    _, cx, cy, w, h = map(float, line.split()[:5])
    return (round((cx - w / 2) * W), round((cy - h / 2) * H),
            round((cx + w / 2) * W), round((cy + h / 2) * H))


def xyxy_to_yolo(box, W, H):
    x1, y1, x2, y2 = box
    return f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def label_path(img_path):
    return LABEL_DIR / img_path.parent.name / (img_path.stem + ".txt")


def load_gt(img_path, W, H):
    """Etiket dosyası yoksa None döner (değerlendirme dışı)."""
    p = label_path(img_path)
    if not p.exists():
        return None
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    return [yolo_to_xyxy(l, W, H) for l in lines]


def list_images(conditions=CONDITIONS):
    items = []
    for cond in conditions:
        for p in sorted((IMG_DIR / cond).glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((cond, p))
    return items


def load_dataset():
    """[(koşul, yol, bgr, gt_kutular veya None), ...]"""
    data = []
    for cond, p in list_images():
        img = cv2.imread(str(p))
        if img is None:
            print(f"UYARI: okunamadı {p}")
            continue
        H, W = img.shape[:2]
        data.append((cond, p, img, load_gt(p, W, H)))
    return data


def write_gt_counts(data, out_csv):
    """Gerçek yüz sayıları tablosu (ödevde 'gerçek sayıyı kaydedin' isteniyor)."""
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "condition", "width", "height", "gt_faces"])
        for cond, p, img, gt in data:
            w.writerow([p.name, cond, img.shape[1], img.shape[0], "" if gt is None else len(gt)])
