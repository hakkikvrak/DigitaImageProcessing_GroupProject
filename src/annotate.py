"""
ADIM 1b - Basit etiketleme aracı (internet gerekmez, fotoğraflar bilgisayardan çıkmaz)

Kullanım:  python src/annotate.py              -> etiketi olmayan görüntüleri sırayla açar
           python src/annotate.py --all        -> tümünü yeniden etiketle
           python src/annotate.py --check      -> etiketleri görüntü üstüne çizip results/label_check/ içine kaydeder

Pencerede:
  - Fare ile yüzün çevresine kutu çizin, ENTER veya SPACE ile onaylayın.
  - Sonraki yüz için yeniden çizin.  Bitirince ESC'ye basın.
  - Hiç yüz yoksa doğrudan ESC -> boş etiket dosyası yazılır.
Kutu kuralı (tüm ekip aynı kuralı uygulasın): alın üst çizgisi - çene alt ucu, sol-sağ yanak kenarı, kulaklar hariç.
"""
import argparse
import cv2
from config import RESULTS_DIR
from dataset import list_images, label_path, xyxy_to_yolo, load_gt

MAX_VIEW = 1000   # ekrana sığması için gösterim boyutu (kayıt orijinal ölçekte yapılır)


def annotate(p):
    img = cv2.imread(str(p))
    H, W = img.shape[:2]
    s = min(1.0, MAX_VIEW / max(H, W))
    view = cv2.resize(img, (round(W * s), round(H * s))) if s < 1 else img.copy()
    title = f"{p.name} - kutu ciz, ENTER=onay, ESC=bitir"
    rois = cv2.selectROIs(title, view, showCrosshair=False)
    cv2.destroyAllWindows()
    boxes = [(round(x / s), round(y / s), round((x + w) / s), round((y + h) / s)) for x, y, w, h in rois]
    lp = label_path(p)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("\n".join(xyxy_to_yolo(b, W, H) for b in boxes) + ("\n" if boxes else ""))
    print(f"{p.name}: {len(boxes)} yüz -> {lp}")


def check():
    out = RESULTS_DIR / "label_check"
    out.mkdir(parents=True, exist_ok=True)
    for cond, p in list_images():
        img = cv2.imread(str(p))
        gt = load_gt(p, img.shape[1], img.shape[0])
        if gt is None:
            print(f"ETİKET YOK: {p.name}")
            continue
        for b in gt:
            cv2.rectangle(img, b[:2], b[2:], (0, 255, 0), 3)
        cv2.putText(img, f"GT={len(gt)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(str(out / p.name), img)
    print(f"Kontrol görselleri: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    else:
        for cond, p in list_images():
            if a.all or not label_path(p).exists():
                annotate(p)
