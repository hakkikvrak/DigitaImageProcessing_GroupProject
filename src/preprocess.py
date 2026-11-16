"""
ADIM 1 - Ön işleme (tek seferlik)
raw_photos/<koşul>/*.jpg  ->  data/images/<koşul>/*.jpg

Yapılanlar:
  1) EXIF yön bilgisine göre döndürme (telefon fotoğrafları yan görünmesin)
  2) Uzun kenarı MAX_SIDE (1280) piksele küçültme  -> etiketleme bu boyutta yapılır
  3) EXIF verisini SİLME (GPS konumu, cihaz bilgisi vb. -> veri güvenliği)
  4) Dosyaları <koşul>_NN.jpg şeklinde yeniden adlandırma

Kullanım:  python src/preprocess.py
"""
from PIL import Image, ImageOps
from config import RAW_DIR, IMG_DIR, CONDITIONS, IMG_EXTS, MAX_SIDE, JPEG_QUALITY


def process_one(src, dst):
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)          # yönü düzelt
    im = im.convert("RGB")
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    # exif parametresi verilmediği için EXIF kaydedilmez
    im.save(dst, "JPEG", quality=JPEG_QUALITY)
    return im.size


def main():
    total = 0
    for cond in CONDITIONS:
        src_dir, dst_dir = RAW_DIR / cond, IMG_DIR / cond
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
        for i, f in enumerate(files, 1):
            dst = dst_dir / f"{cond}_{i:02d}.jpg"
            size = process_one(f, dst)
            print(f"{f.name:30s} -> {dst.relative_to(IMG_DIR.parent.parent)}  {size}")
            total += 1
        print(f"[{cond}] {len(files)} görüntü")
    print(f"Toplam {total} görüntü işlendi.")


if __name__ == "__main__":
    main()
