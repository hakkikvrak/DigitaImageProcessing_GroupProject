"""
ADIM 3 - Görüntü iyileştirme
  raw   : işlenmemiş görüntü
  bc    : parlaklık / kontrast ayarı          out = clip(alpha*in + beta)
  he    : histogram eşitleme  -> KENDİ YAZDIĞIMIZ uygulama (cv2.equalizeHist KULLANILMAZ)
  clahe : kontrast sınırlı adaptif histogram eşitleme (OpenCV)

Renkli görüntülerde eşitleme R,G,B kanallarına ayrı ayrı uygulanmaz (renkler bozulur).
Görüntü YCrCb uzayına çevrilir, işlem yalnızca parlaklık (Y) kanalına uygulanır,
sonra tekrar BGR'ye dönülür. Böylece YOLO renkli, Haar/HOG gri görüntüyü aynı
iyileştirilmiş kaynaktan alır.

Doğrulama:  python src/enhance.py   -> elle yazılan HE ile cv2.equalizeHist karşılaştırılır
"""
import numpy as np
import cv2
from config import BC_ALPHA, BC_BETA, CLAHE_CLIP, CLAHE_TILE

VARIANTS = ["raw", "bc", "he", "clahe"]
VARIANT_NAMES = {
    "raw": "Ham",
    "bc": "Parlaklık/Kontrast",
    "he": "Histogram Eşitleme (elle)",
    "clahe": "CLAHE",
}


# ------------------------------------------------------------------
# 1) Parlaklık / kontrast  (doğrusal nokta işlemi, NumPy ile)
# ------------------------------------------------------------------
def brightness_contrast(bgr, alpha=BC_ALPHA, beta=BC_BETA):
    out = alpha * bgr.astype(np.float32) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# 2) Histogram eşitleme - HAZIR FONKSİYON KULLANMADAN
# ------------------------------------------------------------------
def equalize_hist_manual(gray):
    """
    8-bit tek kanallı görüntü için klasik histogram eşitleme.
      h(k)   : k gri seviyesindeki piksel sayısı            (k = 0..255)
      cdf(k) : sum_{j<=k} h(j)                               (kümülatif dağılım)
      T(k)   = round( (cdf(k) - cdf_min) / (N - cdf_min) * 255 )
    cdf_min: sıfır olmayan ilk cdf değeri, N: toplam piksel sayısı.
    T bir arama tablosudur (LUT); her piksel out = T[in] ile dönüştürülür.
    """
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("8-bit tek kanallı görüntü bekleniyor")
    # histogram: np.bincount ile 256 kutu
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.int64)
    cdf = np.cumsum(hist)
    N = cdf[-1]
    cdf_min = cdf[np.nonzero(hist)[0][0]]
    if N == cdf_min:                     # tek renkli görüntü -> değişiklik yok
        return gray.copy()
    lut = np.round((cdf - cdf_min) / (N - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]                     # LUT uygulama (NumPy indeksleme)


def _on_luma(bgr, fn):
    """fn işlemini yalnızca Y (parlaklık) kanalına uygula."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = fn(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def hist_eq(bgr):
    return _on_luma(bgr, equalize_hist_manual)


# ------------------------------------------------------------------
# 3) CLAHE (OpenCV)
# ------------------------------------------------------------------
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)


def clahe(bgr):
    return _on_luma(bgr, _clahe.apply)


def apply(bgr, variant):
    if variant == "raw":
        return bgr
    if variant == "bc":
        return brightness_contrast(bgr)
    if variant == "he":
        return hist_eq(bgr)
    if variant == "clahe":
        return clahe(bgr)
    raise ValueError(f"Bilinmeyen sürüm: {variant}")


# ------------------------------------------------------------------
# Görsel kalite istatistikleri (araştırma sorusu için)
# ------------------------------------------------------------------
def image_stats(bgr):
    """Ortalama parlaklık, RMS kontrast (std) ve entropi (bit) - gri seviye üzerinden."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p = np.bincount(g.ravel(), minlength=256) / g.size
    p = p[p > 0]
    return {
        "mean_brightness": float(g.mean()),
        "rms_contrast": float(g.std()),
        "entropy": float(-(p * np.log2(p)).sum()),
    }


if __name__ == "__main__":
    # Elle yazılan HE'nin doğrulaması: OpenCV ile karşılaştır (yalnızca test amaçlı)
    import sys
    rng = np.random.default_rng(0)
    tests = [rng.integers(40, 120, (480, 640), dtype=np.uint8)]   # dar histogramlı rastgele görüntü
    for path in sys.argv[1:]:
        tests.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    for i, g in enumerate(tests):
        mine, ref = equalize_hist_manual(g), cv2.equalizeHist(g)
        d = np.abs(mine.astype(int) - ref.astype(int))
        print(f"test {i}: maks fark = {d.max()}, farklı piksel oranı = {(d > 0).mean():.4%}")
