"""Görüntü iyileştirme işlemleri.

- parlaklik_kontrast : KENDİ YAZDIĞIMIZ doğrusal dönüşüm  (g = alfa*f + beta)
- histogram_esitle   : KENDİ YAZDIĞIMIZ histogram eşitleme (cv2.equalizeHist KULLANILMADI)
- clahe              : OpenCV'nin hazır CLAHE fonksiyonu (adaptif histogram eşitleme)

Renkli görüntüde işlemler sadece parlaklık kanalına (YCrCb'nin Y kanalı /
LAB'ın L kanalı) uygulanır; böylece renkler bozulmaz ve YOLO'ya renkli
görüntü verilebilir.
"""
import cv2
import numpy as np

# ---- Sabit parametreler (deney boyunca DEĞİŞMEZ) ----
BC_ALPHA = 1.3        # kontrast katsayısı
BC_BETA = 30         # parlaklık ekleme değeri
CLAHE_CLIP = 2.0     # CLAHE kırpma sınırı
CLAHE_GRID = (8, 8)  # CLAHE karo ızgarası


def brightness_contrast(img, alpha=BC_ALPHA, beta=BC_BETA):
    """g(x,y) = alfa * f(x,y) + beta  ->  [0,255] aralığına kırpılır.
    Hazır fonksiyon (cv2.convertScaleAbs vb.) kullanılmadan numpy ile yazıldı."""
    result = img.astype(np.float32) * alpha + beta
    result = np.clip(result, 0, 255)
    return np.round(result).astype(np.uint8)


def equalize_hist_gray(gray):
    """Tek kanallı (8 bit) görüntüde histogram eşitleme - elle yazılmıştır.

    Adımlar:
      1) Histogram: her gri seviyeden (0..255) kaç piksel var?
      2) Kümülatif dağılım (CDF)
      3) Dönüşüm tablosu: T(k) = round( (CDF(k) - CDF_min) / (N - CDF_min) * 255 )
      4) Her pikseli tablodan geçir.
    """
    hist = np.zeros(256, dtype=np.int64)
    for level, count in zip(*np.unique(gray, return_counts=True)):
        hist[level] = count

    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)[0][0]]          # ilk sıfır olmayan değer
    total = gray.size
    if total == cdf_min:                          # tek renkli görüntü
        return gray.copy()

    table = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    table = np.clip(table, 0, 255).astype(np.uint8)
    return table[gray]                              # look-up table uygulaması


def equalize_hist(img):
    """Renkli görüntüde Y (parlaklık) kanalına elle yazılmış histogram eşitleme."""
    if img.ndim == 2:
        return equalize_hist_gray(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = equalize_hist_gray(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    """Adaptif histogram eşitleme (CLAHE) - OpenCV hazır fonksiyonu, LAB'ın L kanalında."""
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    if img.ndim == 2:
        return c.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Arayüzde ve deneyde kullanılan sıra ve Türkçe adlar
METHODS = {
    "ham": ("Raw (unprocessed)", lambda im: im),
    "parlaklik_kontrast": ("Brightness / Contrast", brightness_contrast),
    "histogram_esitleme": ("Histogram Equalization", equalize_hist),
    "clahe": ("CLAHE (Adaptive HE)", clahe),
}


def apply_method(img, method):
    return METHODS[method][1](img)
