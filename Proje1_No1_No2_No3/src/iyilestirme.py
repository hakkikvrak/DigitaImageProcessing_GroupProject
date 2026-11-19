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
PK_ALFA = 1.3        # kontrast katsayısı
PK_BETA = 30         # parlaklık ekleme değeri
CLAHE_CLIP = 2.0     # CLAHE kırpma sınırı
CLAHE_IZGARA = (8, 8)  # CLAHE karo ızgarası


def parlaklik_kontrast(img, alfa=PK_ALFA, beta=PK_BETA):
    """g(x,y) = alfa * f(x,y) + beta  ->  [0,255] aralığına kırpılır.
    Hazır fonksiyon (cv2.convertScaleAbs vb.) kullanılmadan numpy ile yazıldı."""
    sonuc = img.astype(np.float32) * alfa + beta
    sonuc = np.clip(sonuc, 0, 255)
    return np.round(sonuc).astype(np.uint8)


def histogram_esitle_gri(gri):
    """Tek kanallı (8 bit) görüntüde histogram eşitleme - elle yazılmıştır.

    Adımlar:
      1) Histogram: her gri seviyeden (0..255) kaç piksel var?
      2) Kümülatif dağılım (CDF)
      3) Dönüşüm tablosu: T(k) = round( (CDF(k) - CDF_min) / (N - CDF_min) * 255 )
      4) Her pikseli tablodan geçir.
    """
    hist = np.zeros(256, dtype=np.int64)
    for seviye, adet in zip(*np.unique(gri, return_counts=True)):
        hist[seviye] = adet

    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)[0][0]]          # ilk sıfır olmayan değer
    toplam = gri.size
    if toplam == cdf_min:                          # tek renkli görüntü
        return gri.copy()

    tablo = np.round((cdf - cdf_min) / (toplam - cdf_min) * 255)
    tablo = np.clip(tablo, 0, 255).astype(np.uint8)
    return tablo[gri]                              # look-up table uygulaması


def histogram_esitle(img):
    """Renkli görüntüde Y (parlaklık) kanalına elle yazılmış histogram eşitleme."""
    if img.ndim == 2:
        return histogram_esitle_gri(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = histogram_esitle_gri(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip=CLAHE_CLIP, izgara=CLAHE_IZGARA):
    """Adaptif histogram eşitleme (CLAHE) - OpenCV hazır fonksiyonu, LAB'ın L kanalında."""
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=izgara)
    if img.ndim == 2:
        return c.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Arayüzde ve deneyde kullanılan sıra ve Türkçe adlar
YONTEMLER = {
    "ham": ("Ham (işlemsiz)", lambda im: im),
    "parlaklik_kontrast": ("Parlaklık / Kontrast", parlaklik_kontrast),
    "histogram_esitleme": ("Histogram Eşitleme", histogram_esitle),
    "clahe": ("CLAHE (Adaptif HE)", clahe),
}


def uygula(img, yontem):
    return YONTEMLER[yontem][1](img)
