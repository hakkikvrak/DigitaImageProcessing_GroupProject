"""Image enhancement operations.

- brightness_contrast : OUR OWN linear transform  (g = alpha*f + beta)
- equalize_hist       : OUR OWN histogram equalization (cv2.equalizeHist NOT used)
- clahe               : OpenCV's built-in CLAHE (adaptive histogram equalization)

On color images the operations are applied only to the luminance channel (Y of YCrCb /
L of LAB); so colors are preserved and YOLO can still be given a color
image.
"""
import cv2
import numpy as np

# ---- Fixed parameters (do NOT change during the experiment) ----
BC_ALPHA = 1.3        # contrast gain
BC_BETA = 30         # brightness offset
CLAHE_CLIP = 2.0     # CLAHE clip limit
CLAHE_GRID = (8, 8)  # CLAHE tile grid


def brightness_contrast(img, alpha=BC_ALPHA, beta=BC_BETA):
    """g(x,y) = alpha * f(x,y) + beta  ->  clipped to [0,255].
    Written with numpy, without a ready-made function (cv2.convertScaleAbs etc.)."""
    result = img.astype(np.float32) * alpha + beta
    result = np.clip(result, 0, 255)
    return np.round(result).astype(np.uint8)


def equalize_hist_gray(gray):
    """Histogram equalization on a single-channel (8-bit) image - written by hand.

    Steps:
      1) Histogram: how many pixels of each gray level (0..255)?
      2) Cumulative distribution (CDF)
      3) Mapping table: T(k) = round( (CDF(k) - CDF_min) / (N - CDF_min) * 255 )
      4) Map every pixel through the table.
    """
    hist = np.zeros(256, dtype=np.int64)
    for level, count in zip(*np.unique(gray, return_counts=True)):
        hist[level] = count

    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)[0][0]]          # first non-zero value
    total = gray.size
    if total == cdf_min:                          # single-color image
        return gray.copy()

    table = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    table = np.clip(table, 0, 255).astype(np.uint8)
    return table[gray]                              # apply the look-up table


def equalize_hist(img):
    """Hand-written histogram equalization on the Y (luminance) channel of a color image."""
    if img.ndim == 2:
        return equalize_hist_gray(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = equalize_hist_gray(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    """Adaptive histogram equalization (CLAHE) - OpenCV built-in, on the L channel of LAB."""
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    if img.ndim == 2:
        return c.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Order and display names used in the GUI and the experiment
METHODS = {
    "ham": ("Raw (unprocessed)", lambda im: im),
    "parlaklik_kontrast": ("Brightness / Contrast", brightness_contrast),
    "histogram_esitleme": ("Histogram Equalization", equalize_hist),
    "clahe": ("CLAHE (Adaptive HE)", clahe),
}


def apply_method(img, method):
    return METHODS[method][1](img)
