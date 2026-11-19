"""Veri kümesi: görüntüler, referans (ground truth) kutular ve çekim koşulları."""
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # proje kök klasörü
VERI = os.path.join(KOK, "veri")
GORUNTU_KLASORU = os.path.join(VERI, "goruntuler")

KOSUL_ADLARI = {
    "normal": "Normal ışık",
    "dusuk_isik": "Düşük ışık",
    "asiri_parlak": "Aşırı parlak",
}


def etiketleri_oku():
    """dosya_adi -> [(x1, y1, x2, y2), ...]  (Roboflow CSV formatı)"""
    kutular = defaultdict(list)
    with open(os.path.join(VERI, "etiketler.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            kutular[r["filename"]].append(
                (int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])))
    return kutular


def kosullari_oku():
    """[{'no', 'dosya', 'kosul', 'gercek_yuz_sayisi', 'not'}, ...]"""
    with open(os.path.join(VERI, "kosullar.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def goruntu_oku(yol):
    """Türkçe karakterli yollarda da çalışan okuma (cv2.imread Windows'ta sorun çıkarabilir)."""
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_COLOR)


def goruntu_yaz(yol, img):
    uzanti = os.path.splitext(yol)[1] or ".jpg"
    ok, buf = cv2.imencode(uzanti, img)
    if ok:
        buf.tofile(yol)
