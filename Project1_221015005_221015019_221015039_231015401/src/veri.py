"""Dataset: images, ground-truth boxes and capture conditions."""
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root folder
DATA_DIR = os.path.join(ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "goruntuler")

CONDITION_NAMES = {
    "dusuk_isik": "Dim",
    "normal": "Normal",
    "asiri_parlak": "Bright",
}

DISTANCE_NAMES = {
    "uzak": "Far",
    "yakin": "Close",
}


def read_labels():
    """dosya_adi -> [(x1, y1, x2, y2), ...]  (Roboflow CSV format)"""
    boxes = defaultdict(list)
    with open(os.path.join(DATA_DIR, "etiketler.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            boxes[r["filename"]].append(
                (int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])))
    return boxes


def read_conditions():
    """[{'no', 'dosya', 'kosul', 'mesafe', 'gercek_yuz_sayisi', 'not'}, ...]"""
    with open(os.path.join(DATA_DIR, "kosullar.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_image(path):
    """Reading that also works on paths with Turkish characters (cv2.imread can fail on Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
