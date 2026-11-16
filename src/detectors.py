"""
ADIM 2 - Yüz dedektörleri (ortak arayüz)

Her dedektör:  detect(bgr) -> list[(x1, y1, x2, y2, score)]   (piksel koordinatı, int)
  - haar : OpenCV Haar Cascade (Viola–Jones)        -> zorunlu klasik dedektör
  - hog  : dlib HOG + Linear SVM                     -> ek klasik dedektör
  - yolo : YOLOv11n-face (WIDER FACE ile eğitilmiş)  -> zorunlu YOLO yüz dedektörü

Haar ve HOG gri görüntüde çalışır (dönüşüm dedektörün içinde yapılır).
Not: Haar'dan önce equalizeHist UYGULANMAZ; eşitleme yalnızca 'he' deney sürümünde var.
"""
import hashlib
import urllib.request
import cv2
import numpy as np
import config as C


class HaarDetector:
    name = "haar"
    label = "Haar Cascade (Viola–Jones)"

    def __init__(self, scale_factor=C.HAAR_SCALE_FACTOR, min_neighbors=C.HAAR_MIN_NEIGHBORS,
                 min_size=C.HAAR_MIN_SIZE):
        # Önce proje içindeki kopya (models/), yoksa OpenCV paketinin içindeki dosya
        local = C.MODEL_DIR / C.HAAR_XML
        path = str(local) if local.exists() else cv2.data.haarcascades + C.HAAR_XML
        self.clf = cv2.CascadeClassifier(path)
        if self.clf.empty():
            raise RuntimeError(f"Haar modeli yüklenemedi: {path}")
        self.params = dict(scaleFactor=scale_factor, minNeighbors=min_neighbors,
                           minSize=tuple(min_size))

    def describe(self):
        return {"model": f"OpenCV {cv2.__version__} / {C.HAAR_XML}", **self.params}

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # detectMultiScale3 -> kutular + her kutunun son aşama ağırlığı (güven skoru olarak)
        boxes, _, weights = self.clf.detectMultiScale3(gray, outputRejectLevels=True, **self.params)
        out = []
        for (x, y, w, h), s in zip(boxes, np.ravel(weights) if len(boxes) else []):
            out.append((int(x), int(y), int(x + w), int(y + h), float(s)))
        return out


class HogDetector:
    name = "hog"
    label = "Dlib HOG + Linear SVM"

    def __init__(self, upsample=C.HOG_UPSAMPLE, threshold=C.HOG_THRESHOLD):
        import dlib
        self.dlib = dlib
        self.det = dlib.get_frontal_face_detector()
        self.upsample, self.threshold = upsample, threshold

    def describe(self):
        return {"model": f"dlib {self.dlib.__version__} get_frontal_face_detector",
                "upsample": self.upsample, "adjust_threshold": self.threshold}

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        rects, scores, _ = self.det.run(gray, self.upsample, self.threshold)
        out = []
        for r, s in zip(rects, scores):
            x1, y1 = max(0, r.left()), max(0, r.top())
            x2, y2 = min(W, r.right()), min(H, r.bottom())
            out.append((x1, y1, x2, y2, float(s)))
        return out


def ensure_yolo_weights(path=C.YOLO_WEIGHTS):
    """Ağırlık dosyası yoksa GitHub'dan indir ve SHA-256 ile doğrula."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"YOLO ağırlıkları indiriliyor: {C.YOLO_WEIGHTS_URL}")
        urllib.request.urlretrieve(C.YOLO_WEIGHTS_URL, path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != C.YOLO_WEIGHTS_SHA256:
        print(f"UYARI: ağırlık dosyasının SHA-256 değeri beklenenden farklı ({h})")
    return path


class YoloFaceDetector:
    name = "yolo"
    label = "YOLOv11n-face"

    def __init__(self, conf=C.YOLO_CONF, iou=C.YOLO_NMS_IOU, imgsz=C.YOLO_IMGSZ, device=C.YOLO_DEVICE):
        from ultralytics import YOLO
        import ultralytics
        self.version = ultralytics.__version__
        self.model = YOLO(str(ensure_yolo_weights()))
        if list(self.model.names.values()) != ["face"]:
            raise RuntimeError(f"Model yüz modeli değil: {self.model.names}")
        self.params = dict(conf=conf, iou=iou, imgsz=imgsz, device=device)

    def describe(self):
        return {"model": f"yolov11n-face.pt ({C.YOLO_WEIGHTS_URL}) / ultralytics {self.version}",
                **self.params}

    def detect(self, bgr):
        r = self.model.predict(bgr, verbose=False, **self.params)[0]
        out = []
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), float(s)))
        return out


DETECTORS = {"haar": HaarDetector, "hog": HogDetector, "yolo": YoloFaceDetector}


def build(name, **overrides):
    """overrides: komut satırından gelen parametre değişiklikleri (None olanlar yok sayılır)."""
    kw = {k: v for k, v in overrides.items() if v is not None}
    return DETECTORS[name](**kw)
