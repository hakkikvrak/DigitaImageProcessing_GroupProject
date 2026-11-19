"""İki yüz dedektörü: Klasik Haar Cascade (Viola-Jones) ve YOLOv8n-face.

Her dedektörün `tespit(img)` fonksiyonu  [(x1, y1, x2, y2, skor), ...] listesi döndürür.
Parametreler sabittir; deney boyunca test görüntülerine göre DEĞİŞTİRİLMEZ.
"""
import os
import cv2

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # proje kök klasörü
MODEL_KLASORU = os.path.join(KOK, "modeller")

# ---- Sabit parametreler ----
HAAR_PARAM = dict(scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
YOLO_PARAM = dict(conf=0.25, iou=0.45, imgsz=1280)


class HaarDedektor:
    ad = "Haar Cascade"

    def __init__(self, **param):
        self.param = {**HAAR_PARAM, **param}
        yol = os.path.join(MODEL_KLASORU, "haarcascade_frontalface_default.xml")
        if not os.path.exists(yol):  # yedek: OpenCV ile gelen dosya
            yol = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        # Not: Windows'ta yol Türkçe karakter içerirse (ör. "diğ") CascadeClassifier(yol)
        # dosyayı açamıyor. Bu yüzden XML'i Python ile okuyup bellekten yüklüyoruz.
        with open(yol, encoding="utf-8") as f:
            xml = f.read()
        fs = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        self.model = cv2.CascadeClassifier()
        self.model.read(fs.getFirstTopLevelNode())
        if self.model.empty():
            raise RuntimeError("Haar cascade dosyası yüklenemedi: " + yol)

    def tespit(self, img):
        gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        kutular = self.model.detectMultiScale(gri, **self.param)
        # Haar skor üretmez; 1.0 yazılır
        return [(int(x), int(y), int(x + w), int(y + h), 1.0) for (x, y, w, h) in kutular]


class YoloDedektor:
    ad = "YOLOv8n-face"

    def __init__(self, **param):
        from ultralytics import YOLO  # ağır kütüphane; sadece gerekince yüklenir
        self.param = {**YOLO_PARAM, **param}
        yol = os.path.join(MODEL_KLASORU, "yolov8n-face.pt")
        if not os.path.exists(yol):
            raise FileNotFoundError(
                "YOLO yüz modeli bulunamadı: " + yol + "\nİndirme adresi: "
                "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt")
        self.model = YOLO(yol)

    def tespit(self, img):
        sonuc = self.model.predict(img, verbose=False, device="cpu", **self.param)[0]
        cikti = []
        for kutu, skor in zip(sonuc.boxes.xyxy.tolist(), sonuc.boxes.conf.tolist()):
            x1, y1, x2, y2 = kutu
            cikti.append((int(x1), int(y1), int(x2), int(y2), float(skor)))
        return cikti
