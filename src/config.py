"""
Projenin TÜM sabit ayarları burada.
Ödev kuralı: eşikler görüntüye göre DEĞİŞTİRİLMEZ. Tüm deneylerde bu değerler kullanılır.
Savunmada hoca bir parametreyi değiştirtirse: demo.py / run_experiments.py komut satırı
argümanlarıyla (ör. --minNeighbors 3, --conf 0.4) değiştirilebilir; bu dosyaya dokunmaya gerek yok.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_photos"          # telefondan gelen orijinal fotoğraflar
IMG_DIR = ROOT / "data" / "images"     # ön işlenmiş (küçültülmüş, EXIF'siz) görüntüler
LABEL_DIR = ROOT / "data" / "labels"   # ground truth (YOLO txt formatı)
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

CONDITIONS = ["normal", "low_light", "challenge"]   # challenge = ters ışık / küçük yüz / örtülme
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---- Ön işleme ----
MAX_SIDE = 1280          # uzun kenar bu değere küçültülür (etiketleme bu boyutta yapılır)
JPEG_QUALITY = 95

# ---- Değerlendirme ----
IOU_THRESHOLD = 0.50

# ---- Görüntü iyileştirme (tüm veri setinde sabit) ----
BC_ALPHA = 1.3           # kontrast kazancı   out = alpha*in + beta
BC_BETA = 20             # parlaklık ofseti
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# ---- Haar Cascade (Viola–Jones) ----
HAAR_XML = "haarcascade_frontalface_default.xml"   # OpenCV ile birlikte gelir
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)

# ---- Dlib HOG + Linear SVM (ek klasik dedektör) ----
HOG_UPSAMPLE = 1         # 1 = görüntü 2x büyütülerek aranır (küçük yüzler için)
HOG_THRESHOLD = 0.0      # dlib adjust_threshold

# ---- YOLO yüz modeli ----
YOLO_WEIGHTS = MODEL_DIR / "yolov11n-face.pt"
YOLO_WEIGHTS_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
YOLO_WEIGHTS_SHA256 = "9420a9d4933940d2202f511d45df1750e3c1546dd9efc7a3985caae3bbbf7ed2"
YOLO_CONF = 0.25
YOLO_NMS_IOU = 0.45
YOLO_IMGSZ = 1280        # görüntüler zaten 1280 px; küçültme yapılmasın (küçük yüzler kaybolmasın)
YOLO_DEVICE = "cpu"      # tekrarlanabilirlik ve adil süre karşılaştırması için CPU

SEED = 0                 # deterministik çalışma (YOLO çıkarımında rastgelelik yok, kayıt amaçlı)
TIMING_WARMUP = 2        # süre ölçümünden önce ısınma çalıştırması sayısı
