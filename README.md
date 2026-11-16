# Zorlu Görüntüleme Koşullarında Yüz Tespiti ve Kişi Sayma
### Klasik Yöntem (Haar Cascade, HOG) ve YOLO Karşılaştırması

Grup: `221015019` - Batuhan Kocamanoğlu · `221015039` - Hakkı Kıvrak · `231015401` - Ulaş Deniz Ay

---

## 1. Kurulum

```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

`dlib-bin` kurulamazsa HOG isteğe bağlıdır; tüm komutlara `--detectors haar yolo` ekleyin.

## 2. Kullanılan hazır modeller

| Dedektör | Model / sürüm | Kaynak |
|---|---|---|
| Haar Cascade (Viola–Jones) | `haarcascade_frontalface_default.xml` (OpenCV 4.x ile gelir, kopyası `models/` içinde) | https://github.com/opencv/opencv/tree/4.x/data/haarcascades |
| Dlib HOG + Linear SVM | `dlib.get_frontal_face_detector()` (dlib 20.0.1) | http://dlib.net/face_detector.py.html |
| YOLO yüz modeli | `yolov11n-face.pt`: YOLO11n, WIDER FACE ile yüz için eğitilmiş, tek sınıf `face`, akanametov/yolo-face sürüm 1.0.0, SHA-256 `9420a9d4…7ed2` | https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt |

`models/yolov11n-face.pt` yoksa ilk çalıştırmada otomatik indirilir ve SHA-256 değeri kontrol edilir.
Genel YOLO "person" sınıfı **kullanılmamıştır**.

## 3. Çalıştırma adımları (tekrarlanabilir)

```bash
# 1) Fotoğrafları raw_photos/normal, raw_photos/low_light, raw_photos/challenge içine koyun
python src/preprocess.py            # EXIF yön düzeltme + 1280 px + EXIF silme -> data/images/
# 2) Etiketleme (her yüze kutu çiz, ENTER onay, ESC sonraki görüntü)
python src/annotate.py
python src/annotate.py --check      # etiketleri görsel olarak kontrol et -> results/label_check/
# 3) Tüm deneyler (3 dedektör × 4 sürüm), tablolar, grafikler
python src/run_experiments.py
# 4) Canlı gösterim (herhangi bir görüntü)
python src/demo.py --image yol/yeni.jpg --show
python src/demo.py --image yol/yeni.jpg --minNeighbors 3 --conf 0.4    # parametre değişikliği
python src/demo.py --camera --detectors haar yolo                       # web kamerası
# Doğrulamalar
python src/enhance.py                # elle yazılan histogram eşitleme == cv2.equalizeHist (fark 0)
python src/evaluate.py               # IoU / eşleştirme birim testleri
python src/make_flowchart.py         # report/akis_semasi.png
```

## 4. Sabit deney ayarları (`src/config.py`)

| Ayar | Değer |
|---|---|
| Görüntü boyutu | uzun kenar 1280 px |
| Parlaklık/kontrast | `out = clip(1.3·in + 20)` |
| Histogram eşitleme | **kendi uygulamamız** (`equalize_hist_manual`), YCrCb-Y kanalında |
| CLAHE | clipLimit 2.0, tile 8×8, YCrCb-Y kanalında |
| Haar | scaleFactor 1.1, minNeighbors 5, minSize 30×30 |
| HOG | upsample 1, adjust_threshold 0 |
| YOLO | conf 0.25, NMS IoU 0.45, imgsz 1280, CPU |
| Eşleştirme | IoU ≥ 0.50, bire bir (Macar algoritması) |
| Süre | `infer_ms` = yalnızca dedektör; `e2e_ms` = iyileştirme + dedektör (disk okuma hariç), 2 ısınma çalıştırması |

Eşikler hiçbir görüntü için ayrıca değiştirilmemiştir. Rastgelelik yoktur (seed = 0 kayıt amaçlıdır).

## 5. Çıktılar (`results/`)

| Dosya | İçerik |
|---|---|
| `gt_counts.csv` | görüntü başına gerçek yüz sayısı |
| `per_image.csv` | görüntü × dedektör × sürüm: kutular, TP/FP/FN, sayma hatası, süreler |
| `summary.csv` | dedektör × sürüm × koşul: P, R, F1, MAE, ortalama süre |
| `summary_pivot.csv` | F1 / MAE özet tablosu |
| `image_stats.csv` | her sürümün parlaklık, RMS kontrast, entropi değerleri |
| `research_question.csv` | ham'a göre kontrast değişimi ↔ F1 değişimi |
| `settings.json` | tüm parametreler ve kütüphane sürümleri |
| `figures/` | rapor grafikleri |
| `vis/` | her görüntü için dedektör × sürüm karşılaştırma ızgarası |

## 6. Klasör yapısı

```
├── src/            kaynak kod
├── models/         Haar XML + YOLO ağırlıkları
├── raw_photos/     (zip'e EKLENMEZ: orijinal, EXIF'li fotoğraflar)
├── data/images/    kullanılan veri seti (normal / low_light / challenge)
├── data/labels/    referans etiketler (YOLO txt: 0 cx cy w h, normalize)
├── results/        tablolar, grafikler, görseller
└── report/         Word raporu + akış şeması
```

## 7. Etik ve yapay zekâ kullanımı
- Görüntüler yalnızca grup üyelerinin bilgilendirilmiş onayıyla çekilmiştir; çocuk görüntüsü yoktur.
- EXIF (konum ve cihaz) verisi silinmiştir. Görüntüler yayımlanmamıştır.
- Yüz **tanıma** veya kimlik etiketlemesi yapılmamıştır; tek sınıf `face` kullanılmıştır.
- Kodun önemli bir kısmı yapay zekâ asistanı (Claude, Anthropic) desteğiyle yazılmış; ekip tarafından
  incelenmiş, test edilmiş ve anlaşılmıştır. Ayrıntılar raporun "Yapay Zekâ Kullanımı" bölümündedir.
