# Zorlu Görüntüleme Koşullarında Yüz Tespiti ve Kişi Sayma: Haar Cascade ve YOLO

Bu projede aynı 28 fotoğraf üzerinde iki yüz dedektörü karşılaştırılır:
klasik **Haar Cascade (Viola–Jones)** ve yüz için eğitilmiş **YOLOv8n-face**. Her fotoğraf
4 sürümde test edilir: ham, parlaklık/kontrast, histogram eşitleme ve CLAHE.

## Hızlı başlangıç

```bash
pip install -r requirements.txt
python arayuz.py          # arayüz (canlı gösterim)
python src/deney.py       # tüm deney: tablolar ve grafikler -> sonuclar/
```

### Arayüz nasıl kullanılır?
1. **Fotoğraf:** listeden bir fotoğraf seçin. Listede numara, koşul ve gerçek yüz sayısı yazar.
   Etiketsiz yeni bir fotoğraf için **Bilgisayardan aç…** butonunu kullanın.
2. **İyileştirme:** Ham / Parlaklık-Kontrast / Histogram Eşitleme / CLAHE seçin.
3. **▶ TESPİT ET** butonuna basın. Haar solda, YOLO sağda görünür.
   - Yeşil kutu = gerçek yüz (etiket), mavi = doğru tespit, kırmızı = yanlış tespit
   - Altta: bulunan/gerçek yüz sayısı, doğru, yanlış, kaçırılan ve süre (ms)
- **Ayarlar:** Haar `minNeighbors` ve YOLO güven eşiği savunmada değiştirilebilir.
  **Varsayılan** butonu deneydeki sabit değerlere döndürür.
- **Tüm deneyi çalıştır:** 28 fotoğraf × 4 yöntem × 2 dedektör. Sonuç tablosu pencerede görünür ve dosyalar `sonuclar/` klasörüne yazılır.

## Klasör yapısı

```
Proje1_No1_No2_No3/
├── arayuz.py                 Tkinter arayüzü
├── src/
│   ├── iyilestirme.py        parlaklık/kontrast + histogram eşitleme (ELLE YAZILDI), CLAHE (OpenCV)
│   ├── dedektorler.py        HaarDedektor, YoloDedektor (sabit parametreler burada)
│   ├── metrikler.py          IoU, bire bir eşleştirme, P/R/F1
│   ├── veri.py               etiket ve koşul dosyalarını okuma
│   └── deney.py              toplu deney, tablolar, grafikler
├── modeller/                 yolov8n-face.pt, haarcascade_frontalface_default.xml
├── veri/
│   ├── goruntuler/           28 özgün fotoğraf (Roboflow dışa aktarımı)
│   ├── etiketler.csv         85 referans yüz kutusu (filename, xmin, ymin, xmax, ymax)
│   └── kosullar.csv          fotoğraf → koşul (normal / dusuk_isik / asiri_parlak)
├── sonuclar/                 ozet.csv, kosul_ozet.csv, goruntu_bazli.csv, parametreler.json, grafikler/, ornekler/
├── akis_semasi.png
├── Rapor.docx
└── requirements.txt
```

## Hazır modeller

| Dedektör | Dosya | Sürüm / kaynak |
|---|---|---|
| Haar Cascade (Viola–Jones) | `haarcascade_frontalface_default.xml` | OpenCV 4.x: https://github.com/opencv/opencv/tree/4.x/data/haarcascades |
| YOLO yüz dedektörü | `yolov8n-face.pt` (YOLOv8 nano, tek sınıf `face`, WIDER FACE ile eğitilmiş) | akanametov/yolo-face, sürüm 1.0.0: https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt |

- `yolov8n-face.pt` SHA-256: `d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca`
- Genel YOLO "person" sınıfı **kullanılmadı**. Model doğrudan yüz kutusu üretir.
- Çıkarım için Ultralytics kullanıldı: https://github.com/ultralytics/ultralytics

## Sabit deney ayarları
Değerler test fotoğraflarına göre ayarlanmadı ve deney boyunca değiştirilmedi.

| Ayar | Değer |
|---|---|
| Haar | scaleFactor=1.1, minNeighbors=5, minSize=(24,24), gri görüntü, tam çözünürlük |
| YOLO | conf=0.25, NMS iou=0.45, imgsz=1280, CPU |
| Parlaklık/kontrast | g = 1.3·f + 30, [0,255] aralığına kırpılır |
| Histogram eşitleme | elle yazıldı, YCrCb'nin Y kanalı |
| CLAHE | clipLimit=2.0, tileGridSize=8×8, LAB'ın L kanalı |
| Eşleştirme | IoU ≥ 0.50, bire bir, en yüksek IoU önce (greedy) |
| Süre | `cikarim_ms` = yalnızca dedektör çağrısı; `uctan_uca_ms` = iyileştirme + dedektör (dosya okuma hariç). YOLO ısındırıldıktan sonra ölçüldü. |

Rastgelelik içeren adım yoktur, sonuçlar deterministiktir. Süreler donanıma göre değişir.

## Notlar
- Windows'ta klasör yolunda Türkçe karakter varsa (ör. `diğ`) `cv2.imread` ve `CascadeClassifier(yol)` çalışmaz.
  Bu yüzden kod dosyaları `np.fromfile` ile ve belleğe okuyarak yükler.
- Kendi histogram eşitlememiz `cv2.equalizeHist` ile karşılaştırıldı ve fark 0 çıktı (`sonuclar/parametreler.json`).
- Yapay zekâ kullanımı: kodun önemli bir kısmı yapay zekâ asistanı (Claude) desteğiyle yazıldı ve grup tarafından incelendi. Ayrıntı raporda.
- Etik: fotoğraflar grup üyelerinin onayıyla çekildi. Yüz tanıma veya kimlik etiketleme yapılmadı. Fotoğraflar yayımlanmamalıdır.
