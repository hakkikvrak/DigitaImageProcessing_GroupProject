# Face Detection and People Counting under Challenging Imaging Conditions: Haar Cascade vs. YOLO

This project compares two face detectors on the same 28 photos:
the classic **Haar Cascade (Viola–Jones)** and the face-trained **YOLOv8n-face**. Each photo
is tested in 4 versions: raw, brightness/contrast, histogram equalization and CLAHE.

## Quick start

```bash
pip install -r requirements.txt
python arayuz.py          # GUI (live demo)
python src/deney.py       # full experiment: tables and charts -> sonuclar/
```

### How to use the GUI
1. **Photo ("Fotoğraf"):** pick a photo from the list. Each entry shows the number, condition and true face count.
   For a new, unlabeled photo use the **Bilgisayardan aç…** ("Open from computer") button.
2. **Enhancement ("İyileştirme"):** choose Raw / Brightness-Contrast / Histogram Equalization / CLAHE.
3. Press **▶ TESPİT ET** ("Detect"). Haar is shown on the left, YOLO on the right.
   - Green box = true face (label), blue = correct detection, red = false detection
   - Below: detected/true face count, correct, false, missed and time (ms)
- **Settings ("Ayarlar"):** Haar `minNeighbors` and the YOLO confidence threshold can be changed during the defense.
  The **Varsayılan** ("Default") button restores the fixed values used in the experiment.
- **Tüm deneyi çalıştır** ("Run full experiment"): 28 photos × 4 methods × 2 detectors. The results table appears in a window and the files are written to the `sonuclar/` folder.

## Folder structure

```
Proje1_No1_No2_No3/
├── arayuz.py                 Tkinter GUI
├── src/
│   ├── iyilestirme.py        brightness/contrast + histogram equalization (HAND-WRITTEN), CLAHE (OpenCV)
│   ├── dedektorler.py        HaarDetector, YoloDetector (fixed parameters are here)
│   ├── metrikler.py          IoU, one-to-one matching, P/R/F1
│   ├── veri.py               reading the label and condition files
│   └── deney.py              batch experiment, tables, charts
├── modeller/                 yolov8n-face.pt, haarcascade_frontalface_default.xml
├── veri/
│   ├── goruntuler/           28 original photos (Roboflow export)
│   ├── etiketler.csv         85 ground-truth face boxes (filename, xmin, ymin, xmax, ymax)
│   └── kosullar.csv          photo → condition (normal / dusuk_isik / asiri_parlak)
├── sonuclar/                 ozet.csv, kosul_ozet.csv, goruntu_bazli.csv, parametreler.json, grafikler/, ornekler/
├── akis_semasi.png
├── Rapor.docx
└── requirements.txt
```

## Pretrained models

| Detector | File | Version / source |
|---|---|---|
| Haar Cascade (Viola–Jones) | `haarcascade_frontalface_default.xml` | OpenCV 4.x: https://github.com/opencv/opencv/tree/4.x/data/haarcascades |
| YOLO face detector | `yolov8n-face.pt` (YOLOv8 nano, single class `face`, trained on WIDER FACE) | akanametov/yolo-face, version 1.0.0: https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt |

- `yolov8n-face.pt` SHA-256: `d545bf1add5aa736a4febac4f4f9245a6d596cd0fe70d5d57989fe0cb9e626ca`
- The generic YOLO "person" class was **not used**. The model outputs face boxes directly.
- Ultralytics was used for inference: https://github.com/ultralytics/ultralytics

## Fixed experiment settings
The values were not tuned on the test photos and were not changed during the experiment.

| Setting | Value |
|---|---|
| Haar | scaleFactor=1.1, minNeighbors=5, minSize=(24,24), grayscale image, full resolution |
| YOLO | conf=0.25, NMS iou=0.45, imgsz=1280, CPU |
| Brightness/contrast | g = 1.3·f + 30, clipped to [0,255] |
| Histogram equalization | hand-written, Y channel of YCrCb |
| CLAHE | clipLimit=2.0, tileGridSize=8×8, L channel of LAB |
| Matching | IoU ≥ 0.50, one-to-one, highest IoU first (greedy) |
| Time | `cikarim_ms` = detector call only; `uctan_uca_ms` = enhancement + detector (file reading excluded). Measured after warming up YOLO. |

No step involves randomness; the results are deterministic. Timings depend on the hardware.

## Notes
- On Windows, if the folder path contains Turkish characters (e.g. `diğ`), `cv2.imread` and `CascadeClassifier(path)` fail.
  For this reason the code loads files with `np.fromfile` and reads them into memory.
- Our own histogram equalization was compared with `cv2.equalizeHist` and the difference was 0 (`sonuclar/parametreler.json`).
- AI usage: part of the code was written with the help of an AI assistant (Claude) and reviewed by the group. Details are in the report.
- Ethics: the photos were taken with the consent of the group members. No face recognition or identity labeling was performed. The photos must not be published.
