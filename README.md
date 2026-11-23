# Face Detection and People Counting under Challenging Imaging Conditions: Haar Cascade vs. YOLO

This project compares two face detectors on the same 28 photos:
the classic **Haar Cascade (Viola–Jones)** and the face-trained **YOLOv8n-face**. Each photo
is tested in 4 versions: raw, brightness/contrast, histogram equalization and CLAHE.

## Quick start

```bash
pip install -r requirements.txt
python gui.py             # GUI (live demo)
python src/experiment.py  # full experiment: tables and charts -> results/
```

On Windows, `start.bat` starts the GUI as well.

### How to use the GUI
1. **Photo:** pick a photo from the list. Each entry shows the number, distance (Far / Close),
   lighting (Dim / Normal / Bright) and true face count, e.g. `00 – Far – Normal – 3 faces`.
   For a new, unlabeled photo use the **Open from computer…** button.
2. **Enhancement:** choose Raw / Brightness-Contrast / Histogram Equalization / CLAHE.
3. Press **▶ DETECT**. Haar is shown on the left, YOLO on the right.
   - Green box = true face (label), blue = correct detection, red = false detection
   - Below: detected/true face count, correct, false, missed, Precision/Recall/F1 and time (ms) for this photo,
     plus the full-dataset scores of the selected method (from `results/summary.csv`)
- **Settings:** Haar `minNeighbors` and the YOLO confidence threshold can be changed during the defense.
  The **Default** button restores the fixed values used in the experiment.
- The full experiment (28 photos × 4 methods × 2 detectors) is run from the command line with
  `python src/experiment.py`; the files are written to the `results/` folder.

## Folder structure

```
Project1_221015005_221015019_221015039_231015401/
├── gui.py                    Tkinter GUI
├── src/
│   ├── enhancement.py        brightness/contrast + histogram equalization (HAND-WRITTEN), CLAHE (OpenCV)
│   ├── detectors.py          HaarDetector, YoloDetector (fixed parameters are here)
│   ├── metrics.py            IoU, one-to-one matching, P/R/F1
│   ├── dataset.py            reading the label and condition files
│   └── experiment.py         batch experiment, tables, charts
├── models/                   yolov8n-face.pt, haarcascade_frontalface_default.xml
├── data/
│   ├── images/               28 original photos (Roboflow export)
│   ├── labels.csv            85 ground-truth face boxes (filename, xmin, ymin, xmax, ymax)
│   └── conditions.csv        photo → lighting (dusuk_isik = Dim / normal / asiri_parlak = Bright),
│                             distance (uzak = Far / yakin = Close), true face count, note
├── results/                  summary.csv, condition_summary.csv, per_image.csv, image_quality*.csv,
│                             parameters.json, charts/, samples/
├── flow_chart.png
├── start.bat                 starts the GUI on Windows
├── Report.docx               project report (Word, in Turkish)
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
- Our own histogram equalization was compared with `cv2.equalizeHist` and the difference was 0 (`results/parameters.json`).
- AI usage: a significant part of the code was written with the help of an AI assistant (Claude) and reviewed by the group. Details are in the report.
- Ethics: the photos were taken with the consent of the group members. No face recognition or identity labeling was performed. The photos must not be published.
