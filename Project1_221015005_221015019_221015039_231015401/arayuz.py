"""Face Detection – Haar Cascade vs. YOLO comparison GUI (Tkinter).

Usage:  python arayuz.py

  1) Select a photo (from the dataset or from the computer)
  2) Select the enhancement method
  3) Press "DETECT"  ->  the results of both detectors appear side by side

Colors: GREEN = true face (label), BLUE = correct detection, RED = false detection
"""
import csv
import os
import sys

# Without a console (pythonw / shortcut) there is no stdout/stderr; keep library log calls from crashing.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import dedektorler as D   # noqa: E402
import deney               # noqa: E402
import iyilestirme as I   # noqa: E402
import metrikler as M     # noqa: E402
import veri as V          # noqa: E402

PANEL_W, PANEL_H = 560, 520
FONT = ("Segoe UI", 10)
FONT_B = ("Segoe UI", 11, "bold")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Detection – Haar Cascade vs. YOLO Comparison")
        self.configure(bg="white")
        self.minsize(1180, 820)

        self.labels = V.read_labels()
        self.records = V.read_conditions()
        self.selected_path = None           # photo opened from the computer
        self.haar = D.HaarDetector()
        self.yolo = None                 # loaded in the background
        self._photos = {}               # Tk image references
        self._read_summary()                 # full-dataset scores (if available)
        self._blank = ImageTk.PhotoImage(Image.new("RGB", (PANEL_W, PANEL_H), (244, 244, 244)))

        self._build_ui()
        threading.Thread(target=self._load_yolo, daemon=True).start()

    # ------------------------------------------------------------ GUI
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="white", font=FONT)
        style.configure("TFrame", background="white")
        style.configure("TLabelframe", background="white")
        style.configure("TLabelframe.Label", background="white", font=FONT_B)
        # Main button (DETECT): solid blue
        style.configure("Buyuk.TButton", font=("Segoe UI", 12, "bold"), padding=(22, 9),
                       foreground="white", background="#2a78d6", bordercolor="#1c5cab",
                       lightcolor="#2a78d6", darkcolor="#2a78d6", focuscolor="#2a78d6")
        style.map("Buyuk.TButton",
                 background=[("disabled", "#9ec5f4"), ("active", "#1c5cab")],
                 lightcolor=[("active", "#1c5cab")], darkcolor=[("active", "#1c5cab")])
        # All other buttons: uniform, light blue background + blue text
        style.configure("Ikincil.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 7),
                       foreground="#1c5cab", background="#eef4fc", bordercolor="#86b6ef",
                       lightcolor="#eef4fc", darkcolor="#eef4fc", focuscolor="#eef4fc")
        style.map("Ikincil.TButton",
                 background=[("active", "#d8e7fa")],
                 lightcolor=[("active", "#d8e7fa")], darkcolor=[("active", "#d8e7fa")])

        # ---- Scrollable area: on small screens the bottom part is not squeezed; scroll to see it
        canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(canvas)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(body_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        # ---- Top: 3 steps
        top = ttk.Frame(body, padding=(14, 12, 14, 4))
        top.pack(fill="x")

        ttk.Label(top, text="1) Photo:", font=FONT_B).grid(row=0, column=0, sticky="w")
        self.cb_photo = ttk.Combobox(top, state="readonly", width=46, font=FONT, values=[
            f"{int(k['no']):02d}  –  {V.DISTANCE_NAMES[k['mesafe']]} – {V.CONDITION_NAMES[k['kosul']]}"
            f"  –  {k['gercek_yuz_sayisi']} faces"
            for k in self.records])
        self.cb_photo.current(0)
        self.cb_photo.grid(row=0, column=1, padx=6)
        self.cb_photo.bind("<<ComboboxSelected>>", lambda e: self._select_from_dataset())
        ttk.Button(top, text="Open from computer…", style="Ikincil.TButton",
                   command=self._open_file).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="2) Enhancement:", font=FONT_B).grid(row=0, column=3, sticky="w", padx=(18, 0))
        self.cb_method = ttk.Combobox(top, state="readonly", width=22, font=FONT,
                                      values=[v[0] for v in I.METHODS.values()])
        self.cb_method.current(0)
        self.cb_method.grid(row=0, column=4, padx=6)
        self.cb_method.bind("<<ComboboxSelected>>", lambda e: self._clear())

        self.btn_detect = ttk.Button(top, text="3)  ▶  DETECT", style="Buyuk.TButton",
                                     command=self._detect)
        self.btn_detect.grid(row=0, column=5, padx=(14, 0))

        self.lbl_selected = ttk.Label(top, text="", foreground="#666666")
        self.lbl_selected.grid(row=1, column=1, columnspan=4, sticky="w", padx=6, pady=(4, 0))

        # ---- Color legend
        legend_frame = ttk.Frame(body, padding=(14, 0))
        legend_frame.pack(fill="x")
        for color, msg in [("#00c800", "True face (label)"), ("#0078ff", "Correct detection"),
                            ("#ff0000", "False detection")]:
            tk.Label(legend_frame, text="■", fg=color, bg="white", font=("Segoe UI", 14)).pack(side="left")
            ttk.Label(legend_frame, text=msg + "     ").pack(side="left")

        # ---- Middle: two panels side by side
        middle = ttk.Frame(body, padding=(14, 6))
        middle.pack(fill="both", expand=True)
        self.panels = {}
        for col, name in enumerate(["Haar Cascade (classic)", "YOLOv8n-face"]):
            box = ttk.LabelFrame(middle, text=name, padding=8)
            box.grid(row=0, column=col, sticky="nsew", padx=6)
            middle.columnconfigure(col, weight=1)
            image_lbl = tk.Label(box, bg="#f4f4f4", image=self._blank)
            image_lbl.pack()
            result_lbl = tk.Label(box, text="", bg="white", font=("Segoe UI", 11), justify="left", anchor="w")
            result_lbl.pack(fill="x", pady=(8, 0))
            overall = tk.Label(box, text="", bg="#f4f7fb", fg="#333333", font=("Segoe UI", 10),
                             justify="left", anchor="w", padx=6, pady=4)
            overall.pack(fill="x", pady=(6, 0))
            self.panels[col] = (image_lbl, result_lbl, overall)
        middle.rowconfigure(0, weight=1)

        # ---- Bottom: settings
        bottom = ttk.Frame(body, padding=(14, 4, 14, 10))
        bottom.pack(fill="x")
        settings = ttk.LabelFrame(bottom, text="Settings (fixed in the experiment; can be changed in the defense)", padding=6)
        settings.pack(side="left")
        ttk.Label(settings, text="Haar minNeighbors:").pack(side="left")
        self.sp_haar = ttk.Spinbox(settings, from_=1, to=20, width=4, font=FONT)
        self.sp_haar.set(D.HAAR_PARAM["minNeighbors"])
        self.sp_haar.pack(side="left", padx=(2, 14))
        ttk.Label(settings, text="YOLO confidence threshold:").pack(side="left")
        self.sp_yolo = ttk.Spinbox(settings, from_=0.05, to=0.95, increment=0.05, width=5, font=FONT)
        self.sp_yolo.set(D.YOLO_PARAM["conf"])
        self.sp_yolo.pack(side="left", padx=(2, 14))
        ttk.Button(settings, text="Default", style="Ikincil.TButton",
                   command=self._defaults).pack(side="left")

        self.status = ttk.Label(body, text="Loading YOLO model…", foreground="#666666", padding=(14, 0, 0, 6))
        self.status.pack(fill="x")

        self._select_from_dataset()

    # ------------------------------------------------------------ helpers
    def _load_yolo(self):
        try:
            self.yolo = D.YoloDetector()
            self.yolo.detect(np.zeros((64, 64, 3), np.uint8))   # warm-up (the first call is slow)
            self.after(0, lambda: self.status.config(text="Ready. Select a photo and a method, then press DETECT."))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("YOLO could not be loaded", str(e)))

    def _read_summary(self):
        """If results/ozet.csv exists: (detector, method) -> full-dataset scores."""
        path = os.path.join(deney.RESULTS_DIR, "ozet.csv")
        self.summary = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    self.summary[(r["dedektor"], r["yontem"])] = r

    def _clear(self):
        """When the photo/method changes, clear the old result from the screen (avoid confusion)."""
        for image_lbl, result_lbl, overall in self.panels.values():
            image_lbl.config(image=self._blank)
            result_lbl.config(text="")
            overall.config(text="")
        self._photos.clear()
        if self.yolo is not None:
            self.status.config(text="Selection changed. Press DETECT to see the result.")

    def _select_from_dataset(self):
        self.selected_path = None
        k = self.records[self.cb_photo.current()]
        note = f"   ({k['not']})" if k["not"] else ""
        self.lbl_selected.config(text=f"Selected: dataset photo {int(k['no']):02d}{note}")
        self._clear()

    def _open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.selected_path = path
            self.lbl_selected.config(text=f"Selected: {os.path.basename(path)}   (no labels → only the count is shown)")
            self._clear()

    def _defaults(self):
        self.sp_haar.set(D.HAAR_PARAM["minNeighbors"])
        self.sp_yolo.set(D.YOLO_PARAM["conf"])

    def _get_params(self):
        try:
            self.haar.param["minNeighbors"] = int(float(self.sp_haar.get()))
            if self.yolo:
                self.yolo.param["conf"] = float(self.sp_yolo.get())
            return True
        except ValueError:
            messagebox.showwarning("Settings error", "Settings values must be numbers.")
            return False

    # ------------------------------------------------------------ detection
    def _detect(self):
        if self.yolo is None:
            messagebox.showinfo("Please wait", "The YOLO model is still loading, try again in a few seconds.")
            return
        if not self._get_params():
            return
        if self.selected_path:
            path, truth = self.selected_path, None
        else:
            k = self.records[self.cb_photo.current()]
            path, truth = os.path.join(V.IMAGE_DIR, k["dosya"]), self.labels[k["dosya"]]
        method = list(I.METHODS)[self.cb_method.current()]

        self.btn_detect.state(["disabled"])
        self.status.config(text="Detecting…")
        threading.Thread(target=self._detect_worker, args=(path, truth, method), daemon=True).start()

    def _detect_worker(self, path, truth, method):
        try:
            img = V.read_image(path)
            if img is None:
                raise ValueError("Could not read image: " + path)
            img = I.apply_method(img, method)
            results = []
            for d in (self.haar, self.yolo):
                t0 = time.perf_counter()
                pred = d.detect(img)
                ms = (time.perf_counter() - t0) * 1000
                results.append((pred, ms))
            self.after(0, lambda: self._show(img, truth, results, method))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn_detect.state(["!disabled"]))

    def _show(self, img, truth, results, method):
        for col, (pred, ms) in enumerate(results):
            if truth is not None:
                matches, tp, fp, fn = M.match(pred, truth)
                drawn = deney.draw_boxes(img, truth, pred, matches)
                count_error = abs(len(pred) - len(truth))
                p, r, f1 = M.summary(tp, fp, fn)
                mean_iou = sum(v for _, _, v in matches) / len(matches) if matches else 0.0
                msg = (f"THIS PHOTO:  Detected faces: {len(pred)}  /  True faces: {len(truth)}"
                         f"   (count error: {count_error})\n"
                         f"Correct (TP): {tp}    False (FP): {fp}    Missed (FN): {fn}\n"
                         f"Precision: {p:.2f}    Recall: {r:.2f}    F1: {f1:.2f}    Mean IoU: {mean_iou:.2f}\n"
                         f"Time: {ms:.0f} ms")
            else:
                drawn = deney.draw_boxes(img, [], pred, [(i, 0, 1) for i in range(len(pred))])
                msg = (f"THIS PHOTO:  Detected faces: {len(pred)}\nTime: {ms:.0f} ms\n"
                         "(No labels, so TP/FP/FN, Precision, Recall and F1 cannot be computed.)")
            image_lbl, result_lbl, overall = self.panels[col]
            self._set_image(image_lbl, drawn, col)
            result_lbl.config(text=msg)
            overall.config(text=self._overall_text(("Haar Cascade", "YOLOv8n-face")[col], method))
        self.status.config(text="Done. You can select another photo or method and try again.")

    def _overall_text(self, detector, method):
        """Scores of the selected method on the full dataset (28 photos) – from results/ozet.csv."""
        s = self.summary.get((detector, method))
        if not s:
            return "FULL DATASET: not computed yet → run python src/deney.py."
        return (f"FULL DATASET (28 photos, {I.METHODS[method][0]}):\n"
                f"Precision: {float(s['precision']):.3f}    Recall: {float(s['recall']):.3f}    "
                f"F1: {float(s['F1']):.3f}    MAE: {float(s['MAE']):.2f}    "
                f"Mean time: {float(s['ort_cikarim_ms']):.0f} ms")

    def _set_image(self, label, img_bgr, slot):
        h, w = img_bgr.shape[:2]
        scale = min(PANEL_W / w, PANEL_H / h)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self._photos[slot] = photo
        label.config(image=photo, width=PANEL_W, height=PANEL_H)


if __name__ == "__main__":
    App().mainloop()
