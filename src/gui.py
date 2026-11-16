"""
GRAFİK ARAYÜZ - demo.py'nin pencereli sürümü.

  python src/gui.py

- "Görüntü Aç" ile bir fotoğraf seçin, dedektör / sürüm / parametreleri ayarlayıp "Çalıştır"a basın.
- Görüntü data/images altındaysa ve etiketi varsa TP/FP/FN de hesaplanır.
- "Kamerayı Başlat" ile web kamerasında canlı tespit yapılır.
- "Sonucu Kaydet" ızgarayı results/demo/ içine kaydeder.
Ağır işler (model yükleme, tespit) arka planda çalışır; pencere donmaz.
"""
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import config as C
import enhance as E
import detectors as D
import evaluate as EV
from dataset import load_gt, label_path
from plots import make_grid, draw, DET_NAMES

DET_LABELS = {"haar": "Haar Cascade", "hog": "Dlib HOG", "yolo": "YOLOv11n-face"}

# (anahtar, etiket, tip, varsayılan, dedektör, dedektör argümanı)
PARAMS = [
    ("scaleFactor", "Haar scaleFactor", float, C.HAAR_SCALE_FACTOR, "haar", "scale_factor"),
    ("minNeighbors", "Haar minNeighbors", int, C.HAAR_MIN_NEIGHBORS, "haar", "min_neighbors"),
    ("upsample", "HOG upsample", int, C.HOG_UPSAMPLE, "hog", "upsample"),
    ("conf", "YOLO conf", float, C.YOLO_CONF, "yolo", "conf"),
    ("imgsz", "YOLO imgsz", int, C.YOLO_IMGSZ, "yolo", "imgsz"),
]


def prepare(img):
    s = C.MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def imread_unicode(path):
    """cv2.imread Windows'ta Türkçe karakterli yollarda başarısız olabilir."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yüz Tespiti - Haar / HOG / YOLO")
        icon = C.ROOT / "assets" / "app.ico"
        if icon.exists():
            try:
                self.iconbitmap(default=str(icon))
            except tk.TclError:
                pass
        self.geometry("1400x860")
        self.minsize(1000, 640)

        self.q = queue.Queue()          # arka plan -> arayüz mesajları
        self.det_cache = {}             # (ad, parametreler) -> dedektör
        self.det_lock = threading.Lock()
        self.busy = False
        self.img_path = None
        self.result_bgr = None          # gösterilen son görüntü
        self.tk_img = None
        self.cam_running = False

        self._build()
        self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------- Arayüz ----------------
    def _build(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Kaynak
        src = ttk.LabelFrame(left, text="Kaynak", padding=8)
        src.pack(fill=tk.X)
        ttk.Button(src, text="Görüntü Aç…", command=self.open_image).pack(fill=tk.X)
        self.lbl_file = ttk.Label(src, text="(görüntü seçilmedi)", wraplength=230, foreground="#555")
        self.lbl_file.pack(fill=tk.X, pady=(4, 6))
        self.btn_cam = ttk.Button(src, text="Kamerayı Başlat", command=self.toggle_camera)
        self.btn_cam.pack(fill=tk.X)
        row = ttk.Frame(src)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Kamera sürümü:").pack(side=tk.LEFT)
        self.cam_variant = tk.StringVar(value="raw")
        self.cam_variant_now = "raw"   # iş parçacığı bunu okur (Tk değişkenini değil)
        self.cam_variant.trace_add("write", lambda *_: setattr(self, "cam_variant_now", self.cam_variant.get()))
        ttk.Combobox(row, textvariable=self.cam_variant, values=E.VARIANTS, width=8,
                     state="readonly").pack(side=tk.RIGHT)

        # Dedektörler
        df = ttk.LabelFrame(left, text="Dedektörler", padding=8)
        df.pack(fill=tk.X, pady=(10, 0))
        self.det_vars = {}
        for n in D.DETECTORS:
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(df, text=DET_LABELS[n], variable=v).pack(anchor=tk.W)
            self.det_vars[n] = v

        # Sürümler
        vf = ttk.LabelFrame(left, text="İyileştirme sürümleri", padding=8)
        vf.pack(fill=tk.X, pady=(10, 0))
        self.var_vars = {}
        for n in E.VARIANTS:
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(vf, text=f"{n} - {E.VARIANT_NAMES[n]}", variable=v).pack(anchor=tk.W)
            self.var_vars[n] = v

        # Parametreler
        pf = ttk.LabelFrame(left, text="Parametreler", padding=8)
        pf.pack(fill=tk.X, pady=(10, 0))
        self.param_vars = {}
        for i, (key, label, _, default, _, _) in enumerate(PARAMS):
            ttk.Label(pf, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            v = tk.StringVar(value=str(default))
            ttk.Entry(pf, textvariable=v, width=8).grid(row=i, column=1, sticky=tk.E, padx=(8, 0))
            self.param_vars[key] = v
        ttk.Button(pf, text="Varsayılana dön", command=self.reset_params).grid(
            row=len(PARAMS), column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))

        # Eylemler
        self.btn_run = ttk.Button(left, text="▶  Çalıştır", style="Accent.TButton", command=self.run)
        self.btn_run.pack(fill=tk.X, pady=(14, 4), ipady=4)
        ttk.Button(left, text="Sonucu Kaydet…", command=self.save).pack(fill=tk.X)

        # Sağ: görüntü + tablo
        self.canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render())

        cols = ("det", "var", "n", "ms", "tp", "fp", "fn", "err")
        heads = ("Dedektör", "Sürüm", "Yüz", "Süre (ms)", "TP", "FP", "FN", "|Sayma hatası|")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=7)
        for c, h in zip(cols, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=110 if c in ("det", "var") else 80, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(8, 0))

        self.status = tk.StringVar(value="Hazır. Bir görüntü açın veya kamerayı başlatın.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W,
                  padding=(8, 2)).pack(side=tk.BOTTOM, fill=tk.X, before=left)

    # ---------------- Yardımcılar ----------------
    def reset_params(self):
        for key, _, _, default, _, _ in PARAMS:
            self.param_vars[key].set(str(default))

    def read_params(self):
        """Her dedektör için argüman sözlüğü; hatalı girişte ValueError."""
        out = {n: {} for n in D.DETECTORS}
        for key, label, typ, _, det, arg in PARAMS:
            txt = self.param_vars[key].get().strip().replace(",", ".")
            try:
                out[det][arg] = typ(txt)
            except ValueError:
                raise ValueError(f"'{label}' geçersiz: {txt!r}")
        return out

    def selected(self):
        dets = [n for n, v in self.det_vars.items() if v.get()]
        vars_ = [n for n, v in self.var_vars.items() if v.get()]
        return dets, vars_

    def get_detector(self, name, kw):
        key = (name, tuple(sorted(kw.items())))
        with self.det_lock:
            if key not in self.det_cache:
                self.q.put(("status", f"{DET_LABELS[name]} yükleniyor…"))
                self.det_cache[key] = D.build(name, **kw)
            return self.det_cache[key]

    def set_busy(self, busy):
        self.busy = busy
        self.btn_run.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.config(cursor="watch" if busy else "")

    def show(self, bgr):
        self.result_bgr = bgr
        self._render()

    def _render(self):
        self.canvas.delete("all")
        if self.result_bgr is None:
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                                    text="Sonuç burada görünecek", fill="#999", font=("Segoe UI", 14))
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = self.result_bgr.shape[:2]
        s = min(cw / w, ch / h)
        img = cv2.resize(self.result_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)

    def _poll(self):
        new_img = None
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "status":
                    self.status.set(data)
                elif kind == "row":
                    self.tree.insert("", tk.END, values=data)
                elif kind == "image":
                    new_img = data   # birden çok kare biriktiyse yalnızca sonuncusu çizilir
                elif kind == "done":
                    self.set_busy(False)
                    self.status.set(data)
                elif kind == "error":
                    self.set_busy(False)
                    self.status.set("Hata: " + data)
                    messagebox.showerror("Hata", data)
                elif kind == "cam_stopped":
                    self.cam_running = False
                    self.btn_cam.config(text="Kamerayı Başlat")
                    self.status.set(data)
        except queue.Empty:
            pass
        if new_img is not None:
            self.show(new_img)
        self.after(30, self._poll)

    # ---------------- Eylemler ----------------
    def open_image(self):
        p = filedialog.askopenfilename(
            title="Görüntü seç", initialdir=str(C.IMG_DIR if C.IMG_DIR.exists() else C.ROOT),
            filetypes=[("Görüntüler", "*.jpg *.jpeg *.png *.bmp"), ("Tümü", "*.*")])
        if not p:
            return
        img = imread_unicode(p)
        if img is None:
            messagebox.showerror("Hata", f"Görüntü okunamadı:\n{p}")
            return
        self.img_path = Path(p)
        has_gt = label_path(self.img_path).exists()
        self.lbl_file.config(text=self.img_path.name + ("  (etiketli)" if has_gt else ""))
        self.tree.delete(*self.tree.get_children())
        self.show(prepare(img))
        self.status.set(f"{self.img_path.name} yüklendi. Çalıştır'a basın.")

    def run(self):
        if self.busy:
            return
        if self.cam_running:
            messagebox.showinfo("Bilgi", "Önce kamerayı durdurun.")
            return
        if self.img_path is None:
            messagebox.showinfo("Bilgi", "Önce bir görüntü açın.")
            return
        dets, vars_ = self.selected()
        if not dets or not vars_:
            messagebox.showinfo("Bilgi", "En az bir dedektör ve bir sürüm seçin.")
            return
        try:
            params = self.read_params()
        except ValueError as e:
            messagebox.showerror("Parametre hatası", str(e))
            return
        self.tree.delete(*self.tree.get_children())
        self.set_busy(True)
        threading.Thread(target=self._run_worker, args=(self.img_path, dets, vars_, params),
                         daemon=True).start()

    def _run_worker(self, path, dets, vars_, params):
        try:
            img = prepare(imread_unicode(path))
            gt = load_gt(path, img.shape[1], img.shape[0]) if label_path(path).exists() else None
            enhanced = {(path.name, v): (E.apply(img, v), 0) for v in vars_}
            store = {}
            for n in dets:
                d = self.get_detector(n, params[n])
                self.q.put(("status", f"{DET_LABELS[n]} çalışıyor…"))
                d.detect(img)  # ısınma
                for v in vars_:
                    e = enhanced[(path.name, v)][0]
                    t0 = time.perf_counter()
                    preds = d.detect(e)
                    ms = (time.perf_counter() - t0) * 1000
                    m = EV.image_metrics(preds, gt) if gt is not None else None
                    store[(path.name, n, v)] = (preds, m)
                    self.q.put(("row", (DET_NAMES[n], v, len(preds), f"{ms:.1f}",
                                        *( (m["tp"], m["fp"], m["fn"], m["abs_count_err"]) if m
                                           else ("-", "-", "-", "-") ))))
            grid = make_grid(path.name, gt, enhanced, store, dets, vars_)
            self.q.put(("image", grid))
            gt_txt = f"  GT={len(gt)} yüz" if gt is not None else "  (etiket yok)"
            self.q.put(("done", f"Tamamlandı: {path.name}{gt_txt}"))
        except Exception as e:
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    def save(self):
        if self.result_bgr is None:
            messagebox.showinfo("Bilgi", "Kaydedilecek sonuç yok.")
            return
        out = C.RESULTS_DIR / "demo"
        out.mkdir(parents=True, exist_ok=True)
        stem = self.img_path.stem if self.img_path else "kamera"
        p = filedialog.asksaveasfilename(initialdir=str(out), initialfile=f"{stem}_gui.jpg",
                                         defaultextension=".jpg",
                                         filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not p:
            return
        ok, buf = cv2.imencode(Path(p).suffix or ".jpg", self.result_bgr)
        if ok:
            buf.tofile(p)
            self.status.set(f"Kaydedildi: {p}")

    # ---------------- Kamera ----------------
    def toggle_camera(self):
        if self.cam_running:
            self.cam_running = False   # iş parçacığı kendini kapatır
            return
        if self.busy:
            return
        dets, _ = self.selected()
        if not dets:
            messagebox.showinfo("Bilgi", "En az bir dedektör seçin.")
            return
        try:
            params = self.read_params()
        except ValueError as e:
            messagebox.showerror("Parametre hatası", str(e))
            return
        self.cam_running = True
        self.btn_cam.config(text="Kamerayı Durdur")
        self.tree.delete(*self.tree.get_children())
        threading.Thread(target=self._cam_worker, args=(dets, params), daemon=True).start()

    def _cam_worker(self, dets, params):
        cap = None
        try:
            models = {n: self.get_detector(n, params[n]) for n in dets}
            self.q.put(("status", "Kamera açılıyor…"))
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("Kamera açılamadı.")
            self.q.put(("status", "Kamera çalışıyor. Sürümü sol üstteki listeden değiştirebilirsiniz."))
            while self.cam_running:
                ok, frame = cap.read()
                if not ok:
                    break
                variant = self.cam_variant_now
                e = E.apply(frame, variant)
                panels = []
                for n, d in models.items():
                    t0 = time.perf_counter()
                    preds = d.detect(e)
                    ms = (time.perf_counter() - t0) * 1000
                    c = draw(e, None, preds, None, 1.0)
                    cv2.rectangle(c, (0, 0), (c.shape[1], 40), (255, 255, 255), -1)
                    cv2.putText(c, f"{DET_NAMES[n]} | {variant} | {len(preds)} yuz | {ms:.0f} ms",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
                    panels.append(c)
                self.q.put(("image", np.hstack(panels)))
                time.sleep(0.005)
            self.q.put(("cam_stopped", "Kamera durduruldu."))
        except Exception as e:
            self.q.put(("cam_stopped", f"Kamera hatası: {e}"))
        finally:
            if cap is not None:
                cap.release()

    def _close(self):
        self.cam_running = False
        self.after(150, self.destroy)


if __name__ == "__main__":
    try:  # görev çubuğunda python simgesi yerine uygulamanın kendi simgesi görünsün
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Proje1.YuzTespiti")
    except Exception:
        pass
    App().mainloop()
