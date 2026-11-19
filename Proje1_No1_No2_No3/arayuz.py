"""Yüz Tespiti – Haar Cascade ile YOLO karşılaştırma arayüzü (Tkinter).

Kullanım:  python arayuz.py

  1) Fotoğraf seç (veri kümesinden ya da bilgisayardan)
  2) İyileştirme yöntemini seç
  3) "TESPİT ET" butonuna bas  ->  iki dedektörün sonucu yan yana görünür

Renkler: YEŞİL = gerçek yüz (etiket), MAVİ = doğru tespit, KIRMIZI = yanlış tespit
"""
import csv
import os
import subprocess
import sys

# Konsolsuz çalışırken (pythonw / kısayol) stdout/stderr yoktur; kütüphanelerin log çağrıları çökmesin.
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
YAZI = ("Segoe UI", 10)
YAZI_B = ("Segoe UI", 11, "bold")


class Uygulama(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yüz Tespiti – Haar Cascade ve YOLO Karşılaştırması")
        self.configure(bg="white")
        self.minsize(1180, 820)

        self.etiketler = V.etiketleri_oku()
        self.kayitlar = V.kosullari_oku()
        self.secili_yol = None           # dışarıdan açılan fotoğraf
        self.haar = D.HaarDedektor()
        self.yolo = None                 # arka planda yüklenecek
        self._fotolar = {}               # Tk görüntü referansları
        self._ozet_oku()                 # tüm veri kümesi skorları (varsa)
        self._bos = ImageTk.PhotoImage(Image.new("RGB", (PANEL_W, PANEL_H), (244, 244, 244)))

        self._arayuz_kur()
        threading.Thread(target=self._yolo_yukle, daemon=True).start()

    # ------------------------------------------------------------ arayüz
    def _arayuz_kur(self):
        stil = ttk.Style(self)
        stil.theme_use("clam")
        stil.configure("TLabel", background="white", font=YAZI)
        stil.configure("TFrame", background="white")
        stil.configure("TLabelframe", background="white")
        stil.configure("TLabelframe.Label", background="white", font=YAZI_B)
        # Ana buton (TESPİT ET): dolu mavi
        stil.configure("Buyuk.TButton", font=("Segoe UI", 12, "bold"), padding=(22, 9),
                       foreground="white", background="#2a78d6", bordercolor="#1c5cab",
                       lightcolor="#2a78d6", darkcolor="#2a78d6", focuscolor="#2a78d6")
        stil.map("Buyuk.TButton",
                 background=[("disabled", "#9ec5f4"), ("active", "#1c5cab")],
                 lightcolor=[("active", "#1c5cab")], darkcolor=[("active", "#1c5cab")])
        # Diğer tüm butonlar: tek tip, açık mavi zemin + mavi yazı
        stil.configure("Ikincil.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 7),
                       foreground="#1c5cab", background="#eef4fc", bordercolor="#86b6ef",
                       lightcolor="#eef4fc", darkcolor="#eef4fc", focuscolor="#eef4fc")
        stil.map("Ikincil.TButton",
                 background=[("active", "#d8e7fa")],
                 lightcolor=[("active", "#d8e7fa")], darkcolor=[("active", "#d8e7fa")])

        # ---- Üst: 3 adım
        ust = ttk.Frame(self, padding=(14, 12, 14, 4))
        ust.pack(fill="x")

        ttk.Label(ust, text="1) Fotoğraf:", font=YAZI_B).grid(row=0, column=0, sticky="w")
        self.cb_foto = ttk.Combobox(ust, state="readonly", width=46, font=YAZI, values=[
            f"{int(k['no']):02d}  –  {V.KOSUL_ADLARI[k['kosul']]}  –  {k['gercek_yuz_sayisi']} yüz"
            for k in self.kayitlar])
        self.cb_foto.current(0)
        self.cb_foto.grid(row=0, column=1, padx=6)
        self.cb_foto.bind("<<ComboboxSelected>>", lambda e: self._veri_kumesinden_sec())
        ttk.Button(ust, text="Bilgisayardan aç…", style="Ikincil.TButton",
                   command=self._dosya_ac).grid(row=0, column=2, padx=4)

        ttk.Label(ust, text="2) İyileştirme:", font=YAZI_B).grid(row=0, column=3, sticky="w", padx=(18, 0))
        self.cb_yontem = ttk.Combobox(ust, state="readonly", width=22, font=YAZI,
                                      values=[v[0] for v in I.YONTEMLER.values()])
        self.cb_yontem.current(0)
        self.cb_yontem.grid(row=0, column=4, padx=6)
        self.cb_yontem.bind("<<ComboboxSelected>>", lambda e: self._temizle())

        self.btn_tespit = ttk.Button(ust, text="3)  ▶  TESPİT ET", style="Buyuk.TButton",
                                     command=self._tespit_et)
        self.btn_tespit.grid(row=0, column=5, padx=(14, 0))

        self.lbl_secili = ttk.Label(ust, text="", foreground="#666666")
        self.lbl_secili.grid(row=1, column=1, columnspan=4, sticky="w", padx=6, pady=(4, 0))

        # ---- Renk açıklaması
        aciklama = ttk.Frame(self, padding=(14, 0))
        aciklama.pack(fill="x")
        for renk, metin in [("#00c800", "Gerçek yüz (etiket)"), ("#0078ff", "Doğru tespit"),
                            ("#ff0000", "Yanlış tespit")]:
            tk.Label(aciklama, text="■", fg=renk, bg="white", font=("Segoe UI", 14)).pack(side="left")
            ttk.Label(aciklama, text=metin + "     ").pack(side="left")

        # ---- Orta: iki panel yan yana
        orta = ttk.Frame(self, padding=(14, 6))
        orta.pack(fill="both", expand=True)
        self.paneller = {}
        for sutun, ad in enumerate(["Haar Cascade (klasik)", "YOLOv8n-face"]):
            kutu = ttk.LabelFrame(orta, text=ad, padding=8)
            kutu.grid(row=0, column=sutun, sticky="nsew", padx=6)
            orta.columnconfigure(sutun, weight=1)
            resim = tk.Label(kutu, bg="#f4f4f4", image=self._bos)
            resim.pack()
            sonuc = tk.Label(kutu, text="", bg="white", font=("Segoe UI", 11), justify="left", anchor="w")
            sonuc.pack(fill="x", pady=(8, 0))
            genel = tk.Label(kutu, text="", bg="#f4f7fb", fg="#333333", font=("Segoe UI", 10),
                             justify="left", anchor="w", padx=6, pady=4)
            genel.pack(fill="x", pady=(6, 0))
            self.paneller[sutun] = (resim, sonuc, genel)
        orta.rowconfigure(0, weight=1)

        # ---- Alt: ayarlar + toplu deney
        alt = ttk.Frame(self, padding=(14, 4, 14, 10))
        alt.pack(fill="x")
        ayar = ttk.LabelFrame(alt, text="Ayarlar (deneyde sabit; savunmada değiştirilebilir)", padding=6)
        ayar.pack(side="left")
        ttk.Label(ayar, text="Haar minNeighbors:").pack(side="left")
        self.sp_haar = ttk.Spinbox(ayar, from_=1, to=20, width=4, font=YAZI)
        self.sp_haar.set(D.HAAR_PARAM["minNeighbors"])
        self.sp_haar.pack(side="left", padx=(2, 14))
        ttk.Label(ayar, text="YOLO güven eşiği:").pack(side="left")
        self.sp_yolo = ttk.Spinbox(ayar, from_=0.05, to=0.95, increment=0.05, width=5, font=YAZI)
        self.sp_yolo.set(D.YOLO_PARAM["conf"])
        self.sp_yolo.pack(side="left", padx=(2, 14))
        ttk.Button(ayar, text="Varsayılan", style="Ikincil.TButton",
                   command=self._varsayilan).pack(side="left")

        # Toplu işlemler: ayarların hemen yanında (sağ köşede Windows yazısının altında kalmasın)
        islem = ttk.LabelFrame(alt, text="Tüm veri kümesi", padding=6)
        islem.pack(side="left", padx=(16, 0))
        ttk.Button(islem, text="Tüm deneyi çalıştır", style="Ikincil.TButton",
                   command=self._toplu_deney).pack(side="left")
        ttk.Button(islem, text="Sonuç klasörünü aç", style="Ikincil.TButton",
                   command=self._klasor_ac).pack(side="left", padx=(8, 0))

        self.durum = ttk.Label(self, text="YOLO modeli yükleniyor…", foreground="#666666", padding=(14, 0, 0, 6))
        self.durum.pack(fill="x")

        self._veri_kumesinden_sec()

    # ------------------------------------------------------------ yardımcılar
    def _yolo_yukle(self):
        try:
            self.yolo = D.YoloDedektor()
            self.yolo.tespit(np.zeros((64, 64, 3), np.uint8))   # ısınma (ilk çağrı yavaştır)
            self.after(0, lambda: self.durum.config(text="Hazır. Fotoğraf ve yöntem seçip TESPİT ET'e basın."))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("YOLO yüklenemedi", str(e)))

    def _ozet_oku(self):
        """sonuclar/ozet.csv varsa: (dedektör, yöntem) -> tüm veri kümesi skorları."""
        yol = os.path.join(deney.SONUC, "ozet.csv")
        self.ozet = {}
        if os.path.exists(yol):
            with open(yol, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    self.ozet[(r["dedektor"], r["yontem"])] = r

    def _temizle(self):
        """Fotoğraf/yöntem değişince eski sonucu ekrandan kaldır (karışıklık olmasın)."""
        for resim, sonuc, genel in self.paneller.values():
            resim.config(image=self._bos)
            sonuc.config(text="")
            genel.config(text="")
        self._fotolar.clear()
        if self.yolo is not None:
            self.durum.config(text="Seçim değişti. Sonucu görmek için TESPİT ET'e basın.")

    def _veri_kumesinden_sec(self):
        self.secili_yol = None
        k = self.kayitlar[self.cb_foto.current()]
        not_ = f"   ({k['not']})" if k["not"] else ""
        self.lbl_secili.config(text=f"Seçili: veri kümesi fotoğrafı {int(k['no']):02d}{not_}")
        self._temizle()

    def _dosya_ac(self):
        yol = filedialog.askopenfilename(filetypes=[("Görüntü", "*.jpg *.jpeg *.png *.bmp")])
        if yol:
            self.secili_yol = yol
            self.lbl_secili.config(text=f"Seçili: {os.path.basename(yol)}   (etiket yok → sadece sayım gösterilir)")
            self._temizle()

    def _varsayilan(self):
        self.sp_haar.set(D.HAAR_PARAM["minNeighbors"])
        self.sp_yolo.set(D.YOLO_PARAM["conf"])

    def _parametreleri_al(self):
        try:
            self.haar.param["minNeighbors"] = int(float(self.sp_haar.get()))
            if self.yolo:
                self.yolo.param["conf"] = float(self.sp_yolo.get())
            return True
        except ValueError:
            messagebox.showwarning("Ayar hatası", "Ayar değerleri sayı olmalı.")
            return False

    def _klasor_ac(self):
        os.makedirs(deney.SONUC, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(deney.SONUC)
        else:
            subprocess.Popen(["xdg-open", deney.SONUC])

    # ------------------------------------------------------------ tespit
    def _tespit_et(self):
        if self.yolo is None:
            messagebox.showinfo("Bekleyin", "YOLO modeli hâlâ yükleniyor, birkaç saniye sonra tekrar deneyin.")
            return
        if not self._parametreleri_al():
            return
        if self.secili_yol:
            yol, gercek = self.secili_yol, None
        else:
            k = self.kayitlar[self.cb_foto.current()]
            yol, gercek = os.path.join(V.GORUNTU_KLASORU, k["dosya"]), self.etiketler[k["dosya"]]
        yontem = list(I.YONTEMLER)[self.cb_yontem.current()]

        self.btn_tespit.state(["disabled"])
        self.durum.config(text="Tespit yapılıyor…")
        threading.Thread(target=self._tespit_is, args=(yol, gercek, yontem), daemon=True).start()

    def _tespit_is(self, yol, gercek, yontem):
        try:
            img = V.goruntu_oku(yol)
            if img is None:
                raise ValueError("Görüntü okunamadı: " + yol)
            img = I.uygula(img, yontem)
            sonuclar = []
            for d in (self.haar, self.yolo):
                t0 = time.perf_counter()
                tahmin = d.tespit(img)
                ms = (time.perf_counter() - t0) * 1000
                sonuclar.append((tahmin, ms))
            self.after(0, lambda: self._goster(img, gercek, sonuclar, yontem))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Hata", str(e)))
        finally:
            self.after(0, lambda: self.btn_tespit.state(["!disabled"]))

    def _goster(self, img, gercek, sonuclar, yontem):
        for sutun, (tahmin, ms) in enumerate(sonuclar):
            if gercek is not None:
                esl, tp, fp, fn = M.eslestir(tahmin, gercek)
                ciz = deney.kutulari_ciz(img, gercek, tahmin, esl)
                hata = abs(len(tahmin) - len(gercek))
                p, r, f1 = M.ozet(tp, fp, fn)
                ort_iou = sum(v for _, _, v in esl) / len(esl) if esl else 0.0
                metin = (f"BU FOTOĞRAF:  Bulunan yüz: {len(tahmin)}  /  Gerçek yüz: {len(gercek)}"
                         f"   (sayma hatası: {hata})\n"
                         f"Doğru (TP): {tp}    Yanlış (FP): {fp}    Kaçırılan (FN): {fn}\n"
                         f"Precision: {p:.2f}    Recall: {r:.2f}    F1: {f1:.2f}    Ort. IoU: {ort_iou:.2f}\n"
                         f"Süre: {ms:.0f} ms")
            else:
                ciz = deney.kutulari_ciz(img, [], tahmin, [(i, 0, 1) for i in range(len(tahmin))])
                metin = (f"BU FOTOĞRAF:  Bulunan yüz: {len(tahmin)}\nSüre: {ms:.0f} ms\n"
                         "(Etiket olmadığı için TP/FP/FN, Precision, Recall ve F1 hesaplanamaz.)")
            resim, etiket, genel = self.paneller[sutun]
            self._resim_koy(resim, ciz, sutun)
            etiket.config(text=metin)
            genel.config(text=self._genel_metin(("Haar Cascade", "YOLOv8n-face")[sutun], yontem))
        self.durum.config(text="Bitti. Başka bir fotoğraf veya yöntem seçip tekrar deneyebilirsiniz.")

    def _genel_metin(self, dedektor, yontem):
        """Seçili yöntemin tüm veri kümesindeki (28 foto) skorları – sonuclar/ozet.csv'den."""
        s = self.ozet.get((dedektor, yontem))
        if not s:
            return "TÜM VERİ KÜMESİ: henüz hesaplanmadı → 'Tüm deneyi çalıştır' butonuna basın."
        return (f"TÜM VERİ KÜMESİ (28 foto, {I.YONTEMLER[yontem][0]}):\n"
                f"Precision: {float(s['precision']):.3f}    Recall: {float(s['recall']):.3f}    "
                f"F1: {float(s['F1']):.3f}    MAE: {float(s['MAE']):.2f}    "
                f"Ort. süre: {float(s['ort_cikarim_ms']):.0f} ms")

    def _resim_koy(self, label, img_bgr, anahtar):
        h, w = img_bgr.shape[:2]
        olcek = min(PANEL_W / w, PANEL_H / h)
        kucuk = cv2.resize(img_bgr, (int(w * olcek), int(h * olcek)), interpolation=cv2.INTER_AREA)
        foto = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(kucuk, cv2.COLOR_BGR2RGB)))
        self._fotolar[anahtar] = foto
        label.config(image=foto, width=PANEL_W, height=PANEL_H)

    # ------------------------------------------------------------ toplu deney
    def _toplu_deney(self):
        if self.yolo is None:
            messagebox.showinfo("Bekleyin", "YOLO modeli hâlâ yükleniyor.")
            return
        if not self._parametreleri_al():
            return
        pencere = tk.Toplevel(self)
        pencere.title("Tüm deney – sonuç tablosu")
        pencere.configure(bg="white")
        pencere.geometry("1000x520")
        ilerleme = ttk.Progressbar(pencere, length=960, maximum=1.0)
        ilerleme.pack(padx=20, pady=(16, 4))
        yazi = ttk.Label(pencere, text="Başlıyor… (birkaç dakika sürebilir)")
        yazi.pack()

        sutunlar = ["Dedektör", "Yöntem", "TP", "FP", "FN", "Precision", "Recall", "F1", "MAE", "ms/görüntü"]
        tablo = ttk.Treeview(pencere, columns=sutunlar, show="headings", height=10)
        for s in sutunlar:
            tablo.heading(s, text=s)
            tablo.column(s, width=150 if s in ("Dedektör", "Yöntem") else 80, anchor="center")
        tablo.pack(fill="both", expand=True, padx=20, pady=12)

        def guncelle(metin, oran):
            self.after(0, lambda: (ilerleme.config(value=oran), yazi.config(text=metin)))

        def is_():
            try:
                ozet, _ = deney.calistir(guncelle, haar_param=dict(self.haar.param),
                                         yolo_param=dict(self.yolo.param))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Hata", str(e)))
                return

            def doldur():
                for s in ozet:
                    tablo.insert("", "end", values=(
                        s["dedektor"], I.YONTEMLER[s["yontem"]][0], s["TP"], s["FP"], s["FN"],
                        f"{s['precision']:.3f}", f"{s['recall']:.3f}", f"{s['F1']:.3f}",
                        f"{s['MAE']:.2f}", f"{s['ort_cikarim_ms']:.0f}"))
                yazi.config(text="Bitti. Tablolar ve grafikler 'sonuclar' klasörüne kaydedildi.")
                self._ozet_oku()   # ana penceredeki "tüm veri kümesi" skorlarını güncelle
            self.after(0, doldur)

        threading.Thread(target=is_, daemon=True).start()


if __name__ == "__main__":
    Uygulama().mainloop()
