"""Tüm deneyi çalıştırır: 2 dedektör x 4 görüntü sürümü x 28 görüntü.

Çıktılar (sonuclar/ klasörü):
  goruntu_bazli.csv     her görüntü / dedektör / yöntem için TP, FP, FN, sayma hatası, süre
  ozet.csv              dedektör x yöntem: TP, FP, FN, Precision, Recall, F1, MAE, süre
  kosul_ozet.csv        dedektör x yöntem x koşul (normal / düşük ışık / aşırı parlak)
  goruntu_kalitesi.csv  her yöntemin görüntüye etkisi (ortalama parlaklık, kontrast, entropi)
  parametreler.json     sabit parametreler ve sürümler
  grafikler/*.png       rapor için grafikler
  ornekler/*.jpg        kutuların çizildiği örnek görüntüler

Çalıştırma:  python src/deney.py
"""
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dedektorler as D           # noqa: E402
import iyilestirme as I           # noqa: E402
import metrikler as M             # noqa: E402
import veri as V                  # noqa: E402

KOK = V.KOK
SONUC = os.path.join(KOK, "sonuclar")
YONTEMLER = list(I.YONTEMLER)                         # ham, parlaklik_kontrast, ...
KOSULLAR = list(V.KOSUL_ADLARI)                       # normal, dusuk_isik, asiri_parlak

# Renkler: dedektör kimliği (kategorik, sabit sıra)
RENK = {"Haar Cascade": "#2a78d6", "YOLOv8n-face": "#eb6834"}
GT_RENK_BGR = (0, 200, 0)                             # referans kutular yeşil


def kutulari_ciz(img, gercek, tahmin, eslesmeler):
    """Yeşil = referans, mavi = doğru tespit (TP), kırmızı = yanlış tespit (FP)."""
    out = img.copy()
    kalin = max(2, img.shape[1] // 400)
    for g in gercek:
        cv2.rectangle(out, g[:2], g[2:4], GT_RENK_BGR, kalin)
    tp_idx = {i for i, _, _ in eslesmeler}
    for i, t in enumerate(tahmin):
        renk = (255, 120, 0) if i in tp_idx else (0, 0, 255)
        cv2.rectangle(out, t[:2], t[2:4], renk, kalin)
    return out


def entropi(gri):
    h = np.bincount(gri.ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def dogrulama_histogram():
    """Kendi histogram eşitlememizi cv2.equalizeHist ile karşılaştırır (sadece doğrulama)."""
    kayit = V.kosullari_oku()[0]
    gri = cv2.cvtColor(V.goruntu_oku(os.path.join(V.GORUNTU_KLASORU, kayit["dosya"])),
                       cv2.COLOR_BGR2GRAY)
    fark = np.abs(I.histogram_esitle_gri(gri).astype(int) - cv2.equalizeHist(gri).astype(int))
    return int(fark.max()), float(fark.mean())


def calistir(ilerleme=None, haar_param=None, yolo_param=None):
    """Deneyi çalıştırır. ilerleme(metin, oran) arayüzde ilerleme çubuğu için kullanılır."""
    os.makedirs(os.path.join(SONUC, "grafikler"), exist_ok=True)
    os.makedirs(os.path.join(SONUC, "ornekler"), exist_ok=True)

    etiketler = V.etiketleri_oku()
    kayitlar = V.kosullari_oku()
    dedektorler = [D.HaarDedektor(**(haar_param or {})), D.YoloDedektor(**(yolo_param or {}))]

    # YOLO ısınma (ilk çağrı model yükleme içerir; süreye katılmasın)
    dedektorler[1].tespit(np.zeros((640, 640, 3), np.uint8))

    satirlar, kalite = [], []
    toplam_is = len(kayitlar) * len(YONTEMLER)
    is_no = 0
    for k in kayitlar:
        ham = V.goruntu_oku(os.path.join(V.GORUNTU_KLASORU, k["dosya"]))
        gercek = etiketler[k["dosya"]]
        for y in YONTEMLER:
            is_no += 1
            if ilerleme:
                ilerleme(f"Görüntü {k['no']} / {I.YONTEMLER[y][0]}", is_no / toplam_is)
            t0 = time.perf_counter()
            img = I.uygula(ham, y)
            sure_iyilestirme = (time.perf_counter() - t0) * 1000

            gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kalite.append(dict(no=k["no"], kosul=k["kosul"], yontem=y,
                               parlaklik=float(gri.mean()), kontrast=float(gri.std()),
                               entropi=entropi(gri)))

            for d in dedektorler:
                t1 = time.perf_counter()
                tahmin = d.tespit(img)
                sure_cikarim = (time.perf_counter() - t1) * 1000
                esl, tp, fp, fn = M.eslestir(tahmin, gercek)
                satirlar.append(dict(
                    no=int(k["no"]), dosya=k["dosya"], kosul=k["kosul"], yontem=y, dedektor=d.ad,
                    gercek_sayi=len(gercek), tahmin_sayi=len(tahmin), tp=tp, fp=fp, fn=fn,
                    sayma_hatasi=abs(len(tahmin) - len(gercek)),
                    cikarim_ms=round(sure_cikarim, 2),
                    uctan_uca_ms=round(sure_iyilestirme + sure_cikarim, 2)))
                # Bazı örnek görüntüleri kaydet (her koşuldan biri)
                if int(k["no"]) in (4, 5, 6, 18, 25):
                    ad = f"{int(k['no']):02d}_{y}_{d.ad.split()[0].split('v')[0]}.jpg"
                    V.goruntu_yaz(os.path.join(SONUC, "ornekler", ad),
                                  cv2.resize(kutulari_ciz(img, gercek, tahmin, esl), None, fx=0.5, fy=0.5))

    yaz_csv(os.path.join(SONUC, "goruntu_bazli.csv"), satirlar)
    yaz_csv(os.path.join(SONUC, "goruntu_kalitesi.csv"), kalite)
    ozet = ozetle(satirlar, ["dedektor", "yontem"])
    kosul_ozet = ozetle(satirlar, ["dedektor", "yontem", "kosul"])
    yaz_csv(os.path.join(SONUC, "ozet.csv"), ozet)
    yaz_csv(os.path.join(SONUC, "kosul_ozet.csv"), kosul_ozet)
    kalite_ozet = kalite_ozetle(kalite)
    yaz_csv(os.path.join(SONUC, "goruntu_kalitesi_ozet.csv"), kalite_ozet)

    mx, ort = dogrulama_histogram()
    param = dict(
        haar=dict(dedektorler[0].param, model="haarcascade_frontalface_default.xml"),
        yolo=dict(dedektorler[1].param, model="yolov8n-face.pt (akanametov/yolo-face 1.0.0)", device="cpu"),
        iyilestirme=dict(parlaklik_kontrast=dict(alfa=I.PK_ALFA, beta=I.PK_BETA),
                         histogram_esitleme="elle yazıldı, YCrCb Y kanalı",
                         clahe=dict(clipLimit=I.CLAHE_CLIP, tileGridSize=I.CLAHE_IZGARA, kanal="LAB L")),
        iou_esigi=M.IOU_ESIK, eslestirme="bire bir, en yüksek IoU öncelikli (greedy)",
        sure_olcumu="cikarim_ms = sadece dedektör çağrısı; uctan_uca_ms = iyileştirme + dedektör "
                    "(dosya okuma hariç). YOLO bir kez ısındırıldıktan sonra ölçüldü.",
        kendi_HE_vs_cv2_equalizeHist=dict(max_fark=mx, ort_fark=round(ort, 4)),
        surumler=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                      ultralytics=__import__("ultralytics").__version__,
                      torch=__import__("torch").__version__, platform=platform.platform()),
    )
    with open(os.path.join(SONUC, "parametreler.json"), "w", encoding="utf-8") as f:
        json.dump(param, f, ensure_ascii=False, indent=2)

    grafikler(ozet, kosul_ozet, satirlar, kalite_ozet)
    if ilerleme:
        ilerleme("Bitti", 1.0)
    return ozet, kosul_ozet


def ozetle(satirlar, anahtarlar):
    grup = defaultdict(list)
    for s in satirlar:
        grup[tuple(s[a] for a in anahtarlar)].append(s)
    sira = {"dedektor": list(RENK), "yontem": YONTEMLER, "kosul": KOSULLAR}
    cikti = []
    for anahtar in sorted(grup, key=lambda t: [sira[a].index(v) for a, v in zip(anahtarlar, t)]):
        g = grup[anahtar]
        tp, fp, fn = (sum(s[x] for s in g) for x in ("tp", "fp", "fn"))
        p, r, f1 = M.ozet(tp, fp, fn)
        cikti.append({**dict(zip(anahtarlar, anahtar)),
                      "goruntu": len(g), "gercek_yuz": sum(s["gercek_sayi"] for s in g),
                      "tahmin": sum(s["tahmin_sayi"] for s in g),
                      "TP": tp, "FP": fp, "FN": fn,
                      "precision": round(p, 3), "recall": round(r, 3), "F1": round(f1, 3),
                      "MAE": round(np.mean([s["sayma_hatasi"] for s in g]), 3),
                      "ort_cikarim_ms": round(np.mean([s["cikarim_ms"] for s in g]), 1),
                      "ort_uctan_uca_ms": round(np.mean([s["uctan_uca_ms"] for s in g]), 1)})
    return cikti


def kalite_ozetle(kalite):
    cikti = []
    for kosul in KOSULLAR + ["tumu"]:
        for y in YONTEMLER:
            g = [q for q in kalite if q["yontem"] == y and (kosul == "tumu" or q["kosul"] == kosul)]
            cikti.append(dict(kosul=kosul, yontem=y,
                              ort_parlaklik=round(np.mean([q["parlaklik"] for q in g]), 1),
                              ort_kontrast=round(np.mean([q["kontrast"] for q in g]), 1),
                              ort_entropi=round(np.mean([q["entropi"] for q in g]), 3)))
    return cikti


def yaz_csv(yol, satirlar):
    with open(yol, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)


# ------------------------------------------------------------------ grafikler
def _eksen_sade(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors="#444444")
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def _cubuk(ax, degerler_dict, etiketler, bicim="{:.2f}"):
    n = len(degerler_dict)
    gen = 0.8 / n
    x = np.arange(len(etiketler))
    for i, (ad, degerler) in enumerate(degerler_dict.items()):
        konum = x - 0.4 + gen * (i + 0.5)
        bars = ax.bar(konum, degerler, gen * 0.92, color=RENK[ad], label=ad)
        for b, v in zip(bars, degerler):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), bicim.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(etiketler)
    _eksen_sade(ax)


def grafikler(ozet, kosul_ozet, satirlar, kalite_ozet):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yer = os.path.join(SONUC, "grafikler")
    y_ad = [I.YONTEMLER[y][0] for y in YONTEMLER]

    def deger(tablo, alan, **filtre):
        return {d: [next(s[alan] for s in tablo if s["dedektor"] == d and s["yontem"] == y
                         and all(s[k] == v for k, v in filtre.items())) for y in YONTEMLER]
                for d in RENK}

    # 1) F1 ve MAE - tüm veri
    for alan, baslik, dosya, bicim in [
            ("F1", "F1 skoru (IoU ≥ 0,50) – tüm veri kümesi", "f1_genel.png", "{:.2f}"),
            ("MAE", "Ortalama mutlak sayma hatası (MAE) – düşük daha iyi", "mae_genel.png", "{:.2f}"),
            ("ort_cikarim_ms", "Ortalama çıkarım süresi (ms / görüntü)", "sure.png", "{:.0f}")]:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        _cubuk(ax, deger(ozet, alan), y_ad, bicim)
        if alan == "F1":
            ax.set_ylim(0, 1.05)
        ax.set_title(baslik, loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(os.path.join(yer, dosya))
        plt.close(fig)

    # 2) Koşullara göre F1 (küçük çoklu grafik)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, kosul in zip(axs, KOSULLAR):
        _cubuk(ax, deger(kosul_ozet, "F1", kosul=kosul), ["Ham", "P/K", "HE", "CLAHE"])
        ax.set_title(V.KOSUL_ADLARI[kosul], fontsize=11)
        ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("F1")
    axs[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Çekim koşuluna göre F1 skoru", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(yer, "f1_kosullara_gore.png"))
    plt.close(fig)

    # 3) Recall koşullara göre
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, kosul in zip(axs, KOSULLAR):
        _cubuk(ax, deger(kosul_ozet, "recall", kosul=kosul), ["Ham", "P/K", "HE", "CLAHE"])
        ax.set_title(V.KOSUL_ADLARI[kosul], fontsize=11)
        ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("Recall")
    axs[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Çekim koşuluna göre Recall (bulunan yüz oranı)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(yer, "recall_kosullara_gore.png"))
    plt.close(fig)

    # 4) Araştırma sorusu: iyileştirme, ham görüntüye göre F1'i artırıyor mu?
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, kosul in zip(axs, KOSULLAR):
        f1 = deger(kosul_ozet, "F1", kosul=kosul)
        fark = {d: [v - f1[d][0] for v in f1[d][1:]] for d in RENK}   # ham'a göre fark
        _cubuk(ax, fark, ["P/K", "HE", "CLAHE"], "{:+.2f}")
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_title(V.KOSUL_ADLARI[kosul], fontsize=11)
    axs[0].set_ylabel("F1 farkı (iyileştirilmiş − ham)")
    axs[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("İyileştirme ham görüntüye göre F1'i ne kadar değiştirdi? (0'ın altı = kötüleşme)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(yer, "f1_degisimi.png"))
    plt.close(fig)


if __name__ == "__main__":
    def yaz(metin, oran):
        print(f"\r[{oran * 100:5.1f}%] {metin:<50}", end="", flush=True)

    ozet, _ = calistir(yaz)
    print("\n\nÖZET (tüm veri kümesi)")
    print(f"{'Dedektör':<14}{'Yöntem':<22}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>7}{'R':>7}{'F1':>7}{'MAE':>7}{'ms':>8}")
    for s in ozet:
        print(f"{s['dedektor']:<14}{I.YONTEMLER[s['yontem']][0]:<22}{s['TP']:>4}{s['FP']:>4}{s['FN']:>4}"
              f"{s['precision']:>7.3f}{s['recall']:>7.3f}{s['F1']:>7.3f}{s['MAE']:>7.2f}{s['ort_cikarim_ms']:>8.1f}")
    print("\nSonuçlar:", SONUC)
