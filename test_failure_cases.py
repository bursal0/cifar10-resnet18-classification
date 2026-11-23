"""Basarisiz durum analizi: telefon fotograflarindan turetilmis zorlastirilmis varyasyonlar.

Kullanim: .venv/bin/python test_failure_cases.py
Cikti: outputs/failure_cases.png ve outputs/failure_cases.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from app import TR, classify


def load(name):
    return ImageOps.exif_transpose(Image.open(f"phone_photos/{name}.jpg")).convert("RGB")


car, cat1, cat2, cat3 = load("automobile_1"), load("cat_1"), load("cat_2"), load("cat_3")
W, H = car.size
# (aciklama, goruntu, beklenen sinif veya None = 10 sinifin hicbiri degil)
cases = [
    ("cat_1 180° ters çevrilmiş", cat1.rotate(180), "cat"),
    ("cat_3 180° ters çevrilmiş", cat3.rotate(180), "cat"),
    ("automobile_1 üst kısım:\naraçlar uzakta ve küçük", car.crop((0, 0, W, int(H * 0.27))), "automobile"),
    ("cat_2 kırpılmış: yalnızca kutu\n(10 sınıfın hiçbiri değil)", cat2.crop((0, 0, 340, 620)), None),
]

rows = ["| Varyasyon | Beklenen | Tahmin (güven) | 2. tahmin | Sonuç |", "|---|---|---|---|---|"]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flat
for ax, (desc, img, exp) in zip(axes, cases):
    probs, _ = classify(img)
    (l1, p1), (l2, p2) = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
    pred = l1.split(" ")[0]
    if exp is None:
        status, color = "Yanlış (kapsam dışı nesne, zorla bir sınıfa atandı)", "red"
    elif pred != exp:
        status, color = "Yanlış", "red"
    elif p1 < 0.5:
        status, color = "Kısmen (doğru ama düşük güven)", "darkorange"
    else:
        status, color = "Doğru", "green"
    exp_txt = f"{exp} ({TR[exp]})" if exp else "yok (sınıf dışı)"
    rows.append(f"| {desc.replace(chr(10), ' ')} | {exp_txt} | {l1} (%{p1*100:.1f}) | {l2} (%{p2*100:.1f}) | {status} |")
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{desc}\nTahmin: {pred} (%{p1*100:.1f})", color=color, fontsize=12)
plt.tight_layout()
plt.savefig("outputs/failure_cases.png", dpi=110)
Path("outputs/failure_cases.md").write_text("\n".join(rows))
print("\n".join(rows))
