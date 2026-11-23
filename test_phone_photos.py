"""Telefonla cekilen fotograflari test eder.

Kullanim: fotograflari phone_photos/ klasorune koyun, dosya adi gercek sinifla baslasin:
  phone_photos/cat_1.jpg, phone_photos/automobile_2.jpg, ...
  .venv/bin/python test_phone_photos.py
Cikti: outputs/phone_results.png ve outputs/phone_results.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from app import CLASSES, TR, classify

files = sorted(p for p in Path("phone_photos").iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"})
rows = []
import math
cols = min(3, len(files)); nrows = math.ceil(len(files) / cols)
fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 5.2 * nrows), squeeze=False)
for a in axes.flat[len(files):]:
    a.axis("off")
axes = axes.flat
for ax, f in zip(axes, files):
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    probs, _ = classify(img)
    pred_label, conf = max(probs.items(), key=lambda kv: kv[1])
    pred = pred_label.split(" ")[0]
    true = f.stem.split("_")[0].lower()
    ok = pred == true
    rows.append((f.name, true, pred, conf, ok))
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"Gerçek: {true}\nTahmin: {pred} (%{conf*100:.1f})",
                 color="green" if ok else "red", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/phone_results.png", dpi=110)

md = ["| Dosya | Gerçek | Tahmin | Güven | Sonuç |", "|---|---|---|---|---|"]
for name, t, p, c, ok in rows:
    md.append(f"| {name} | {t} ({TR.get(t, '?')}) | {p} ({TR[p]}) | %{c*100:.1f} | {'Doğru' if ok else 'Yanlış'} |")
md.append(f"\nDoğruluk: {sum(r[4] for r in rows)}/{len(rows)}")
Path("outputs/phone_results.md").write_text("\n".join(md))
print("\n".join(md))
