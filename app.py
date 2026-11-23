"""Gradio arayuzu: kullanici fotograf yukler, model CIFAR-10 sinifini tahmin eder.

Calistirma: .venv/bin/python app.py   ->  http://127.0.0.1:7860
"""
import time
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
TR = {"airplane": "uçak", "automobile": "otomobil", "bird": "kuş", "cat": "kedi", "deer": "geyik",
      "dog": "köpek", "frog": "kurbağa", "horse": "at", "ship": "gemi", "truck": "kamyon"}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torchvision.models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("outputs/best_model.pt", map_location=device))
model.to(device).eval()
with torch.no_grad():  # GPU isinmasi: ilk tahminin yavas olmamasi icin
    model(torch.zeros(1, 3, 128, 128, device=device))

# Egitimdeki pipeline ile ayni: 32x32'ye kucult -> 128'e buyut -> ImageNet normalize
tf = transforms.Compose([
    transforms.Resize(32), transforms.CenterCrop(32), transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def classify(img):
    if img is None:
        return {}, ""
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    t = time.time()
    probs = model(x).softmax(1)[0].cpu()
    ms = (time.time() - t) * 1000
    return ({f"{c} ({TR[c]})": float(p) for c, p in zip(CLASSES, probs)},
            f"Tahmin süresi: {ms:.1f} ms")


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Fotoğraf yükle", height=340),
    outputs=[gr.Label(num_top_classes=5, label="Tahmin"), gr.Textbox(label="Süre")],
    title="CIFAR-10 Görüntü Sınıflandırma (ResNet18)",
    description="Uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi veya kamyon "
                "içeren bir fotoğraf yükleyin.",
    examples=sorted(str(p) for p in Path("phone_photos").glob("*.jpg")) or None,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
