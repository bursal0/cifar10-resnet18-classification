"""CIFAR-10 goruntu siniflandirma - ResNet18 (ImageNet on-egitimli) fine-tuning.

Calistirma:  .venv/bin/python train.py
Ciktilar:    outputs/ (model, metrikler, grafikler)
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

# ---------------- Ayarlar ----------------
SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 4
LR = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_SIZE = 5000

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def load_split(train):
    ds = torchvision.datasets.CIFAR10("data", train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,32,32
    y = torch.tensor(ds.targets)
    return x, y


def preprocess(xb, augment):
    """uint8 32x32 batch -> normalize edilmis IMG_SIZE batch (GPU uzerinde)."""
    xb = xb.to(device).float() / 255.0
    if augment:
        # random crop (padding=4) + yatay cevirme
        xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
        i, j = np.random.randint(0, 9, 2)
        xb = xb[:, :, i:i + 32, j:j + 32]
        flip = torch.rand(xb.size(0), device=device) < 0.5
        xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
    xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return (xb - MEAN) / STD


def build_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


@torch.no_grad()
def predict(model, x, bs=256):
    model.eval()
    preds, probs = [], []
    for k in range(0, len(x), bs):
        logits = model(preprocess(x[k:k + bs], augment=False))
        p = logits.softmax(1)
        probs.append(p.cpu())
        preds.append(p.argmax(1).cpu())
    return torch.cat(preds), torch.cat(probs)


def main():
    x_full, y_full = load_split(train=True)
    x_test, y_test = load_split(train=False)
    perm = torch.randperm(len(x_full))
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    x_tr, y_tr = x_full[tr_idx], y_full[tr_idx]
    x_val, y_val = x_full[val_idx], y_full[val_idx]
    print(f"train={len(x_tr)} val={len(x_val)} test={len(x_test)} device={device}")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = (len(x_tr) + BATCH_SIZE - 1) // BATCH_SIZE
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.2)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x_tr))
        tot_loss, correct, n = 0.0, 0, 0
        te = time.time()
        for k in range(0, len(x_tr), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            xb = preprocess(x_tr[idx], augment=True)
            yb = y_tr[idx].to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(idx)
        # validation
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(x_val), 256):
                xb = preprocess(x_val[k:k + 256], augment=False)
                yb = y_val[k:k + 256].to(device)
                lo = model(xb)
                vl += criterion(lo, yb).item() * len(yb)
                vc += (lo.argmax(1) == yb).sum().item()
        hist["train_loss"].append(tot_loss / n)
        hist["train_acc"].append(correct / n)
        hist["val_loss"].append(vl / len(x_val))
        hist["val_acc"].append(vc / len(x_val))
        print(f"epoch {ep+1}/{EPOCHS} train_loss={tot_loss/n:.4f} train_acc={correct/n:.4f} "
              f"val_loss={vl/len(x_val):.4f} val_acc={vc/len(x_val):.4f} ({time.time()-te:.0f}s)")
        if vc / len(x_val) > best_val:
            best_val = vc / len(x_val)
            torch.save(model.state_dict(), OUT / "best_model.pt")
    train_time = time.time() - t0

    # ---------------- Test ----------------
    model.load_state_dict(torch.load(OUT / "best_model.pt", map_location=device))
    preds, _ = predict(model, x_test)
    y_true = y_test.numpy()
    y_pred = preds.numpy()
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # tahmin suresi (tek goruntu, batch=1)
    model.eval()
    sample = x_test[:1]
    with torch.no_grad():
        for _ in range(10):
            model(preprocess(sample, False))
        if device.type == "mps":
            torch.mps.synchronize()
        ts = time.time()
        for i in range(200):
            model(preprocess(x_test[i:i + 1], False))
        if device.type == "mps":
            torch.mps.synchronize()
    infer_ms = (time.time() - ts) / 200 * 1000

    metrics = {
        "test_accuracy": acc, "test_precision_macro": p, "test_recall_macro": r,
        "test_f1_macro": f1, "best_val_accuracy": best_val,
        "train_time_sec": train_time, "inference_ms_per_image": infer_ms,
        "history": hist,
        "hyperparameters": {"img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                            "lr": LR, "weight_decay": WEIGHT_DECAY,
                            "label_smoothing": LABEL_SMOOTHING, "scheduler": "OneCycleLR"},
        "device": str(device),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "classification_report.txt").write_text(report)

    # grafikler
    ep = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], "o-", label="train")
    ax[0].plot(ep, hist["val_loss"], "o-", label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], "o-", label="train")
    ax[1].plot(ep, hist["val_acc"], "o-", label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=120); plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Tahmin"); plt.ylabel("Gercek"); plt.title("Confusion Matrix (Test)")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=120); plt.close()

    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
