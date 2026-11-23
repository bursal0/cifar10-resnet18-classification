# CIFAR-10 Görüntü Sınıflandırma — ResNet18 Transfer Learning

ImageNet ön-eğitimli ResNet18, CIFAR-10 üzerinde fine-tune edildi. Test doğruluğu **%95.03** (macro F1 %95.02). Model, Gradio arayüzü ile çalışan bir web uygulaması olarak sunuluyor.

Detaylı rapor: [RAPOR.pdf](RAPOR.pdf) / [RAPOR.md](RAPOR.md) / `RAPOR.docx`

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Kullanım

```bash
.venv/bin/python train.py               # eğitim + test (CIFAR-10 otomatik indirilir), çıktılar outputs/
.venv/bin/python app.py                 # Gradio arayüzü: http://127.0.0.1:7860
.venv/bin/python test_phone_photos.py   # phone_photos/ içindeki fotoğrafları test eder
.venv/bin/python test_failure_cases.py  # başarısız durum analizi
```

Eğitilmiş model `outputs/best_model.pt` dosyasında. Uygulamayı çalıştırmak için yeniden eğitim gerekmez.

## Sonuçlar

| Metrik | Değer |
|---|---|
| Accuracy | %95.03 |
| Precision (macro) | %95.02 |
| Recall (macro) | %95.03 |
| F1 (macro) | %95.02 |
| Eğitim süresi | ~18.5 dk (Apple M2, 4 epoch) |
| Tahmin süresi | ~6.8 ms / görüntü |
| Telefon fotoğrafları | 6/6 doğru |
