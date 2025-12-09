# 🧠 Brain Tumor Detection AI

Bu proje, Derin Öğrenme (Deep Learning) ve Transfer Learning yöntemlerini kullanarak Beyin MR görüntülerinden tümör tespiti yapan yapay zeka destekli bir web uygulamasıdır.

## 🚀 Özellikler

- **Çift Model Mimarisi:**
  - 🛡️ **Güvenlik Modeli:** Yüklenen resmin bir Beyin MR görüntüsü olup olmadığını kontrol eder.
  - 👨‍⚕️ **Teşhis Modeli:** MR görüntüsünde tümör riski olup olmadığını analiz eder.
- **Grad-CAM (Explainable AI):** Yapay zekanın kararı verirken resmin hangi bölgesine odaklandığını gösteren ısı haritası (Heatmap) oluşturur.
- **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş modern web arayüzü.
- **Veritabanı Kaydı:** Geçmiş taramaları SQLite veritabanında saklar ve listeler.

## ⚠️ Önemli Not (Kurulum)

Model dosyaları (`.h5`) boyutları nedeniyle bu depoya eklenmemiştir. Projeyi kendi bilgisayarınızda çalıştırmak için önce modelleri eğitmelisiniz:

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ## 📸 Ekran Görüntüleri

**Yapay Zeka Analiz Sonucu (Isı Haritası ile Tümör Tespiti):**

![Örnek Sonuç](https://github.com/oguzhanozmenn/Brain-Tumor-Detection-AI/blob/main/gecmis_taramalar/scan_20251209_180703.jpg)