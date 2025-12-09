import os
import cv2
import numpy as np

# Görüntü boyutu
IMG_SIZE = 224

# Yollar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MRI_PATH = os.path.join(BASE_DIR, "dataset", "mri_data")

data = []
labels = []

print("--- TÜMÖR MODELİ İÇİN VERİ HAZIRLANIYOR ---")

# Sadece YES ve NO klasörlerine bakıyoruz (Araba/Çiçek resimleri bu modele girmemeli)
categories = ["no", "yes"]  # no=0 (Temiz), yes=1 (Tümörlü)

for category in categories:
    path = os.path.join(MRI_PATH, category)
    class_num = categories.index(category)  # 0 veya 1

    if os.path.exists(path):
        files = os.listdir(path)
        print(f" -> '{category}' klasörü taranıyor... ({len(files)} dosya)")

        for img_name in files:
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                if img_array is not None:
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append(img_array)
                    labels.append(class_num)
            except:
                pass

data = np.array(data)
labels = np.array(labels)

print(f"\nTOPLAM MR SAYISI: {len(data)}")

if len(data) > 0:
    # İsimleri farklı kaydediyoruz ki diğerleri ile karışmasın
    np.save("X_tumor.npy", data)
    np.save("y_tumor.npy", labels)
    print("✅ Veriler 'X_tumor.npy' ve 'y_tumor.npy' olarak kaydedildi.")
else:
    print("❌ HATA: Veri bulunamadı.")