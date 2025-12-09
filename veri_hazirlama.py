import os
import cv2
import numpy as np

# Görüntü boyutu
IMG_SIZE = 224

# Yollar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MRI_PATH = os.path.join(BASE_DIR, "dataset", "mri_data")
NON_MRI_PATH = os.path.join(BASE_DIR, "dataset", "non_mri_data")

data = []
labels = []

print("--- VERİ İŞLEME BAŞLADI ---")

# 1. ADIM: MR Görüntülerini Yükle (YES ve NO klasörleri)
print("1. MR Görüntüleri taranıyor...")
mri_count = 0
for category in ["yes", "no"]:
    path = os.path.join(MRI_PATH, category)
    if os.path.exists(path):
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(path, img_name)
                    img_array = cv2.imread(img_path)
                    if img_array is not None:
                        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        data.append(img_array)
                        labels.append(1) # 1 = MR Görüntüsü
                        mri_count += 1
                except:
                    pass

print(f"   -> Bulunan MR Görüntüsü: {mri_count}")

# 2. ADIM: MR OLMAYAN Görüntüleri Yükle (Alt klasörler dahil!)
print("2. MR Olmayan görüntüler taranıyor (Alt klasörler dahil)...")
non_mri_count = 0

# os.walk komutu tüm alt klasörleri (dog, fruit, person...) gezmesini sağlar
for root, dirs, files in os.walk(NON_MRI_PATH):
    for img_name in files:
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(root, img_name)
                img_array = cv2.imread(img_path)
                if img_array is not None:
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append(img_array)
                    labels.append(0) # 0 = MR Değil
                    non_mri_count += 1
            except:
                pass

print(f"   -> Bulunan Diğer Resimler: {non_mri_count}")

# Sonuçları Kaydet
data = np.array(data)
labels = np.array(labels)

print(f"\nTOPLAM VERİ SAYISI: {len(data)}")

if len(data) > 0:
    np.save("X_filter.npy", data)
    np.save("y_filter.npy", labels)
    print("✅ Veriler 'X_filter.npy' ve 'y_filter.npy' olarak başarıyla kaydedildi.")
else:
    print("❌ HATA: Hiç veri bulunamadı.")