import os
# GPU'yu görünmez yapıyoruz, sadece CPU kullanacak
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# Mac Metal (MPS) hızlandırmasını da kapatıyoruz (Donmayı önlemek için)
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

print("--- CPU MODUNDA BAŞLATILIYOR (M4 İşlemci Gücü) ---")

# 1. Verileri Yükle
print("Veriler yükleniyor...")
if not os.path.exists("X_filter.npy") or not os.path.exists("y_filter.npy"):
    print("HATA: .npy dosyaları bulunamadı!")
    exit()

X = np.load("X_filter.npy")
y = np.load("y_filter.npy")

# 2. Normalizasyon
X = X / 255.0

# 3. Veriyi Böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli Kur
print("Model oluşturuluyor...")
model = Sequential()

# Input katmanını açıkça belirtiyoruz (Hata mesajını engellemek için)
model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 5. Modeli Derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Eğitimi Başlat
print("Eğitim başlıyor...")
# Batch size'ı azalttık ve epoch sayısını 5'e çektik (Hız için yeterli)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=16)

# 7. Modeli Kaydet
model.save("mri_validator_model.h5")
print("\nBAŞARILI: Model eğitildi ve 'mri_validator_model.h5' olarak kaydedildi.")

# Sonuç
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluk Oranı: %{accuracy * 100:.2f}")