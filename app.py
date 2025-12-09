import os
import streamlit as st
import cv2
import numpy as np
import sqlite3
from PIL import Image
from datetime import datetime

# --- KRİTİK AYARLAR (MAC M4) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import heatmap_utils

# --- AYARLAR ---
st.set_page_config(page_title="Beyin MR Analiz Asistanı", layout="wide")

MODEL_VALIDATOR_PATH = "mri_validator_model.h5"
MODEL_TUMOR_PATH = "tumor_detector_model.h5"
HISTORY_DIR = "gecmis_taramalar"
DB_FILE = "taramalar_v3.db"

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# --- VERİTABANI İŞLEMLERİ ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sonuclar
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tarih TEXT,
                  dosya_adi TEXT,
                  sonuc TEXT,
                  oran REAL)''')
    conn.commit()
    conn.close()

def save_result(dosya_adi, sonuc, oran):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO sonuclar (tarih, dosya_adi, sonuc, oran) VALUES (?, ?, ?, ?)",
              (tarih, dosya_adi, sonuc, oran))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM sonuclar ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data

# --- MODEL MİMARİSİ (ELLE KURULUM) ---
def build_tumor_model():
    """
    Modelin iskeletini kodla sıfırdan kuruyoruz.
    Bu yöntem 'Layer has no input' hatasını %100 çözer.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
    from tensorflow.keras.models import Model

    # Giriş Katmanını Açıkça Tanımlıyoruz (Functional API)
    inputs = Input(shape=(224, 224, 3))

    # Katmanlar (Eğitimdeki mimarinin aynısı)
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D(2,2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    # Modeli oluştur
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_models_lazy():
    import tensorflow as tf
    try:
        # 1. Validator Modelini Normal Yüklüyoruz (Bunda Grad-CAM yok, sorun çıkmaz)
        validator = tf.keras.models.load_model(MODEL_VALIDATOR_PATH)

        # 2. Doktor Modelini 'TRANSFER' Yöntemiyle Yüklüyoruz
        # Önce kayıtlı dosyayı geçici olarak yükle
        temp_model = tf.keras.models.load_model(MODEL_TUMOR_PATH)

        # Şimdi kendi sağlam modelimizi oluştur
        doctor = build_tumor_model()

        # Kayıtlı dosyadaki 'beyni' (ağırlıkları) bizim sağlam modele aktar
        doctor.set_weights(temp_model.get_weights())

        return validator, doctor
    except Exception as e:
        st.error(f"Model yükleme hatası detaylı: {e}")
        return None, None

init_db()

# --- ARAYÜZ ---
st.title("🧠 Beyin Tümörü Tespit Sistemi (AI + Grad-CAM)")
st.write("Bu sistem, yüklenen görüntülerin **Beyin MR** olup olmadığını kontrol eder, tümör riski analizi yapar ve **şüpheli bölgeyi işaretler.**")

with st.spinner("Yapay Zeka Hazırlanıyor..."):
    model_validator, model_doctor = load_models_lazy()

if model_validator is None or model_doctor is None:
    st.error("HATA: Modeller yüklenemedi! Dosya yollarını kontrol edin.")
    st.stop()
else:
    st.sidebar.success("✅ Sistem Aktif")

# Geçmiş Menüsü
st.sidebar.title("🗂 Geçmiş Taramalar")
gecmis = get_history()

if len(gecmis) > 0:
    for kayit in gecmis:
        icon = "🔴" if "Riskli" in kayit[3] else "🟢"
        try:
            oran_degeri = float(kayit[4])
        except:
            oran_degeri = 0.0
        st.sidebar.markdown(f"{icon} **{kayit[3]}** (%{oran_degeri:.1f})\n<small>{kayit[1]}</small>", unsafe_allow_html=True)
        st.sidebar.divider()
else:
    st.sidebar.info("Henüz kayıt yok.")

# RESİM YÜKLEME ALANI
uploaded_file = st.file_uploader("Analiz edilecek MR görüntüsünü seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    with col1:
        st.info("Orijinal Görüntü")
        # 'use_column_width' uyarısını düzeltmek için width parametresi kullanıyoruz
        st.image(image, width=350)

    if st.button("🔍 Detaylı Analizi Başlat", type="primary"):
        with st.spinner('Yapay Zeka görüntüyü tarıyor...'):

            img_array = np.array(image.convert('RGB'))
            orig_img_path = "temp_img.jpg"
            image.save(orig_img_path)

            img_resized = cv2.resize(img_array, (224, 224))
            img_normalized = img_resized / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)

            # 1. AŞAMA
            is_mri_prob = model_validator.predict(img_input, verbose=0)[0][0]

            if is_mri_prob < 0.5:
                st.error(f"❌ BU BİR MR GÖRÜNTÜSÜ DEĞİL! (Güven: %{(1-is_mri_prob)*100:.2f})")
                st.warning("Lütfen sisteme sadece Beyin MR taramaları yükleyin.")
            else:
                st.success("✅ Görüntü Doğrulandı: Beyin MR")

                # 2. AŞAMA
                tumor_prob = model_doctor.predict(img_input, verbose=0)[0][0]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"scan_{timestamp}.jpg"
                save_path = os.path.join(HISTORY_DIR, save_name)

                if tumor_prob > 0.5:
                    guven = tumor_prob * 100
                    st.error(f"⚠️ DİKKAT: TÜMÖR RİSKİ TESPİT EDİLDİ")
                    st.write(f"Tespit Oranı: **%{guven:.2f}**")

                    # ISI HARİTASI (ARTIK ÇALIŞACAK)
                    try:
                        st.write("🔎 **Yapay Zeka Odak Analizi Yapılıyor...**")

                        last_conv_layer = heatmap_utils.get_last_conv_layer_name(model_doctor)
                        heatmap = heatmap_utils.make_gradcam_heatmap(img_input, model_doctor, last_conv_layer)
                        final_img = heatmap_utils.save_and_display_gradcam(orig_img_path, heatmap)

                        with col2:
                            st.error("Yapay Zeka Tespit Alanı")
                            st.image(final_img, caption="Kırmızı alanlar tümör şüphesi taşıyan bölgelerdir.", width=350)

                        Image.fromarray(final_img).save(save_path)
                        save_result(save_name, "Riskli (Tümör)", guven)

                    except Exception as e:
                        st.warning(f"Isı haritası oluşturulamadı: {e}")
                        image.save(save_path)
                        save_result(save_name, "Riskli (Tümör)", guven)

                else:
                    guven = (1 - tumor_prob) * 100
                    st.success(f"🟢 SONUÇ: NEGATİF (TEMİZ)")
                    st.write(f"Temizlik Oranı: **%{guven:.2f}**")
                    image.save(save_path)
                    save_result(save_name, "Temiz (Normal)", guven)