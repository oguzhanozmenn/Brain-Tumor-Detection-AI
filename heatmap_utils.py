import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm


def get_last_conv_layer_name(model):
    """
    Modelin içindeki son evrişim (Convolutional) katmanının adını otomatik bulur.
    """
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            return layer.name
    raise ValueError("Modelde Konvolüsyon (Conv) katmanı bulunamadı!")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM algoritması ile ısı haritası matrisini oluşturur.
    """
    # Sequential modellerde bazen model.inputs liste olarak gelir,
    # bazen model.input tekil tensor olarak gelir.
    # En garantisi model.input kullanmaktır.

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradyan hesaplama
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradyanları al
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Isı haritasını oluştur
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizasyon
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Isı haritasını orijinal resimle birleştirir.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)