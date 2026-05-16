import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

import cv2
import matplotlib.cm as cm

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "improved_cnn_gtsrb.keras"
IMG_SIZE = (64, 64)
LAST_CONV_LAYER_NAME = "conv2d_9"

SIGN_LABELS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 tons",
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None

        for layer in model.layers:
            x = layer(x)

            if layer.name == last_conv_layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        return np.zeros(conv_outputs.shape[1:3])

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return heatmap.numpy()

    heatmap = heatmap / max_value

    return heatmap.numpy()


def create_gradcam_overlay(img_array, heatmap, alpha=0.4):
    img = img_array[0]

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    overlay = jet_heatmap * alpha + img
    overlay = np.clip(overlay, 0, 1)

    return overlay

st.set_page_config(page_title="German Traffic Sign AI", page_icon="🚦")

st.title("🚦 German Traffic Sign AI")
st.write(
    "AI-powered German traffic sign classifier built using TensorFlow and CNNs."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "ppm"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)[0]

    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    st.subheader("Prediction")
    st.write(f"**Class:** {SIGN_LABELS[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    st.subheader("Top 3 Predictions")

    top_3 = predictions.argsort()[-3:][::-1]

    for idx in top_3:
        st.progress(float(predictions[idx]))
        st.write(f"{SIGN_LABELS[int(idx)]}: {predictions[idx]*100:.2f}%")
    
    st.subheader("Grad-CAM Explanation")

    heatmap = make_gradcam_heatmap(
        processed_image,
        model,
        LAST_CONV_LAYER_NAME,
        pred_index=predicted_class
    )

    overlay = create_gradcam_overlay(processed_image, heatmap)

    st.image(
        overlay,
        caption="Highlighted regions influenced the model's prediction",
        use_container_width=True
    )

    st.caption(
        "Warmer regions indicate areas that contributed more strongly to the prediction."
    )