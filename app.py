import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
import cv2


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

@st.cache_resource
def load_dae():
    return tf.keras.models.load_model(
        r"dae_latent_64.keras",
        custom_objects={'psnr_metric': psnr_metric}
    )

dae_model = load_dae()

st.title("🧽 Denoising Autoencoder - MNIST Digit Cleaner")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=6,  # similar to MNIST line thickness
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_mnist_style(img):
    img = img[:, :, 0]  # take red channel
    img = img.astype("uint8")

    # Threshold the image
    _, img_bin = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Find bounding box of the digit
    coords = cv2.findNonZero(img_bin)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img_bin[y:y+h, x:x+w]
    else:
        digit = img_bin

    # Resize while maintaining aspect ratio
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.zeros((28, 28), dtype="float32")
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized / 255.0

    return padded.reshape(1, 28, 28, 1)

if canvas_result.image_data is not None:
    img = preprocess_mnist_style(canvas_result.image_data)

    st.subheader("✏️ Preprocessed Digit")
    st.image(img.reshape(28, 28), width=150, clamp=True)

    add_noise = st.checkbox("🌪 Add Gaussian Noise?", value=False)
    noise_level = st.slider("Noise Level", 0.0, 0.5, 0.3, 0.05) if add_noise else 0.0

    if add_noise:
        noise = np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        noisy_img = np.clip(img + noise_level * noise, 0.0, 1.0)
        st.image(noisy_img.reshape(28, 28), width=150, caption="Noisy Input", clamp=True)
    else:
        noisy_img = img

    if st.button("🧼 Denoise"):
        denoised = dae_model.predict(noisy_img)

        st.subheader("✅ Denoised Output")
        st.image(denoised.reshape(28, 28), width=150, clamp=True)
