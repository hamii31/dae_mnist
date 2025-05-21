import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

@st.cache_resource
def load_dae():
    return tf.keras.models.load_model(
        r"dae_latent_64.keras",
        custom_objects={'psnr_metric': psnr_metric}
    )

dae_model = load_dae()

st.title("DAE for Handwritten Digits")
st.subheader("Simply draw a digit in the box below and click 'Denoise' to see the result.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=6,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=140, 
    width=140, 
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_mnist_style(img):
    img = img[:, :, 0] # Red channel
    img = img.astype("uint8")

    # Threshold the image (simple numpy threshold)
    threshold = np.percentile(img, 80)  # auto-adjusts threshold
    img_bin = (img > threshold).astype(np.uint8) * 255

    # Find bounding box of the digit
    coords = np.argwhere(img_bin)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        digit = img_bin[y0:y1+1, x0:x1+1]
    else:
        digit = img_bin

    # Resizing while maintaining aspect ratio using PIL
    digit_img = Image.fromarray(digit)
    h, w = digit_img.size[1], digit_img.size[0] 

    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    digit_img = digit_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create new 28x28 image and paste resized digit centered
    new_img = Image.new('L', (28, 28), 0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    new_img.paste(digit_img, (x_offset, y_offset))

    arr = np.array(new_img).astype("float32") / 255.0

    return arr.reshape(1, 28, 28, 1)

if canvas_result.image_data is not None:
    img = preprocess_mnist_style(canvas_result.image_data)

    noise = np.random.normal(loc=0.0, scale=1.0, size=img.shape)
    noisy_img = np.clip(img + 0.5 * noise, 0.0, 1.0)
    if noisy_img is not None: 
        st.subheader("Noisy Input")
        st.image(noisy_img.reshape(28, 28), width=150, clamp=True)

    if st.button("Denoise"):
        denoised = dae_model.predict(noisy_img)

        st.subheader("Denoised Output")
        st.image(denoised.reshape(28, 28), width=150, clamp=True)
