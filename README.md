Denoising Autoencoder - Handwritten Digit Cleaner

A simple and interactive Streamlit web app that lets you draw a digit, adds Gaussian noise, and uses a custom Denoising Autoencoder (DAE) trained on MNIST to clean and restore your digit.
Features

    Draw digits on a 140x140 black canvas using white strokes, similar in style and thickness to MNIST digits.

    Automatic preprocessing: thresholds, crops, resizes, and centers your drawing into a 28×28 grayscale image.

    Adds random Gaussian noise to simulate real-world input distortion.

    Uses a deep learning autoencoder to denoise and reconstruct a clean version of your digit.

    Displays both the noisy and denoised images for comparison.

How to Run

Clone this repository:

    git clone https://github.com/your-username/dae-digit-cleaner.git
    cd dae-digit-cleaner

Install dependencies:

    pip install -r requirements.txt

Run the app:

    streamlit run app.py

Usage:

        Draw a digit in the canvas.

        Click the "Denoise" button.

        View the noisy and denoised outputs below the canvas.

Requirements

    Python 3.8+

    TensorFlow

    Streamlit

    streamlit-drawable-canvas

    numpy

    Pillow

Project Structure

    app.py — Main Streamlit app script.

    dae_latent_64.keras — Pretrained DAE model file.

    requirements.txt — Python dependencies.

Notes

    The model is trained on MNIST digits normalized to the [0, 1] range.

    The preprocessing mimics the MNIST format by applying thresholding, cropping, resizing, and padding.

    Update the model path in app.py to match the location of your .keras file.
