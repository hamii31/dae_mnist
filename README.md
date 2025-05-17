Denoising Autoencoder - MNIST Digit Cleaner

A Streamlit web app that lets you draw noisy handwritten digits and cleans them using a pretrained Denoising Autoencoder (DAE) model trained on MNIST.
Features

    Draw digits on a black canvas with white strokes (similar thickness to MNIST digits).

    Automatic preprocessing: cropping, resizing, and padding your drawing to 28x28 pixels.

    Optionally add Gaussian noise to simulate noisy inputs.

    Use a deep learning Denoising Autoencoder to clean the noisy digit.

    Visualize input, noisy, and denoised images side-by-side.

How to Run

    Clone this repository.

    Install dependencies:

pip install -r requirements.txt

    Run the app:

streamlit run app.py

    Draw a digit on the canvas and click Denoise to see the cleaned output.

Requirements

    Python 3.8+

    TensorFlow

    Streamlit

    streamlit-drawable-canvas

    numpy

    Pillow

Project Structure

    app.py - Main Streamlit app script.

    dae_latent_64.keras - Pretrained DAE model file.

    requirements.txt - Python dependencies.

Notes

    The model is trained on MNIST digits normalized to [0, 1].

    Preprocessing mimics MNIST digit style by cropping, resizing, and padding input drawings.

    The app currently requires the DAE model file path to be updated according to your environment.
