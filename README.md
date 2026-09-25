# 🎧 Audio Deepfake Detection Using Fourier Transform and CNN

## 📌 Project Overview

Audio Deepfake Detection is a machine learning project designed to distinguish between **real human speech** and **AI-generated (synthetic) speech**.

The project uses **Short-Time Fourier Transform (STFT)** to convert audio into a time-frequency spectrogram. These spectrograms are then processed using a **Convolutional Neural Network (CNN)** to classify the audio as either **REAL** or **FAKE**.

## 🎯 Objectives

* Detect AI-generated and synthetic speech.
* Convert audio signals into spectrograms using Fourier-based analysis.
* Extract useful time-frequency patterns from audio.
* Train a CNN model for real/fake classification.
* Provide a simple web interface using Streamlit.

## 🔄 Methodology

```text
Audio Input
     ↓
Audio Preprocessing
     ↓
Short-Time Fourier Transform (STFT)
     ↓
Spectrogram Generation
     ↓
Resize to 128 × 128
     ↓
CNN Model
     ↓
REAL / DEEPFAKE
```

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* SoundFile

## 📊 Dataset

This project uses the **WaveFake** dataset for training and evaluation.

WaveFake contains real speech and AI-generated speech produced using multiple speech synthesis architectures.

Dataset:
**WaveFake – A Data Set to Facilitate Audio DeepFake Detection**

## 🤖 CNN Architecture

The CNN model contains:

* Input layer: `128 × 128 × 1`
* Convolutional Layer: 32 filters
* Max Pooling
* Convolutional Layer: 64 filters
* Max Pooling
* Convolutional Layer: 128 filters
* Max Pooling
* Flatten layer
* Dense layer: 128 neurons
* Dropout: 0.5
* Output layer: Sigmoid

The final output classifies the audio as:

```text
0 → REAL
1 → FAKE
```

## ⚙️ Audio Processing

The audio is processed at a sampling rate of **16 kHz**.

STFT parameters:

```text
n_fft = 1024
hop_length = 512
```

The resulting spectrogram is converted into decibel scale and resized to:

```text
128 × 128 × 1
```

## 🖥️ Streamlit Application

The project includes a Streamlit web application where users can upload an audio file and receive a prediction.

The application:

1. Accepts an audio file.
2. Preprocesses the audio.
3. Generates its spectrogram.
4. Passes the spectrogram to the trained CNN.
5. Displays the prediction as **REAL** or **DEEPFAKE**.

Run the application using:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Audio_Deepfake_Detection/
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── model/
│   └── audio_deepfake_model.keras
│
├── train.py
├── predict.py
├── app.py
├── test_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Audio_Deepfake_Detection
```

Create a virtual environment:

```bash
py -3.12 -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## ▶️ Training

Place the real and fake audio files inside:

```text
dataset/real/
dataset/fake/
```

Then run:

```bash
python train.py
```

The trained model will be saved in:

```text
model/audio_deepfake_model.keras
```

## 🔍 Prediction

To test a single audio file:

```bash
python predict.py
```

## 📈 Results

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

> **Note:** Actual numerical results should be added here after training the final model.

Example:

```text
Accuracy  : XX%
Precision : XX%
Recall    : XX%
F1-Score  : XX%
```

## 🔮 Future Scope

* Support longer audio recordings using multiple audio segments.
* Improve detection of unseen AI voice generators.
* Add audio augmentation techniques.
* Compare STFT spectrograms with Mel spectrograms.
* Test different CNN architectures.
* Improve robustness against background noise and different audio formats.
* Deploy the application as a web-based detection system.

## 👨‍💻 Project

**Audio Deepfake Detection Using Fourier Transform and CNN**

Developed as an academic AI/ML project.

---

**Note:** This project is intended for educational and research purposes.
