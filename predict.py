import numpy as np
import librosa
import tensorflow as tf

# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "model/audio_deepfake_model.keras"
)


# -----------------------------
# CREATE SPECTROGRAM
# -----------------------------

def create_spectrogram(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        duration=3
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],
        [128, 128]
    )

    return mel_db.numpy()


# -----------------------------
# AUDIO FILE
# -----------------------------

file_path = "test.wav"

spectrogram = create_spectrogram(
    file_path
)

# Add batch dimension
spectrogram = np.expand_dims(
    spectrogram,
    axis=0
)


# -----------------------------
# PREDICTION
# -----------------------------

prediction = model.predict(
    spectrogram
)[0][0]


# -----------------------------
# RESULT
# -----------------------------

if prediction >= 0.5:

    print("Prediction: DEEPFAKE")

    print(
        "Confidence:",
        round(prediction * 100, 2),
        "%"
    )

else:

    print("Prediction: REAL")

    print(
        "Confidence:",
        round((1 - prediction) * 100, 2),
        "%"
    )