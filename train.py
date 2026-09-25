import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

# -----------------------------
# SETTINGS
# -----------------------------

DATASET_PATH = "dataset"
IMG_SIZE = 128

X = []
y = []

# -----------------------------
# CREATE SPECTROGRAM
# -----------------------------

def create_spectrogram(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        duration=3
    )

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Resize to 128 x 128
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],
        [IMG_SIZE, IMG_SIZE]
    )

    return mel_db.numpy()


# -----------------------------
# LOAD REAL AUDIO
# Label = 0
# -----------------------------

real_path = os.path.join(
    DATASET_PATH,
    "real"
)

for file in os.listdir(real_path):

    if file.endswith(".wav"):

        path = os.path.join(
            real_path,
            file
        )

        try:
            spectrogram = create_spectrogram(path)

            X.append(spectrogram)
            y.append(0)

            print("Real:", file)

        except Exception as e:
            print("Error:", file, e)


# -----------------------------
# LOAD FAKE AUDIO
# Label = 1
# -----------------------------

fake_path = os.path.join(
    DATASET_PATH,
    "fake"
)

for file in os.listdir(fake_path):

    if file.endswith(".wav"):

        path = os.path.join(
            fake_path,
            file
        )

        try:
            spectrogram = create_spectrogram(path)

            X.append(spectrogram)
            y.append(1)

            print("Fake:", file)

        except Exception as e:
            print("Error:", file, e)


# -----------------------------
# CONVERT TO NUMPY
# -----------------------------

X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# CNN MODEL
# -----------------------------

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 1)
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    )
])


# -----------------------------
# COMPILE
# -----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# SHOW MODEL
# -----------------------------

model.summary()


# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)


# -----------------------------
# TEST
# -----------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)


# -----------------------------
# SAVE MODEL
# -----------------------------

os.makedirs("model", exist_ok=True)

model.save(
    "model/audio_deepfake_model.keras"
)

print("\nModel saved successfully!")