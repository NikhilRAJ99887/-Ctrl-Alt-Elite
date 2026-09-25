import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "model/audio_deepfake_model.keras"
)


# -----------------------------
# PAGE
# -----------------------------

st.title("Audio Deepfake Detection")

st.write(
    "Upload an audio file to check whether "
    "it is real human speech or AI-generated speech."
)


# -----------------------------
# UPLOAD AUDIO
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3" ,]
)


# -----------------------------
# ANALYZE
# -----------------------------

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Analyze Audio"):

        # Save uploaded file
        with open(
            "temp_audio.wav",
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )


        # Load audio
        audio, sr = librosa.load(
            "temp_audio.wav",
            sr=16000,
            duration=3
        )


        # -----------------------------
        # MEL SPECTROGRAM
        # -----------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )


        # -----------------------------
        # SHOW SPECTROGRAM
        # -----------------------------

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )

        ax.set_title(
            "Mel Spectrogram"
        )

        st.pyplot(fig)


        # -----------------------------
        # PREPARE FOR MODEL
        # -----------------------------

        image = tf.image.resize(
            mel_db[..., np.newaxis],
            [128, 128]
        )

        image = np.expand_dims(
            image.numpy(),
            axis=0
        )


        # -----------------------------
        # PREDICT
        # -----------------------------

        prediction = model.predict(
            image
        )[0][0]


        # -----------------------------
        # RESULT
        # -----------------------------

        if prediction >= 0.5:

            st.error(
                "DEEPFAKE AUDIO"
            )

            st.write(
                "Confidence:",
                round(
                    prediction * 100,
                    2
                ),
                "%"
            )

        else:

            st.success(
                "REAL HUMAN SPEECH"
            )

            st.write(
                "Confidence:",
                round(
                    (1 - prediction) * 100,
                    2
                ),
                "%"
            )