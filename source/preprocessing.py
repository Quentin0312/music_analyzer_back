import io
import librosa
import numpy as np
import pandas as pd

import joblib
from typing import List

from . import var
from .var import PreprocessingType

# TODO: Vérifier les typages


def get_3sec_sample(uploaded_audio: bytes) -> List[np.ndarray]:
    audio, sample_rate = librosa.load(
        io.BytesIO(uploaded_audio),
        sr=None,
    )

    segment_duration = 3  # Durée de chaque segment en secondes
    segment_length = int(sample_rate * segment_duration)
    segments = []

    # Découpage
    for i in range(0, len(audio), segment_length):
        segment = audio[i : i + segment_length]
        segments.append(segment)

    return segments


def audio_pipeline(audio: np.ndarray) -> List[float]:
    features = []

    # Chromagram
    chroma_stft = librosa.feature.chroma_stft(y=audio)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))  # var => variance

    # RMS (Root Mean Square value for each frame)
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    # Calcul du Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio)
    features.append(np.mean(spectral_centroids))
    features.append(np.var(spectral_centroids))

    # Spectral bandwith
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.var(spectral_bandwidth))

    # Calcul du spectral rolloff point
    rolloff = librosa.feature.spectral_rolloff(y=audio)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    # Calcul du ZCR (Zero Crossing Rate)
    zcr = librosa.zero_crossings(audio)
    # features.append(np.sum(zcr))  # Custom
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    # Harmonic
    harmony = librosa.effects.harmonic(y=audio)
    features.append(np.mean(harmony))
    features.append(np.var(harmony))

    # Tempo
    tempo = librosa.feature.tempo(y=audio)
    features.append(tempo[0])

    # Calcul des moyennes des MFCC
    mfcc = librosa.feature.mfcc(y=audio)
    for x in mfcc:
        features.append(np.mean(x))
        features.append(np.var(x))

    return features


def preprocess_data(
    scaler_path: str, uploaded_audio: bytes, preprocessing_type: PreprocessingType
) -> List[pd.DataFrame]:
    scaler = joblib.load(scaler_path)
    dfs = []
    segments = get_3sec_sample(uploaded_audio)

    # TODO: Refactor !
    if preprocessing_type == PreprocessingType.fast:
        lighten_segments = []
        for i in range(len(segments)):
            if i % 3 == 0:
                lighten_segments.append(segments[i])
        segments = lighten_segments

    for audio in segments:
        # Perform audio feature extraction
        features = audio_pipeline(audio)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform([features])

        # Create a DataFrame
        df = pd.DataFrame(scaled_features, columns=var.column_names)
        dfs.append(df)

    return dfs
