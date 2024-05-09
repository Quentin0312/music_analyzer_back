import joblib
import pandas as pd
from typing import List

from .var import PreprocessingType, column_names
from .utils import preprocessing_utils

# TODO: Vérifier les typages


def preprocess_data(
    scaler_path: str, uploaded_audio: bytes, preprocessing_type: PreprocessingType
) -> List[pd.DataFrame]:
    scaler = joblib.load(scaler_path)
    dfs = []
    segments = preprocessing_utils.get_3sec_sample(uploaded_audio)

    if preprocessing_type == PreprocessingType.fast:
        segments = preprocessing_utils.get_30_percent(segments)

    for audio in segments:
        # Perform audio feature extraction
        features = preprocessing_utils.audio_pipeline(audio)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform([features])

        # Create a DataFrame
        df = pd.DataFrame(scaled_features, columns=column_names)
        dfs.append(df)

    return dfs
