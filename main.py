from fastapi import FastAPI, UploadFile

import torch

from my_code.preprocessing import preprocess_data
from my_code.var import column_names, genre_mapping
from my_code.model import MusicClassifier, predict

import time

# TODO: Rename things
app = FastAPI()


@app.post("/predict")
def prediction(audio: UploadFile):
    beginning = time.time()
    # Preprocessing
    dfs = preprocess_data(
        scaler_path="./resources/trained_standard_scaler.pkl",
        column_names=column_names,
        uploaded_audio=audio,
    )

    # Load model
    my_model = MusicClassifier(input_features=55, output_features=10)
    my_model.load_state_dict(
        torch.load(
            f="./resources/actual_model_fast.pth", map_location=torch.device("cpu")
        )
    )

    # Predict
    result = predict(my_model, dfs, genre_mapping)

    return {"Prédiction => ": result, "prediction time": time.time() - beginning}
