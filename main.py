import time
import io
from io import BytesIO
from datetime import datetime

from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# import torch
import onnxruntime
import numpy as np

# from source.model import MusicClassifier, onnx_predict, predict
from source.model import onnx_predict
from source.preprocessing import fast_preprocess_data, preprocess_data
from source.var import column_names, genre_mapping


# TODO: Rename things
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ! TODO: Use a safer way to store this data
"""
use the cache (cachetools) TRY THIS !
"""
data_dict = {}


def prediction_task(audio: BytesIO, key: str):
    beginning = time.time()

    # Preprocessing
    dfs = preprocess_data(
        scaler_path="./resources/trained_standard_scaler.pkl",
        column_names=column_names,
        uploaded_audio=audio,
    )

    onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

    # Predict
    result = onnx_predict(onnx_session, dfs, genre_mapping)

    print("pred ", key, " done")

    data_dict[key] = {"predicted": result, "prediction_time": time.time() - beginning}


@app.post("/onnx_predict")
async def prediction(audio: UploadFile, background_tasks: BackgroundTasks):
    key = str(datetime.now())
    """
    UploadFile to BytesIO done here because background_tasks cause issues
    if done after
    """
    audio_file = io.BytesIO(audio.file.read())
    background_tasks.add_task(prediction_task, audio_file, key)
    return {"key": key}


@app.get("/datas")
def get_pred_results():
    return data_dict


@app.get("/data/{key}")
def get_pred_results(key):
    return data_dict[key]


# ! TODO: Adapt using background task
@app.post("/onnx_fast_predict")
def prediction(audio: UploadFile):
    print(str(audio))
    beginning = time.time()

    # Preprocessing
    dfs = fast_preprocess_data(
        scaler_path="./resources/trained_standard_scaler.pkl",
        column_names=column_names,
        uploaded_audio=audio,
    )

    onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

    # Predict
    result = onnx_predict(onnx_session, dfs, genre_mapping)

    return {"predicted": result, "prediction_time": time.time() - beginning}


# @app.post("/predict")
# def prediction(audio: UploadFile):
#     beginning = time.time()
#     # Preprocessing
#     dfs = preprocess_data(
#         scaler_path="./resources/trained_standard_scaler.pkl",
#         column_names=column_names,
#         uploaded_audio=audio,
#     )

#     # Load model
#     my_model = MusicClassifier(input_features=55, output_features=10)
#     my_model.load_state_dict(
#         torch.load(
#             f="./resources/actual_model_fast.pth", map_location=torch.device("cpu")
#         )
#     )

#     # Predict
#     result = predict(my_model, dfs, genre_mapping)

#     return {"predicted": result, "prediction_time": time.time() - beginning}


# @app.post("/fast_predict")
# def fast_prediction(audio: UploadFile):
#     beginning = time.time()
#     # Preprocessing
#     dfs = fast_preprocess_data(
#         scaler_path="./resources/trained_standard_scaler.pkl",
#         column_names=column_names,
#         uploaded_audio=audio,
#     )

#     # Load model
#     my_model = MusicClassifier(input_features=55, output_features=10)
#     my_model.load_state_dict(
#         torch.load(
#             f="./resources/actual_model_fast.pth", map_location=torch.device("cpu")
#         )
#     )

#     # Predict
#     result = predict(my_model, dfs, genre_mapping)

#     return {
#         "predicted": result,
#         "prediction_time": time.time() - beginning,
#     }  # ajouter raw result

# ! DO NOT DELETE ----
# Only use locally, to move to another branch / repo
"""
This is only used locally to create the ONNX model (.mar) to
deploy without the need of pytorch dependancy

Notes :
- Needs pytorch installed to work
- torch.Tensor(dfs[0].to_numpy()) is a dummy data for it to
    know what format is the input
"""
# @app.post("/createonnx")
# def prediction(audio: UploadFile):
#     # beginning = time.time()
#     # Preprocessing
#     dfs = preprocess_data(
#         scaler_path="./resources/trained_standard_scaler.pkl",
#         column_names=column_names,
#         uploaded_audio=audio,
#     )

#     # Load model
#     my_model = MusicClassifier(input_features=55, output_features=10)
#     my_model.load_state_dict(
#         torch.load(
#             f="./resources/actual_model_fast.pth", map_location=torch.device("cpu")
#         )
#     )

#     my_model.eval()

#     torch.onnx.export(
#         my_model,
#         torch.Tensor(dfs[0].to_numpy()),
#         "./resources/model.onnx",
#         verbose=True,
#     )

#     return {"done"}
# ! DO NOT DELETE ----
