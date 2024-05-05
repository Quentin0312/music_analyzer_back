from fastapi import FastAPI, WebSocket
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

# TODO : Find a way to handle file > 15Mo

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# TODO: Make only one route with a parameter in URL
@app.websocket("/ws/fast")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()

        # Preprocessing
        dfs = fast_preprocess_data(
            scaler_path="./resources/trained_standard_scaler.pkl",
            column_names=column_names,
            uploaded_audio=data,
        )

        onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

        # Predict
        result = onnx_predict(onnx_session, dfs, genre_mapping)

        await websocket.send_text(f"Done : {result}")


@app.websocket("/ws/complete")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()

        # Preprocessing
        dfs = preprocess_data(
            scaler_path="./resources/trained_standard_scaler.pkl",
            column_names=column_names,
            uploaded_audio=data,
        )

        onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

        # Predict
        result = onnx_predict(onnx_session, dfs, genre_mapping)

        await websocket.send_text(f"Done : {result}")


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

# ! DO NOTE DELETE
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
