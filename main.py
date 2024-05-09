from enum import Enum

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime

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


class PreprocessingType(Enum):
    fast = "fast"
    complete = "complete"


"""
TODO : Change prediction execution flow
Do loop(preprocess 3sec, predict and send)
Instead of preprocess all and predict all 
"""


# TODO: Use try catch to send error message to front-end in case of error
@app.websocket("/ws/{preprocessing_type}")
async def websocket_endpoint(
    websocket: WebSocket, preprocessing_type: PreprocessingType
):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()

        # TODO: Redondancy => put the arg (fast or complete) to preprocess_data()
        # Preprocessing
        if preprocessing_type == PreprocessingType.fast:
            dfs = fast_preprocess_data(
                scaler_path="./resources/trained_standard_scaler.pkl",
                column_names=column_names,
                uploaded_audio=data,
            )
        else:
            dfs = preprocess_data(
                scaler_path="./resources/trained_standard_scaler.pkl",
                column_names=column_names,
                uploaded_audio=data,
            )

        onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

        # Predict
        result = onnx_predict(onnx_session, dfs, genre_mapping)

        await websocket.send_text(result)
