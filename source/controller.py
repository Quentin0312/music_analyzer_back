from fastapi import WebSocket, WebSocketDisconnect
import onnxruntime

from source.model import onnx_predict

from source import preprocessing
from source.var import PreprocessingType
from source.utils import websocket_utils, log_utils


async def websocket_endpoint(websocket: WebSocket):
    preprocessing_type: PreprocessingType = websocket.path_params["preprocessing_type"]

    if not websocket_utils.is_connection_authorized(websocket):
        return None

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            # Preprocessing
            log_utils.print_preprocessing_title()
            dfs = preprocessing.preprocess_data(
                scaler_path="./resources/trained_standard_scaler.pkl",
                uploaded_audio=data,
                preprocessing_type=preprocessing_type,
            )

            # Load model
            onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

            # Predict
            log_utils.print_predicting_title()
            result = onnx_predict(onnx_session, dfs)

            log_utils.print_job_done()
            await websocket.send_text(result)

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e}")
