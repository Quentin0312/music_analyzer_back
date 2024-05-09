from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime

from source.model import onnx_predict

from source import preprocessing
from source.var import PreprocessingType

# TODO: Rename things
app = FastAPI()

# TODO: Do the minimal to secure access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        while True:
            data = await websocket.receive_bytes()

            # Preprocessing
            dfs = preprocessing.preprocess_data(
                scaler_path="./resources/trained_standard_scaler.pkl",
                uploaded_audio=data,
                preprocessing_type=preprocessing_type,
            )

            # Load model
            onnx_session = onnxruntime.InferenceSession("./resources/model.onnx")

            # Predict
            result = onnx_predict(onnx_session, dfs)

            await websocket.send_text(result)
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e}")


""" TODO: Move into another file
    Routing needs to be done !
"""
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
#         uploaded_audio=audio,
#         preprocessing_type=PreprocesingType.complete,
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
