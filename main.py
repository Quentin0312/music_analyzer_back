from starlette.routing import WebSocketRoute
from fastapi import FastAPI

from source import controller

app = FastAPI()

"""
TODO : Change prediction execution flow
Do loop(preprocess 3sec, predict and send)
Instead of preprocess all and predict all 
"""


app.router.routes.append(
    WebSocketRoute(
        path="/ws/{preprocessing_type}",
        endpoint=controller.websocket_endpoint,
    )
)


# TODO: Move into another file
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
