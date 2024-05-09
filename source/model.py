# from torch import nn
# import torch

from typing import List

import numpy as np
import pandas as pd
from onnxruntime import InferenceSession

from . import var

# ! DELETE NOTHING
# ! DO NOT DELETE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Only use locally, to move to another branch / repo

# class MusicClassifier(nn.Module):
#     def __init__(self, input_features, output_features):
#         super().__init__()
#         self.linear_layer_stack = nn.Sequential(
#             nn.Linear(
#                 in_features=input_features, out_features=256, dtype=torch.float32
#             ),
#             nn.GELU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=256, out_features=128, dtype=torch.float32),
#             nn.GELU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(
#                 in_features=128, out_features=output_features, dtype=torch.float32
#             ),
#         )

#     def forward(self, x):
#         return self.linear_layer_stack(x)


# def predict(
#     model: MusicClassifier, dfs: List[pd.DataFrame], genre_mapping: dict[int, str]
# ) -> str:
#     model.eval()

#     class_predictions = []
#     raw_results = []

#     for df in dfs:
#         y_logits = model(torch.from_numpy(df.to_numpy()).type(torch.float32))
#         y_softmax = torch.softmax(y_logits, dim=1)
#         print("y_softmax => ", y_softmax)
#         y_pred = y_softmax.argmax(dim=1)
#         print("y_pred => ", y_pred)

#         # print(genre_mapping[y_pred.detach().numpy()[0]])
#         # print(list(torch.round(y_softmax * 1000) / 1000))

#         raw_results.append(y_softmax.detach().numpy())
#         class_predictions.append(genre_mapping[y_pred.detach().numpy()[0]])

#     unique_values = set(class_predictions)
#     actual_best = 0
#     for elt in unique_values:
#         if class_predictions.count(elt) > actual_best:
#             actual_best = class_predictions.count(elt)
#             prediction = elt
#         print(elt, class_predictions.count(elt))

#     print("Results =>", prediction)
#     return prediction


# * Carreful => Blackbox
def softmax(z):
    assert len(z.shape) == 2

    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def onnx_predict(
    onnx_session: InferenceSession,
    dfs: List[pd.DataFrame],
) -> str:
    # model.eval()

    class_predictions = []

    for df in dfs:
        # Prepare input data for inference
        input_data = {
            onnx_session.get_inputs()[0].name: df.to_numpy().astype(np.float32)
        }

        y_logits = onnx_session.run(None, input_data)
        y_softmax = softmax(np.array(y_logits[0]))
        y_pred_list = y_softmax[0].tolist()
        y_pred_list.sort()
        y_pred = y_softmax[0].tolist().index(y_pred_list[-1])

        class_predictions.append(var.genre_mapping[y_pred])

    unique_values = set(class_predictions)
    actual_best = 0
    for elt in unique_values:
        if class_predictions.count(elt) > actual_best:
            actual_best = class_predictions.count(elt)
            prediction = elt
        print(elt, class_predictions.count(elt))

    print("Results =>", prediction)
    return prediction
