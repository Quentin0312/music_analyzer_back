import torch
from torch import nn

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

from typing import List

# TODO: Add type to everything
# TODO: Ranger avec bloc de commentaires OU plusieurs modules (pred, visualisation, ...)


class MusicClassifier(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(
                in_features=input_features, out_features=256, dtype=torch.float32
            ),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=128, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=128, out_features=output_features, dtype=torch.float32
            ),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


def accuracy_fn(y_true, y_pred):
    correct = (
        torch.eq(input=y_true, other=y_pred).sum().item()
    )  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100  # Calcul simple de pourcentage
    return acc


def get_training_fns(model: MusicClassifier):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.011)

    return loss_fn, optimizer


def init_model(MusicClassifier: MusicClassifier):
    torch.manual_seed(42)

    return MusicClassifier(input_features=55, output_features=10)


def predict(model: MusicClassifier, df: pd.DataFrame, genre_mapping: dict[int, str]):
    # TODO: Rewrite
    model.eval()

    class_predictions = []
    raw_results = []
    total_rows, _ = df.shape

    for i in range(total_rows):
        y_logits = model(
            torch.from_numpy(df.iloc[i].to_numpy().reshape(55, 1).transpose()).type(
                torch.float32
            )
        )

        y_softmax = torch.softmax(y_logits, dim=1)
        y_pred = y_softmax.argmax(dim=1)

        raw_results.append(y_softmax.detach().numpy())
        class_predictions.append(genre_mapping[y_pred.detach().numpy()[0]])

    unique_values = set(class_predictions)
    actual_best = 0
    for elt in unique_values:
        if class_predictions.count(elt) > actual_best:
            actual_best = class_predictions.count(elt)
            prediction = elt
    return prediction, raw_results


def filter_data(
    kept_df: pd.DataFrame,
    threshold: float,
    raw_results: list,
    dataframe: pd.DataFrame,
    real_class: int,
    column_names: List[str],
):
    for i in range(len(raw_results)):
        if raw_results[i][0][int(real_class)] > threshold:
            kept_df = pd.concat(
                [
                    kept_df,
                    pd.DataFrame(
                        dataframe.iloc[i].to_numpy().reshape(55, 1).transpose(),
                        columns=column_names,
                    ),
                ],
                axis=0,
            )
            kept_df["label"] = real_class
    return kept_df


def display_training_metrics(
    epochs: int,
    loss_history: List[float],
    test_loss_history: List[float],
    acc_history: List[float],
    test_acc_history: List[float],
):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), loss_history, label="Training loss")
    plt.plot(range(epochs), test_loss_history, label="Testing loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), acc_history, label="Training acc")
    plt.plot(range(epochs), test_acc_history, label="Testing acc")
    plt.legend()
    plt.title("Accuracy")

    plt.show()


def plot_accuracy_evolution(values: List[float]):
    # TODO: Refactor
    plt.plot(values)
    plt.title("Accuracy evolution through trainings")
    plt.xlabel("Trainings")
    plt.ylabel("Accuracy")
    plt.show()


def plot_metrics_evolution(metrics_history: dict, metrics_name: str):
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.plot(metrics_history[i], label=f"{metrics_name} class {i}")
    plt.title(f"Class {metrics_name} evolutions through trainings")
    plt.xlabel("Trainings")
    plt.ylabel(metrics_name)
    plt.legend()
    plt.show()


def plot_global_metrics_evolution(
    micro_history, macro_history, weighted_history, metric_name: str
):
    plt.figure(figsize=(15, 5))
    plt.plot(micro_history, label="micro")
    plt.plot(macro_history, label="macro")
    plt.plot(weighted_history, label="weighted")
    plt.title(f"Mean {metric_name} through trainings")
    plt.legend()
    plt.show()


def get_data(df: pd.DataFrame, device: str):
    # TODO: Séparer avec une nouvelle fonction get_test_data !
    # Prepare data
    X = (
        torch.from_numpy(df.drop(columns=["label"]).to_numpy())
        .type(torch.float32)
        .to(device)
    )
    y = torch.from_numpy(df["label"].to_numpy()).type(torch.long).to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def training_loop(
    MusicClassifier: MusicClassifier,
    df: pd.DataFrame,
):
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init model
    model = init_model(MusicClassifier).to(device)

    loss_fn, optimizer = get_training_fns(model)

    # Prepare data
    X_train, X_test, y_train, y_test = get_data(df, device)

    precision = {}
    recall = {}
    f1_score_values = {}

    # Training loop
    torch.manual_seed(42)
    epochs = 126

    loss_history = []
    acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        """
        Train
        """
        model.train()

        # 1. Forward pass
        y_logits = model(X_train)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # 2. Metrics
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 2.1 Save metrics
        loss_history.append(loss.cpu().detach().numpy())
        acc_history.append(acc)

        # 3. Zero Grad
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Optimmizer step
        optimizer.step()

        """
        Test
        """
        model.eval()

        with torch.inference_mode():
            # 1. Forward pass
            y_test_logits = model(X_test)
            y_test_pred = torch.softmax(y_test_logits, dim=1).argmax(dim=1)

            # 2. Metrics
            test_loss = loss_fn(y_test_logits, y_test)
            test_acc = accuracy_fn(y_pred=y_test_pred, y_true=y_test)

            # 2.1 Save metrics
            test_loss_history.append(test_loss.cpu().detach().numpy())
            test_acc_history.append(test_acc)

        # Print out what's happening
        if epoch == 125:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%"
            )

    # TODO: Factoriser
    # Matrice de confusion
    ConfusionMatrixDisplay(
        confusion_matrix(y_test.cpu(), y_test_pred.cpu().numpy())
    ).plot()

    # Metrics d'entrainements
    display_training_metrics(
        epochs, loss_history, test_loss_history, acc_history, test_acc_history
    )

    # Métrics supplémentaires
    print(classification_report(y_true=y_test.cpu(), y_pred=y_test_pred.cpu()))

    # TODO: Factoriser
    for i in range(10):
        # Per class
        precision[i] = precision_score(
            y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average=None
        )[
            i
        ]  # TODO: Check what is the good value for 'average'
        recall[i] = recall_score(
            y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average=None
        )[i]
        f1_score_values[i] = f1_score(
            y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average=None
        )[i]

    # Global
    precision_micro = precision_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="micro"
    )
    precision_macro = precision_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="macro"
    )
    precision_weighted = precision_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="weighted"
    )

    recall_micro = recall_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="micro"
    )
    recall_macro = recall_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="macro"
    )
    recall_weighted = recall_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="weighted"
    )

    f1_score_micro = f1_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="micro"
    )
    f1_score_macro = f1_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="macro"
    )
    f1_score_weighted = f1_score(
        y_true=y_test.cpu(), y_pred=y_test_pred.cpu(), average="weighted"
    )
    # ! Faire pareil pour recall et f1_score !
    # TODO: Retourner somme et détails de precision, recall et f1_score pour plots les évolutions !
    # ! Utiliser sklearn.metrics.precision etc...
    return (
        test_acc,
        precision,
        recall,
        f1_score_values,
        precision_micro,
        precision_macro,
        precision_weighted,
        recall_micro,
        recall_macro,
        recall_weighted,
        f1_score_micro,
        f1_score_macro,
        f1_score_weighted,
    )
