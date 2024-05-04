# import torch
# from torch import nn

# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# def training_loop(MusicClassifier):
#     # TODO: Externalise this
#     # Init le model
#     torch.manual_seed(42)
#     model = MusicClassifier(input_features=55, output_features=10)

#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.011)

#     def accuracy_fn(y_true, y_pred):
#         correct = (
#             torch.eq(input=y_true, other=y_pred).sum().item()
#         )  # torch.eq() calculates where two tensors are equal
#         acc = (correct / len(y_pred)) * 100  # Calcul simple de pourcentage
#         return acc

#     # Prepare data
#     df = pd.read_csv("./csv/actual_dataset.csv")
#     # df = pd.read_csv("/app/resources/original_dataset.csv")
#     X = torch.from_numpy(df.drop(columns=["label"]).to_numpy()).type(torch.float32)
#     y = torch.from_numpy(df["label"].to_numpy()).type(torch.long)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Training loop
#     torch.manual_seed(42)
#     epochs = 125
#     for epoch in range(epochs + 1):
#         """
#         Train
#         """
#         model.train()

#         # 1. Forward pass
#         y_logits = model(X_train)
#         y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

#         # 2. Metrics
#         loss = loss_fn(y_logits, y_train)
#         acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

#         # 2.1 Save metrics
#         # loss_history.append(loss.cpu().detach().numpy())
#         # acc_history.append(acc)

#         # 3. Zero Grad
#         optimizer.zero_grad()

#         # 4. Backpropagation
#         loss.backward()

#         # 5. Optimmizer step
#         optimizer.step()

#         """
#         Test
#         """
#         model.eval()

#         with torch.inference_mode():
#             # 1. Forward pass
#             y_test_logits = model(X_test)
#             y_test_pred = torch.softmax(y_test_logits, dim=1).argmax(dim=1)

#             # 2. Metrics
#             test_loss = loss_fn(y_test_logits, y_test)
#             test_acc = accuracy_fn(y_pred=y_test_pred, y_true=y_test)

#             # 2.1 Save metrics
#             # test_loss_history.append(test_loss.cpu().detach().numpy())
#             # test_acc_history.append(test_acc)

#         # Print out what's happening
#         if epoch % 25 == 0:
#             print(
#                 f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%"
#             )

#     # if epoch == 125:
#     cm = confusion_matrix(y_test, y_test_pred.numpy())
#     ConfusionMatrixDisplay(cm).plot()
#     # Save the model
#     torch.save(obj=model.state_dict(), f="./actual_model_fast.pth")
