# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# # from dataset import CustomDataset
# from model import HandGestureClassifier

# class Trainer:
#     def __init__(self, model, train_loader, val_loader, criterion, optimizer, device=torch.device('cpu')):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.device = device

#     def train_model(self, epochs=10):
#         for epoch in range(epochs):
#             self.model.train()
#             running_loss = 0.0
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             epoch_loss = running_loss / len(self.train_loader)
#             print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

#     def evaluate_model(self):
#         self.model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in self.val_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         accuracy = correct / total
#         print(f"Accuracy on validation set: {100 * accuracy:.2f}%")

# # Example usage:
# # model = HandGestureClassifier(input_channels=3, output_classes=8)
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
# # trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
# # trainer.train_model(epochs=10)
# # trainer.evaluate_model()
