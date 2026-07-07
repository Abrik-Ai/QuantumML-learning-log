# PATTERN: Splitting data into Training and Testing sets
# Learned: x[:8] and y[:8] isolates data the model learns from, while x[8:] 
# and y[8:] reserves completely unseen data for evaluation. This forces 
# the model to learn the true mathematical function (y=2x) instead of 
# memorizing specific inputs (overfitting).
# 
# KEY RESULT: The model trained exclusively on numbers 1 through 8, but when
# evaluated on unseen inputs (9 and 10), it correctly generalized—predicting 
# ~17.74 and ~19.66. This confirms the network learned the underlying pattern,
# not just the training points.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0],[10.0],[12.0],[14.0],[16.0],[18.0],[20.0]])

x_train, y_train = x[:8], y[:8]
x_test, y_test = x[8:], y[8:]

print("Train x:", x_train.flatten())
print("Test x:", x_test.flatten())

train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)

class HiddenLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(1, 8)
    self.activation = nn.ReLU()
    self.layer2 = nn.Linear(8, 1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    return x

model = HiddenLayer()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(100):
  epoch_loss = []
  for x_train, y_train in train_dataloader:
      y_pred = model(x_train)
      loss = loss_fn(y_pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss.append(loss.item())

avg_loss = sum(epoch_loss) / len(epoch_loss)
print(f"Final training loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_pred = model(x_test)
    test_loss = loss_fn(test_pred, y_test)

print(f"Test predictions: {test_pred.flatten()}")
print(f"Actual test values: {y_test.flatten()}")
print(f"Test loss: {test_loss.item():.4f}")
