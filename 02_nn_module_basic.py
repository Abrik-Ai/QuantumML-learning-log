# PATTERN: Same problem as file 01, but using PyTorch's nn.Module, nn.Linear,
# nn.MSELoss, and torch.optim instead of manual tensors and manual updates
# Learned: nn.Module, self, __init__, forward(), nn.Linear, nn.MSELoss,
# optimizer.zero_grad(), optimizer.step() — the "real world" PyTorch pattern
# Goal: same y = 2x problem, solved the way real PyTorch projects are structured

import torch
import torch.nn as nn

# Training data: y = 2x (note: 2D shape required by nn.Linear)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # 1 input, 1 output — replaces manual w

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    y_pred = model(x)              # forward pass, using the model object
    loss = loss_fn(y_pred, y)      # built-in MSE loss function

    optimizer.zero_grad()          # reset gradients (replaces manual w.grad.zero_())
    loss.backward()                # backpropagation (same as before)
    optimizer.step()               # gradient descent update (replaces manual w -= ...)

    print(f"epoch {epoch}: loss = {loss.item():.3f}")
