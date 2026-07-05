# PATTERN: Full training pipeline with batching (TensorDataset + DataLoader),
# a hidden layer network, and optimizer comparison (SGD vs Adam)
# Learned: TensorDataset, DataLoader, batch_size, shuffle=True,
# looping over batches inside each epoch, averaging loss across batches
# for clean per-epoch reporting
#
# KEY LESSON FROM EXPERIMENTATION:
# - SGD (lr=0.01)   -> too high, got stuck around loss ~30-40
# - SGD (lr=0.001)  -> excellent, converged to loss ~0.002
# - Adam (lr=0.01)  -> decent but didn't fully converge (~0.376)
# - Adam (lr=0.001) -> underperformed, too slow (~34)
# Takeaway: neither optimizer is "automatically better" — learning rate
# tuning matters more than the choice of optimizer alone.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Training data: y = 2x, now with 10 examples instead of 4
x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0],[10.0],[12.0],[14.0],[16.0],[18.0],[20.0]])

# Pair x and y together, then set up automatic batching + shuffling
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=3, shuffle=True)

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
optimizer = optim.SGD(model.parameters(), lr=0.001)   # best result from experiments
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # for comparison

for epoch in range(100):
    epoch_losses = [] # List to save errors
    for batch_x, batch_y in loader: #loop not across a single item
        y_pred = model(batch_x)     #but across batches inside epoch
        loss = loss_fn(y_pred, batch_y) # calculate how wrong this batch's predictions are

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item()) #add loss values to epoch_losses list(array)

    avg_loss = sum(epoch_losses) / len(epoch_losses) # average across all batches this epoch, so we get one clean number instead of several
    print(f"epoch {epoch}: avg loss = {avg_loss:.3f}")
