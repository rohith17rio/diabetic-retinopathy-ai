
import torch
from torch import nn
from src.model import RetinalClassifier

# 1. Create the model
model = RetinalClassifier()

# 2. Create four fake images and their correct grades
images = torch.rand(4, 3, 4, 4)
grades = torch.tensor([0, 1, 2, 4])

# 3. Choose a loss function
criterion = nn.CrossEntropyLoss()

# 4. Choose an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. Forward propagation: make predictions
scores = model(images)

# 6. Calculate how wrong the predictions are
loss = criterion(scores, grades)
print("Loss before training:", loss.item())

# 7. Clear old gradients
optimizer.zero_grad()

# 8. Backpropagation: calculate gradients
loss.backward()

print(
    "Gradients calculated:",
    model.network[1].weight.grad is not None
)

# 9. Optimizer: update the model's weights
optimizer.step()

print("Training step completed!")
