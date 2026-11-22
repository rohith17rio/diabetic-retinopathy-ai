
import torch
from torch import nn
from src.model import RetinalClassifier

# 1. Create the model
model = RetinalClassifier()

# 2. Create fake images and correct grades
images = torch.rand(20, 3, 4, 4)
grades = torch.randint(0, 5, (20,))

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01
)

# 4. Train for five epochs
for epoch in range(5):
    # Forward propagation
    scores = model(images)

    # Calculate loss
    loss = criterion(scores, grades)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    print(
        f"Epoch {epoch + 1}/5, "
        f"Loss: {loss.item():.4f}"
    )

print("Training loop completed!")
