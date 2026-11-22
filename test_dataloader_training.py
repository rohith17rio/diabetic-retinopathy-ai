
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from src.model import RetinalClassifier

# 1. Create 20 fake images and their grades
images = torch.rand(20, 3, 4, 4)
grades = torch.randint(0, 5, (20,))

# 2. Package images and grades into a dataset
dataset = TensorDataset(images, grades)

# 3. Load four images per batch
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# 4. Create the model, loss function, and optimizer
model = RetinalClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01
)

# 5. Train for three epochs
for epoch in range(3):
    total_loss = 0.0

    for batch_images, batch_grades in loader:
        # Forward propagation
        scores = model(batch_images)

        # Calculate loss
        loss = criterion(scores, batch_grades)

        # Backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/3, "
        f"Average loss: {average_loss:.4f}"
    )

print("DataLoader training completed!")
