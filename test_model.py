import torch
from src.model import RetinalCNN

model = RetinalCNN()

# Simulate one batch of 16 transformed retinal images
images = torch.randn(16, 3, 224, 224)

# Pass the batch through the network
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Number of model parameters:",
      sum(p.numel() for p in model.parameters()))
print("Model test successful!")
