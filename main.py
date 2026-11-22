
import torch

# Create a small fake RGB image:
# 3 colour channels, 4 pixels tall, 5 pixels wide
image = torch.rand(3, 4, 5)

print("Image tensor:")
print(image)

print("Image shape:", image.shape)
print("Number of values:", image.numel())
