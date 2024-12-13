# edge-ai-solutions/edge_ai_example.py

import torch

# Example of a simple AI model optimized for edge devices (e.g., MobileNetV2)
from torchvision import models

# Load MobileNetV2 model (lightweight for edge devices)
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Example forward pass
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output)
