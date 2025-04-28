import torch
import torchvision.models as models
import torchvision

# Load the trained model (Assuming your model is saved as 'best_cat_model.pth')
trained_model_path = "best_cat_model.pth"

# Load the model architecture (ResNet50 in this case)
model = torchvision.models.resnet50(pretrained=False)  # Use the same architecture as you trained on
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # Adjust the final layer for your task (2 classes)

# Load the weights from the trained model
model.load_state_dict(torch.load(trained_model_path))

# Dummy input (adjust shape to your model's input size)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, "IsLizzyTheCat.onnx", input_names=['input'], output_names=['output'], opset_version=11)
