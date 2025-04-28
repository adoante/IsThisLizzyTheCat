import torch
from torchvision import transforms, datasets, models
from PIL import Image
import os

# Settings
MODEL_PATH = "best_cat_model.pth"  # Path to the saved model
IMAGE_PATH = "00000455_017.jpg"  # Path to the image you want to test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform to preprocess the image before inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
model = models.resnet50(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # Assuming 2 classes
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model = model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# Load and preprocess the image
img = Image.open(IMAGE_PATH).convert("RGB")  # Open image and ensure it's in RGB
img = transform(img).unsqueeze(0).to(DEVICE)  # Apply the transform and add batch dimension

# Inference
with torch.no_grad():
    outputs = model(img)  # Pass the image through the model
    _, predicted = torch.max(outputs, 1)  # Get the predicted class

# Output the predicted class
class_names = ["my_cat", "not_my_cat"]  # Replace with your class names
predicted_class = class_names[predicted.item()]

print(f"Predicted class: {predicted_class}")
