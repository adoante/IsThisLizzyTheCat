import torch
import torchvision
import qai_hub as hub
from typing import Tuple

# Load the trained model (Assuming your model is saved as 'best_cat_model.pth')
trained_model_path = "best_cat_model.pth"

# Load the model architecture (ResNet50 in this case)
model = torchvision.models.resnet50(pretrained=False)  # Use the same architecture as you trained on
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # Adjust the final layer for your task (2 classes)

# Load the weights from the trained model
model.load_state_dict(torch.load(trained_model_path))
model.eval()  # Set the model to evaluation mode

# Trace the model with an example input
input_shape: Tuple[int, ...] = (1, 3, 224, 224)  # Adjust input shape for your task
example_input = torch.rand(input_shape)
traced_model = torch.jit.trace(model, example_input)

# Compile model on a specific device
compile_job = hub.submit_compile_job(
    traced_model,
    name="LizzyTheCat",  # Name your model
    device=hub.Device("Samsung Galaxy S24 (Family)"),  # Target device (e.g., Galaxy S24)
    input_specs=dict(image=input_shape),  # Specify the input shape
)

# Ensure that the compile job was submitted successfully
assert isinstance(compile_job, hub.CompileJob)

print("Model successfully submitted for compilation.")
