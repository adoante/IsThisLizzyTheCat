import qai_hub as hub
from PIL import Image
import numpy as np
import h5py

# Preprocess image based on AI model input spec
def preprocess_image(image_path):
    # Open image and account for grey scale
    image = Image.open(image_path).convert("RGB")

    # All image classification models require a size of 224x224
    image = image.resize((224, 224)) 
	
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # All image classification models require a batch dimension of 1
    image = np.expand_dims(image, axis=0)

    # Rearrange dimensions to (1, 3, 224, 224) from (1, 224, 224, 3)
    image = np.transpose(image, (0, 3, 1, 2))  # Change from (1, 224, 224, 3) to (1, 3, 224, 224)

    return image

image_path = "images/PXL_20240201_163549345.jpg"

# Preprocess image
processed_image = preprocess_image(image_path)

# Prepare the input data as a dictionary
inputs = {"image": [processed_image]}

# Submit inference job
inference_job = hub.submit_inference_job(
    model = hub.get_model("mn70ejl8q"),
    device = hub.Device("Samsung Galaxy S24 (Family)"),
    inputs = inputs,
)

# Ensure the job is valid
assert isinstance(inference_job, hub.InferenceJob)

h5_file_path = "results_dataset.h5"

# Download inference results dataset to specific folder
inference_job.download_output_data(h5_file_path)

# Open the H5 file
with h5py.File(h5_file_path, "r") as f:
	# Access the 'data' group
	data_group = f['data']

	# Access the '0' group
	group_0 = data_group['0']

	# Access the dataset
	batch_0_data = group_0['batch_0'][:]
		
# Apply softmax to convert logits to probabilities
def softmax(logits):
	exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
	return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

# Get probabilities
probabilities = softmax(batch_0_data)

# Find the predicted class (index with highest probability)
predicted_class = np.argmax(probabilities)

class_labels = ["lizzy_the_cat", "not_lizzy_the_cat"]

print(f"Raw Output: {batch_0_data}")
print(f"Probabilities: {round(probabilities[0][0] * 100, 2)}, {round(probabilities[0][1] * 100, 2)}")
print("Predicted Class:", predicted_class)
print("Class Label:", class_labels[predicted_class])