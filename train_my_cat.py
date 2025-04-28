import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

# Settings
DATA_DIR = "dataset/"  # Path to your already split dataset
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "best_cat_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.ImageFolder(DATA_DIR + "train/", transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR + "val/", transform=val_test_transform)
test_dataset = datasets.ImageFolder(DATA_DIR + "test/", transform=val_test_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Function to initialize and return the model
def initialize_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
model = initialize_model()  # Initialize the model here

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Mixed precision
scaler = torch.amp.GradScaler()

# Early stopping
PATIENCE = 5
best_val_loss = float('inf')  # Initialize best_val_loss here
patience_counter = 0

def train_and_evaluate(model):  # Make sure the model is passed to this function
    global best_val_loss  # Make sure to reference the global variable

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")

        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # Corrected the use of autocast
            with torch.amp.autocast(device_type=DEVICE.type):  # Specify the device type
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Use autocast and scale the loss
            scaler.scale(loss).backward()

            # Explicitly check for NaN/Inf
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                print(f"Warning: NaN or Inf loss detected. Skipping batch.")
                continue  # Skip this batch if NaN/Inf is detected

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved new best model: {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    print("🎉 Training finished!")

    # Now we test the model

    # Load the trained model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    # Testing loop
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    train_and_evaluate(model)  # Pass the initialized model here
