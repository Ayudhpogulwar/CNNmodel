
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# --- 1. Model Definition (Must be the same as trained model) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Our input images are 250x250. After two MaxPool2d layers with stride 2,
        # the spatial dimensions become 250 / 2 / 2 = 62.5, which is 62 due to floor operation.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. Configuration Parameters (Must match training setup) ---
TARGET_SIZE = (250, 250)
CLASS_NAMES = ['fork', 'spoon'] # Ensure this matches your dataset classes
MODEL_PATH = 'model_spoon_fork.pth'

# --- 3. Load Model and Define Transformations ---
@st.cache_resource # Cache the model loading to avoid re-loading on every rerun
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Load state dict, mapping to CPU if necessary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Transformations for input image (must match test_transforms from training)
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory.")
    st.stop()

model, device = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

# --- 4. Streamlit App Interface ---
st.title("Spoon/Fork Image Classifier")
st.write("Upload an image to classify it as a spoon or a fork!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch
    input_batch = input_batch.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    predicted_class = CLASS_NAMES[predicted_idx.item()]

    st.success(f"Prediction: This is a **{predicted_class}**!")

