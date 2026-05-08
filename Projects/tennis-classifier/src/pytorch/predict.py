import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from src.pytorch.model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["forehand", "backhand", "serve", "volley", "overhead"]
MODEL_PATH = os.path.join("models", "best_scratch_cnn.pt")
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

def predict(image_path):
    print("Starting prediction...")
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[0][predicted_index].item()

    print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    predict(image_path)