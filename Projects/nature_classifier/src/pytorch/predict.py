import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from src.pytorch.model import SimpleCNN, get_resnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["forehand", "backhand", "serve", "volley", "overhead"]
NORMALIZE = {
    "scratch": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "frozen": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "gradual": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
}
MODEL_PATHS = {
    "scratch": os.path.join("models", "best_scratch_cnn.pt"),
    "frozen": os.path.join("models", "best_frozen_resnet50.pt"),
    "gradual": os.path.join("models", "best_gradual_resnet50.pt")
}

def transform_image(image_path, model_type):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE[model_type]["mean"],
            std=NORMALIZE[model_type]["std"]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def predict(image_path, model_type="scratch"):
    print("Starting prediction...")

    if model_type not in MODEL_PATHS:
        print(f"Error: model_type must be one of {list(MODEL_PATHS.keys())}")
        sys.exit(1)

    if model_type == "scratch":
        model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    else:
        model = get_resnet50(num_classes=len(CLASS_NAMES)).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATHS[model_type], map_location=DEVICE))
    model.eval()

    input_tensor = transform_image(image_path, model_type)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[0][predicted_index].item()

    print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <model_type>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_type = sys.argv[2]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    predict(image_path, model_type)