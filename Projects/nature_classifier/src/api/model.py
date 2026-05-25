import torch
from torchvision import transforms
from PIL import Image
import io
from src.pytorch.model import get_pretrained_model

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
STANDARDIZE_MEAN = [0.485, 0.456, 0.406]
STANDARDIZE_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=STANDARDIZE_MEAN, std=STANDARDIZE_STD)
])

def load_model(model_path):
    """
    Load the model from the specified path.
    """
    model = get_pretrained_model(model_name="resnet50", num_classes=6, dropout_rate=0.5, freeze=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model

def predict(model, image_bytes):
    """
    Predict the class of the input image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    scores = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))}
    predicted = max(scores, key=scores.get)

    return {"class": predicted, "confidence": scores[predicted], "scores": scores}