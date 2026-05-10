import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from src.pytorch.model import get_resnet50
from src.pytorch.data import get_dataloaders
from src.pytorch.trainer import train, evaluate

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PROCESSED_DIR = os.path.join("data", "split")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
SAVE_PATH = os.path.join("models", "best_frozen_resnet50.pt")

if __name__ == "__main__":
    wandb.init(project="nature_scenes", name="train_frozen_resnet50", entity="ivan.ozerets", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "model": "ResNet50 Frozen"
    })

    model = get_resnet50(num_classes=NUM_CLASSES, freeze=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader, test_loader = get_dataloaders(PROCESSED_DIR, BATCH_SIZE, NORMALIZE_MEAN, NORMALIZE_STD)
    train(model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE, SAVE_PATH)

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})

    wandb.finish()