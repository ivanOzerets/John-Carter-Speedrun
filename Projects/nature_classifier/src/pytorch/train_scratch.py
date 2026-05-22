import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from src.pytorch.model import SimpleCNN
from src.pytorch.data import get_dataloaders
from src.pytorch.trainer import train, evaluate

EPOCHS = 100
BATCH_SIZE = 32
PROCESSED_DIR = os.path.join("data", "split")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val/acc",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {"values": [0.001, 0.0005, 0.0001]},
        "num_conv_layers": {"values": [4, 5, 6]},
        "base_filters": {"values": [32, 64, 128]},
        "dropout": {"values": [0.3, 0.5, 0.7]},
        "weight_decay": {"values": [0, 1e-4, 1e-3]},
        "epochs": {"value": EPOCHS},
        "batch_size": {"value": BATCH_SIZE},
        "num_classes": {"value": NUM_CLASSES}
    }
}

def sweep_train():
    wandb.init(project="nature-classifier", entity="ivan.ozerets")
    config = wandb.config
    wandb.run.name = f"lr{config.learning_rate}_depth{config.num_conv_layers}_filters{config.base_filters}_drop{config.dropout}_wd{config.weight_decay}"

    save_path = os.path.join("models", f"scratch_{wandb.run.name}.pt")

    train_loader, val_loader, test_loader = get_dataloaders(PROCESSED_DIR, BATCH_SIZE, NORMALIZE_MEAN, NORMALIZE_STD)
    model = SimpleCNN(num_conv_layers=config.num_conv_layers, base_filters=config.base_filters, num_classes=NUM_CLASSES, dropout_rate=config.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    _ = train(model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE, save_path)

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="nature-classifier", entity="ivan.ozerets")
    wandb.agent(sweep_id, function=sweep_train, count=30)