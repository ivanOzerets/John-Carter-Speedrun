import torch
import torch.nn as nn
import wandb
import os
from src.pytorch.model import get_pretrained_model
from src.pytorch.data import get_dataloaders
from src.pytorch.trainer import train_strategy, evaluate

EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
PATIENCE = 10
WEIGHT_DECAY = 1e-4

sweep_config = {
    "name": "train_transfer_sweep_p2",
    "method": "grid",
    "metric": {
        "name": "val/acc",
        "goal": "maximize"
    },
    "parameters": {
        "model_name": {"values": ["efficientnet_b3"]},
        "data_size": {"values": ["full"]},
        "strategy": {"values": ["gradual", "finetune"]},
        "learning_rate": {"value": LEARNING_RATE},
        "dropout": {"value": DROPOUT_RATE},
        "batch_size": {"value": BATCH_SIZE},
        "num_classes": {"value": NUM_CLASSES},
        "epochs": {"value": EPOCHS},
        "weight_decay": {"value": WEIGHT_DECAY},
        "patience": {"value": PATIENCE}
    }
}

def sweep_train():
    '''
    Main training loop for the sweep. Initializes the model, dataloaders, and starts training according to the strategy.
    '''
    wandb.init(project="nature-classifier", entity="ivan.ozerets")
    config = wandb.config
    wandb.run.name = f"{config.model_name}_{config.strategy}_{config.data_size}"

    save_path = os.path.join("models", f"transfer_{wandb.run.name}.pt")

    data_dirs = {
        "small": os.path.join("data", "split_small"),
        "medium": os.path.join("data", "split_medium"),
        "full": os.path.join("data", "split")
    }
    processed_dir = data_dirs[config.data_size]

    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir,
        config.batch_size,
        NORMALIZE_MEAN,
        NORMALIZE_STD,
        test_dir=os.path.join("data", "split", "test")
    )

    model = get_pretrained_model(
        model_name = config.model_name,
        num_classes = config.num_classes,
        dropout_rate = config.dropout,
        freeze=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    train_strategy(model, config, train_loader, val_loader, criterion, DEVICE, save_path)

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="nature-classifier", entity="ivan.ozerets")
    wandb.agent(sweep_id, function=sweep_train, count=18)