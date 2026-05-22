import tensorflow as tf
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from src.tensorflow.model import SimpleCNN
from src.tensorflow.data import get_datasets

EPOCHS = 100
BATCH_SIZE = 32
PROCESSED_DIR = os.path.join("data", "split")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]
PATIENCE = 10

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "val/acc",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {"values": [0.001]},
        "num_conv_layers": {"values": [5]},
        "base_filters": {"values": [64]},
        "dropout": {"values": [0.5]},
        "weight_decay": {"values": [0]},
        "epochs": {"value": EPOCHS},
        "batch_size": {"value": BATCH_SIZE},
        "num_classes": {"value": NUM_CLASSES}
    }
}

def sweep_train():
    wandb.init(project="nature-classifier", entity="ivan.ozerets")
    config = wandb.config
    wandb.run.name = f"lr{config.learning_rate}_depth{config.num_conv_layers}_filters{config.base_filters}_drop{config.dropout}_wd{config.weight_decay}"

    save_path = os.path.join("models", f"scratch_{wandb.run.name}.keras")

    train_dataset, val_dataset, test_dataset = get_datasets(PROCESSED_DIR, BATCH_SIZE, NORMALIZE_MEAN, NORMALIZE_STD)
    model = SimpleCNN(num_conv_layers=config.num_conv_layers, base_filters=config.base_filters, num_classes=NUM_CLASSES, dropout_rate=config.dropout)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    wandb_callback = WandbMetricsLogger()
    model_checkpoint = WandbModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=EPOCHS,
              callbacks=[early_stopping, wandb_callback, model_checkpoint])

    test_loss, test_acc = model.evaluate(test_dataset)
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="nature-classifier", entity="ivan.ozerets")
    wandb.agent(sweep_id, function=sweep_train, count=1)