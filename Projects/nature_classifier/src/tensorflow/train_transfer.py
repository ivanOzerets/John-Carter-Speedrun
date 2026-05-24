import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger
from src.tensorflow.model import get_pretrain_model, unfreeze_model, get_gradual_layers
from src.tensorflow.data import get_datasets

EPOCHS = 100
BATCH_SIZE = 32
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
PATIENCE = 10
WEIGHT_DECAY = 1e-4

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "model_name": {"values": ["resnet50"]},
        "data_size": {"values": ["full"]},
        "strategy": {"values": ["gradual"]},
        "learning_rate": {"value": LEARNING_RATE},
        "dropout": {"value": DROPOUT_RATE},
        "batch_size": {"value": BATCH_SIZE},
        "num_classes": {"value": NUM_CLASSES},
        "epochs": {"value": EPOCHS},
        "weight_decay": {"value": WEIGHT_DECAY},
        "patience": {"value": PATIENCE}
    }
}

def compile_and_fit(model, train_dataset, val_dataset, config, save_path, lr, initial_epoch=0):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True), 
        WandbMetricsLogger(), 
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=config.weight_decay)
    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks, initial_epoch=initial_epoch)

    return initial_epoch + len(history.history['loss'])

def sweep_train():
    '''
    Main training loop for the sweep. Initializes the model, dataloaders, and starts training according to the strategy.

    Strategies:
    - frozen: train classification head only, backbone frozen
    - gradual: staged fine-tuning - train head first, then gradually unfreeze
               the first few layers until convergence at each stage, reducing
               the learning rate at each stage.
    - finetune: unfreeze all backbone layers with reduced learning rate
    '''
    wandb.init(project="nature-classifier", entity="ivan.ozerets")
    config = wandb.config
    wandb.run.name = f"transfer_{config.strategy}_{config.data_size}"

    save_path = os.path.join("models", f"transfer_{wandb.run.name}.keras")

    data_dirs = {
        "small": os.path.join("data", "split_small"),
        "medium": os.path.join("data", "split_medium"),
        "full": os.path.join("data", "split")
    }
    processed_dir = data_dirs[config.data_size]

    train_dataset, val_dataset, test_dataset = get_datasets(
        processed_dir,
        config.batch_size,
        NORMALIZE_MEAN,
        NORMALIZE_STD,
        test_dir=os.path.join("data", "split", "test")
    )

    model = get_pretrain_model(num_classes=NUM_CLASSES, dropout_rate=config.dropout, freeze=True)

    if config.strategy == "frozen":
        _ = compile_and_fit(model, train_dataset, val_dataset, config, save_path, config.learning_rate)
    elif config.strategy == "gradual":
        layer_groups = get_gradual_layers(config.model_name)

        for i, group in enumerate(layer_groups):
            unfreeze_model(model, group, backbone_name=config.model_name)
            lr = config.learning_rate / (10 ** i)
            epochs_completed = compile_and_fit(model, train_dataset, val_dataset, config, save_path, lr, initial_epoch=epochs_completed)
    elif config.strategy == "finetune":
        model.get_layer(config.model_name).trainable = True
        _ = compile_and_fit(model, train_dataset, val_dataset, config, save_path, config.learning_rate/10)

    test_loss, test_acc = model.evaluate(test_dataset)
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="nature-classifier", entity="ivan.ozerets")
    wandb.agent(sweep_id, function=sweep_train, count=3)