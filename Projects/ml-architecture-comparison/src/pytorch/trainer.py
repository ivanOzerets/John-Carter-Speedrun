import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from src.pytorch.model import unfreeze_layers, get_gradual_layers

def optimizer_and_train(model, config, train_loader, val_loader, criterion, device, save_path, lr, epoch_offset=0):
    """
    Set up the optimizer and train the model according to the specified strategy.
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=config.weight_decay
    )

    return train(model, train_loader, val_loader, criterion, optimizer,
        config.epochs, device, save_path, epoch_offset=epoch_offset, patience=config.patience
    )

def train_strategy(model, config, train_loader, val_loader, criterion, device, save_path):
    '''
    Train the model according to the specified strategy.

    Strategies:
    - frozen: train classification head only, backbone frozen
    - gradual: staged fine-tuning - train head first, then gradually unfreeze
               the first few layers until convergence at each stage, reducing
               the learning rate at each stage.
    - finetune: unfreeze all backbone layers with reduced learning rate
    '''

    if config.strategy == "finetune":
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer_and_train(model, config, train_loader, val_loader, criterion, device, save_path, config.learning_rate/10)
        return

    # frozen run
    completed_epochs = optimizer_and_train(model, config, train_loader, val_loader, criterion, device, save_path, config.learning_rate)

    if config.strategy == "gradual":
        layer_groups = get_gradual_layers(config.model_name)

        for i, phase_layers in enumerate(layer_groups):
            unfreeze_layers(model, phase_layers)
            lr = config.learning_rate / (10 ** (i + 1))
            
            epochs_run = optimizer_and_train(model, config, train_loader, val_loader, criterion, device, save_path, lr, epoch_offset=completed_epochs)
            completed_epochs += epochs_run

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path, epoch_offset=0, patience=10):
    """
    Train the model and evaluate on the validation set after each epoch.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = 0.0
        train_correct, train_total = 0, 0
        
        model.train()
        for inputs, labels, in tqdm(train_loader, desc=f"Epoch {epoch+1+epoch_offset}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1+epoch_offset}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "epoch": epoch + 1 + epoch_offset
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1 + epoch_offset}")
                wandb.log({"early_stop": True, "stop_at_epoch": epoch + 1 + epoch_offset})
                break
    else:
        wandb.log({"early_stop": False, "stop_at_epoch": epoch + 1 + epoch_offset})

    return epoch + 1
            
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on the input loader.
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = total_loss / len(loader)
    acc = correct / total

    return loss, acc