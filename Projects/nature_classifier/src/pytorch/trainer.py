import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from src.pytorch.model import unfreeze_layers, get_gradual_layers

def train_strategy(model, config, train_loader, val_loader, criterion, device, save_path):
    '''
    Train the model according to the specified strategy.
    '''
    if config.strategy == "frozen":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
        train(model, train_loader, val_loader, criterion, optimizer, config.epochs, device, save_path, patience=config.patience)
    elif config.strategy == "gradual":
        layer_groups = get_gradual_layers(config.model_name)

        for i, phase_layers in enumerate(layer_groups):
            unfreeze_layers(model, phase_layers)
            lr = config.learning_rate / (10 ** i)
            phase_epochs = config.epochs // len(layer_groups)
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.weight_decay)
            train(model, train_loader, val_loader, criterion, optimizer,
                  phase_epochs, device, save_path, 
                  epoch_offset=i*phase_epochs, 
                  total_epochs=config.epochs, 
                  patience=config.patience)
    elif config.strategy == "finetune":
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate/10, weight_decay=config.weight_decay)
        train(model, train_loader, val_loader, criterion, optimizer, config.epochs, device, save_path, patience=config.patience)

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path, epoch_offset=0, total_epochs=None, patience=10):
    """
    Train the model and evaluate on the validation set after each epoch.
    """
    if total_epochs is None:
        total_epochs = epochs + epoch_offset

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = 0.0
        train_correct, train_total = 0, 0
        
        model.train()
        for inputs, labels, in tqdm(train_loader, desc=f"Epoch {epoch+1+epoch_offset}/{total_epochs}"):
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

        print(f"Epoch {epoch+1+epoch_offset}/{total_epochs}")
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
        wandb.log({"early_stop": False, "stop_at_epoch": total_epochs})
            
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