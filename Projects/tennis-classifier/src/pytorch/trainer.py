import torch
import wandb
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path, epoch_offset=0, total_epochs=10):
    """
    Train the model and evaluate on the validation set after each epoch.
    """
    best_val_loss = float("inf")

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
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

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