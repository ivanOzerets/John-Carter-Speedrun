import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os


def get_transforms(mean, std):
    """
    Define the image transformations for training and validation/test datasets.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def get_dataloaders(processed_dir, batch_size, mean, std):
    """
    Create DataLoaders for training, validation, and test datasets.
    """
    train_transform, val_transform = get_transforms(mean, std)
    train_dataset = datasets.ImageFolder(os.path.join(processed_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(processed_dir, "val"), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(processed_dir, "test"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader