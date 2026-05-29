import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset, DataLoader
import os

def get_transforms(mean, std):
    """
    Define the image transformations for training and validation/test datasets.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform

def get_dataloaders(processed_dir, batch_size, mean, std, test_dir=None, combine_train_val=False):
    """
    Create DataLoaders for training, validation, and test datasets.
    """
    transform = get_transforms(mean, std)
    train_dataset = datasets.ImageFolder(os.path.join(processed_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(processed_dir, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(test_dir if test_dir else os.path.join(processed_dir, "test"), transform=transform)
    
    if combine_train_val:
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = None if combine_train_val else DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader