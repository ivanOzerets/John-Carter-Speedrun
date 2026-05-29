import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import wandb
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier
from src.pytorch.data import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
PROCESSED_DIR = os.path.join("data", "split")
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32

def get_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model.to(DEVICE)

def extract_features(model, dataloader):
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels 

def run_experiments():
    print("Starting sklearn experiments...")
    wandb.init(project="nature-classifier", entity="ivan.ozerets")
    wandb.run.name = "train_sklearn_experiments"

    extractor = get_feature_extractor()

    train_loader, _, test_loader = get_dataloaders(
        PROCESSED_DIR, BATCH_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, 
        test_dir=os.path.join("data", "split", "test"),
        combine_train_val=True)

    train_features, train_labels = extract_features(extractor, train_loader)
    test_features, test_labels = extract_features(extractor, test_loader)

    classifiers = {
        "SVM": SVC(kernel="rbf", probability=True),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(n_estimators=100)
    }

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(train_features, train_labels)
        test_acc = clf.score(test_features, test_labels)
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        wandb.log({f"{name}/test_acc": test_acc})
    
    wandb.finish()

if __name__ == "__main__":
    run_experiments()