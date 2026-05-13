import torch
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
PROCESSED_DIR = os.path.join("data", "split")
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, lbls in tqdm(loader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
            
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels