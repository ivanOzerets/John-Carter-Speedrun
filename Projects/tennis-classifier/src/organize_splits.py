import shutil
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

RAW_IMAGES_DIR = os.path.join("data", "raw", "images")
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["forehand", "backhand", "serve", "volley", "overhead"]

def load_paths():
    """
    Load image file paths and labels from the raw images directory.
    Returns a list of image paths and their corresponding labels.
    """
    paths = []
    labels = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(RAW_IMAGES_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(class_dir, filename)
                paths.append(image_path)
                labels.append(class_name)

    return paths, labels

def split_data(paths, labels):
    """
    Split the dataset into training, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=1-TRAIN_RATIO, stratify=labels, random_state=42)

    val_test_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_test_ratio, stratify=y_temp, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_split(paths, labels, split_name):
    """
    Copy images into train/val/test split folders under the processeed directory.
    """
    for path, label in zip(paths, labels):
        output_dir = os.path.join(PROCESSED_DIR, split_name, label)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(path, output_dir)

if __name__ == "__main__":
    print("Loading image paths and labels...")
    paths, labels = load_paths()

    print("Splitting data into train/val/test...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(paths, labels)

    print("Saving training set...")
    save_split(X_train, y_train, "train")

    print("Saving validation set...")
    save_split(X_val, y_val, "val")

    print("Saving test set...")
    save_split(X_test, y_test, "test")

    print("Done")