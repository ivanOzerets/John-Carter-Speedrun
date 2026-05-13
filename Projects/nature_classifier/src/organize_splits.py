import shutil
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

RAW_IMAGES_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "split")
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def load_paths(max_per_class=None):
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

        class_paths = []
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                class_paths.append(os.path.join(class_dir, filename))

        if max_per_class is not None:
            class_paths = class_paths[:max_per_class]

        paths.extend(class_paths)
        labels.extend([class_name] * len(class_paths))

    return paths, labels

def split_data(paths, labels):
    """
    Split the dataset into training and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, 
        test_size=VAL_RATIO, 
        stratify=labels, 
        random_state=42)

    return (X_train, y_train), (X_val, y_val)

def save_split(paths, labels, split_name, output_dir=PROCESSED_DIR):
    """
    Copy images into train/val/test split folders under the processeed directory.
    """
    for path, label in tqdm(zip(paths, labels), total=len(paths), desc=f"Saving {split_name}"):
        output_name = os.path.join(output_dir, split_name, label)
        os.makedirs(output_name, exist_ok=True)
        shutil.copy(path, output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join("data", "split"),
                        help="Where to save the split data")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Max images per class (None = all)")
    args = parser.parse_args()

    print("Loading image paths and labels...")
    paths, labels = load_paths(max_per_class=args.max_per_class)

    print("Splitting data into train and val sets...")
    (X_train, y_train), (X_val, y_val) = split_data(paths, labels)

    print("Saving training set...")
    save_split(X_train, y_train, "train", output_dir=args.output_dir)

    print("Saving validation set...")
    save_split(X_val, y_val, "val", output_dir=args.output_dir)

    print("Done")