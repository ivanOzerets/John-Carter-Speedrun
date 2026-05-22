import tensorflow as tf
import os

def get_datasets(processed_dir, batch_size, mean, std, test_dir=None, combine_train_val=False):
    """
    Create datasets for training, validation, and test datasets.
    """
    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean) / std

        return image, label
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(processed_dir, "train"),
        image_size=(224, 224),
        batch_size=batch_size,
    ).map(normalize)

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(processed_dir, "val"),
        image_size=(224, 224),
        batch_size=batch_size,
    ).map(normalize)

    test_path = test_dir if test_dir else os.path.join(processed_dir, "test")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=(224, 224),
        batch_size=batch_size,
    ).map(normalize)

    if combine_train_val:
        train_dataset = train_dataset.concatenate(val_dataset)
        val_dataset = None
    
    return train_dataset, val_dataset, test_dataset