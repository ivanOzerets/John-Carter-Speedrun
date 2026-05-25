import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

def get_gradual_layers(model_name):
    '''
    Returns layer groups for gradual unfreezing based on the model architecture.
    '''
    if model_name == "resnet50v2":
        return ["conv5", "conv4", "conv3"]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def SimpleCNN(num_conv_layers=3, base_filters=32, num_classes=6, dropout_rate=0.5):
    '''
    Returns a simple CNN model with the specified input shape, number of classes, and dropout rate
    '''
    model_layers = []
    out_channels = base_filters
    
    for _ in range(num_conv_layers):
        model_layers.extend([
            layers.Conv2D(out_channels, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        out_channels *= 2

    model_layers.extend([
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes)
    ])    

    return keras.Sequential(model_layers, name='SimpleCNN')

def get_pretrained_model(num_classes, dropout_rate=0.5, freeze=True):
    '''
    Returns a pre-trained model from Keras Applications with the specified number of classes and dropout rate.
    '''
    base_model = keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = not freeze

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes)(x)

    model = Model(inputs, outputs)

    return model

def unfreeze_model(model, layer_name, backbone_name="resnet50v2"):
    '''
    Unfreezes the specified layers of the model for fine-tuning.
    '''
    backbone = model.get_layer(backbone_name)
    for layer in backbone.layers:
        if layer_name in layer.name:
            layer.trainable = True