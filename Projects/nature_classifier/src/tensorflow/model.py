import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

class SimpleCNN(Model):
    def __init__(self, num_conv_layers=3, base_filters=32, num_classes=6, dropout_rate=0.3, input_shape=(224, 224, 3)):
        super().__init__()
        
        self.conv_layers = []
        self.bn_layers = []
        out_channels = base_filters

        for _ in range(num_conv_layers):
            self.conv_layers.append(layers.Conv2D(out_channels, kernel_size=3, padding='same'))
            self.bn_layers.append(layers.BatchNormalization())
            out_channels *= 2

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(num_classes)

        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)

        return x

def get_pretrain_model(num_classes, dropout_rate=0.5, freeze=True):
    '''
    Returns a pre-trained model from Keras Applications with the specified number of classes and dropout rate.
    '''
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = not freeze

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes)(x)

    model = Model(inputs, outputs)

    return model

def unfreeze_model(model, num_unfreeze):
    '''
    Unfreezes the last `num_layers_to_unfreeze` layers of the model for fine-tuning.
    '''
    if num_unfreeze <= 0:
        print("No layers to unfreeze.")
        return

    for layer in model.layers[-num_unfreeze:]:
        layer.trainable = True