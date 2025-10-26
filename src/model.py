#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models


class BinaryMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, name="mean_io_u", dtype=None):
        super().__init__(num_classes=2, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # threshold probabilities -> integers (0/1) and flatten
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        return super().update_state(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]), sample_weight)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # soft dice (works with probabilities) -- good for loss
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    # combine BCE + soft Dice
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name="dice_metric", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_f = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_bin)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_bin)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        self.total.assign_add(dice)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 1e-12)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# --- U-Net model with BatchNorm + Dropout ----------------------------------
def conv_block(x, filters, kernel_size=3, dropout_rate=0.0):
    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate and dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def unet_model(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256, dropout_rate=0.3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = conv_block(p3, 512, dropout_rate=0.5)

    # Decoder
    u1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(b)
    u1 = layers.concatenate([u1, c3])
    c4 = conv_block(u1, 256)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, 128)

    u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = conv_block(u3, 64)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c6)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
