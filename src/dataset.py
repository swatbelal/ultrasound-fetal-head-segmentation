import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

import config as cfg


def load_image_mask(img_path, mask_path, img_size=(256, 256)):
    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)

    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size)
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask


def get_datasets(val_split=0.2, batch_size=cfg.BATCH_SIZE, augment=False):
    img_paths = sorted(list(Path(cfg.TRAIN_IMG_DIR).glob("*.png")))
    mask_paths = sorted(list(Path(cfg.TRAIN_MASK_IMG_DIR).glob("*.png")))

    # Train/Val split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        img_paths, mask_paths, test_size=val_split, random_state=42
    )

    def gen(img_list, mask_list):
        for img_path, mask_path in zip(img_list, mask_list):
            img, mask = load_image_mask(img_path, mask_path, cfg.IMG_SIZE)
            yield img, mask

    # TF datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: gen(train_imgs, train_masks),
        output_signature=(
            tf.TensorSpec(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1), dtype=tf.float32),
        )
    ).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: gen(val_imgs, val_masks),
        output_signature=(
            tf.TensorSpec(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1), dtype=tf.float32),
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
