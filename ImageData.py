import functools
import random

import tensorflow as tf

from config import get_config_from_json


def decode_png(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 127.5 - 1
    return img


def process_path(img_path, label):
    img = tf.io.read_file("faces_images/" + img_path)
    img = decode_png(img)
    return img, label


def process_only_path(img_path):
    img = tf.io.read_file("faces_images/" + img_path)
    img = decode_png(img)
    return img


def resize_and_crop(img, origin_shape, resize_size, seed):
    image = tf.image.resize(img, size=(resize_size, resize_size))
    image = tf.image.random_crop(image, origin_shape, seed=seed)
    return image


class ImageData:
    def __init__(self, config):
        self.config = config

    def process_path(self, img_path, label):
        img = tf.io.read_file("faces_images/" + img_path)
        img = decode_png(img)

        return img, label

    def augmentation(self, image, label):
        if random.random() > 0.75:
            seed = random.randint(0, 2 ** 31 - 1)
            ori_image_shape = tf.shape(image)
            augment_list = [
                functools.partial(tf.image.random_flip_left_right, seed=seed),
                functools.partial(tf.keras.preprocessing.image.random_rotation, rg=30.),
                functools.partial(
                    resize_and_crop, origin_shape=ori_image_shape,
                    resize_size=self.config.data.resize_size, seed=seed),
                functools.partial(
                    tf.keras.preprocessing.image.random_shear, intensity=15
                ),
                functools.partial(tf.keras.preprocessing.image.random_brightness, brightness_range=(0.8, 1))
            ]

            for augment in augment_list:
                p = random.random()
                if p < 0.5:
                    image = augment(image)

        return image, label


if __name__ == "__main__":
    config = get_config_from_json("config.json")
    image_data = ImageData(config)
    img = tf.ones((128, 128, 3))
    label = tf.zeros((6,))
    augmented_img, label = image_data.augmentation(img, label)
