import functools
import random

import tensorflow as tf
import tensorflow_addons as tfa

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


def resize_and_crop(img: tf.Tensor):
    seed = random.randint(0, 2 ** 31 - 1)
    image = tf.image.resize(img, size=(196, 196))
    image = tf.image.random_crop(image, (128, 128, 3), seed=seed)
    return image


def flip_left_right(img: tf.Tensor):
    seed = random.randint(0, 2 ** 31 - 1)
    img = tf.image.random_flip_left_right(img, seed=seed)
    return img


def random_rotation(img: tf.Tensor):
    angle = random.randrange(-30, 30)
    img = tfa.image.rotate(img, angle)
    return img


def random_shear(img: tf.Tensor):
    img = tf.keras.preprocessing.image.random_shear(img, intensity=15)
    return img


def random_brihtness(img: tf.Tensor):
    img = tf.keras.preprocessing.image.random_brightness(img, brightness_range=(0.8, 1))
    return img


def get_augmentation_list():
    return [resize_and_crop, flip_left_right, random_rotation]


@tf.function
def augmentation(img: tf.Tensor, label: tf.Tensor, augment=None):
    img = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: augment(img), lambda: img)
    return img, label


class ImageData:
    def __init__(self, config):
        self.config = config

    def process_path(self, img_path, label):
        img = tf.io.read_file("faces_images/" + img_path)
        img = decode_png(img)

        return img, label

    def augmentation(self, image, label):
        # if random.random() > 0.75:
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
            # if p < 0.5:
            image = augment(image)

        return image, label


if __name__ == "__main__":
    config = get_config_from_json("config.json")
    image_data = ImageData(config)
    img = tf.ones((128, 128, 3))
    label = tf.zeros((6,))
    augmented_img, label = image_data.augmentation(img, label)
