import random

import tensorflow as tf


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


class ImageData:
    def __init__(self, config):
        self.augment_flag = config.aug

    def process_path(self, img_path, label):
        img = tf.io.read_file("faces_images/" + img_path)
        img = decode_png(img)

        if self.augment_flag:
            p = random.random()
            if p > 0.5:
                img = self.augmentation(img)

        return img, label

    def augmentation(self, image):
        seed = random.randint(0, 2 ** 31 - 1)
        ori_image_shape = tf.shape(image)
        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.resize_images(image, [self.augment_size, self.augment_size])
        image = tf.random_crop(image, ori_image_shape, seed=seed)
        image = tf.image.random_rotation(image, 30)
        return image