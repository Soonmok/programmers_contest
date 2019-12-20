import functools
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from ImageData import process_path, process_only_path, resize_and_crop, get_augmentation_list, augmentation
from config import get_config_from_json


def get_resampled_dataset(img_paths, labels):
    total_size = len(img_paths)
    indices_list = []
    data_weight_list = []
    datasets = []
    for i in range(6):
        indices = np.where(labels == i)[0]
        data_weight_list.append(1 / (len(indices) / total_size))
        dataset = tf.data.Dataset.from_tensor_slices((img_paths[indices], labels[indices])).repeat()
        indices_list.append(indices)
        datasets.append(dataset)
    data_weight_list = list(map(lambda x: x / sum(data_weight_list), data_weight_list))
    resampled_ds = tf.data.experimental.sample_from_datasets(datasets, weights=[0.16, 0.16, 0.16, 0.16, 0.16, 0.16, ])
    # resampled_ds = tf.data.experimental.sample_from_datasets(datasets[:2], weights=[0.1, 0.9])
    return resampled_ds


def get_dataset(config, img_paths, labels, dev=False):
    if dev:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    else:
        dataset = get_resampled_dataset(img_paths, labels)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(config.data.shuffle_batch)
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if not dev and config.data.aug:
        augmentation_list = get_augmentation_list()
        for augment in augmentation_list:
            dataset = dataset.map(functools.partial(augmentation, augment=augment), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=config.data.batch_size)
    return dataset


def get_test_dataset(config, img_paths):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(process_only_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=config.data.batch_size)
    return dataset


def get_train_data_loader(config, csv_path):
    train_img_paths, dev_img_paths, train_labels, dev_labels = get_paths(config, csv_path)
    train_dataset = get_dataset(config, train_img_paths, train_labels)
    dev_dataset = get_dataset(config, dev_img_paths, dev_labels, dev=True)
    return train_dataset, dev_dataset


def get_test_data_loader(config, csv_path):
    csv_file = pd.read_csv(csv_path)
    test_img_paths = csv_file['filename'].to_numpy()
    test_dataset = get_test_dataset(config, test_img_paths)
    return test_dataset


def get_data(config, img_paths, labels, dev=False):
    if dev:
        dataset = get_dataset(config, img_paths, labels, dev=True).repeat()
    else:
        dataset = get_dataset(config, img_paths, labels).repeat()

    return dataset


def get_paths(config, csv_path):
    csv_file = pd.read_csv(csv_path)
    img_paths = csv_file['filename'].to_numpy()
    labels = csv_file['label'].to_numpy()
    labels -= 1  # to set start index to zero
    indices = np.random.permutation(img_paths.shape[0])
    train_idx, dev_idx = indices[len(labels) // 10:], indices[:len(labels) // 10]
    train_img_paths, dev_img_paths = img_paths[train_idx], img_paths[dev_idx]
    train_labels, dev_labels = labels[train_idx], labels[dev_idx]
    config.data.dev_data_size = len(dev_idx)
    return train_img_paths, dev_img_paths, train_labels, dev_labels



if __name__ == "__main__":
    config = get_config_from_json("config.json")
    train_data_loader, dev_data_loader = get_train_data_loader(config, "train_vision.csv")
    # seed = random.randint(0, 2 ** 31 - 1)
    # augment_list = [
    #     functools.partial(tf.image.random_flip_left_right, seed=seed),
    #     functools.partial(tf.keras.preprocessing.image.random_rotation, rg=30.),
    #     functools.partial(
    #         resize_and_crop, origin_shape=(128, 128, 3),
    #         resize_size=config.data.resize_size, seed=seed),
    #     functools.partial(
    #         tf.keras.preprocessing.image.random_shear, intensity=15
    #     ),
    #     functools.partial(tf.keras.preprocessing.image.random_brightness, brightness_range=(0.8, 1))
    # ]

    for train_x, y in train_data_loader:
        img = train_x[0]
        label = y[0]
        augment_list = get_augmentation_list()
        for augment in augment_list:
            img, _ = augmentation(img, label, augment)
            print(img.shape)
        # unique, counts = np.unique(y.numpy(), return_counts=True)
        # print(dict(zip(unique, counts)))
    # test_data_loader = get_test_data_loader(config, "test_vision.csv")
    # for test_img_path in test_data_loader:
    #     print(test_img_path)
