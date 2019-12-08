import tensorflow as tf
import pandas as pd
import numpy as np

from ImageData import ImageData, process_path, process_only_path
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
    data_weight_list = list(map(lambda x: x/sum(data_weight_list), data_weight_list))
    resampled_ds = tf.data.experimental.sample_from_datasets(datasets, weights=[0.16, 0.16, 0.16, 0.16, 0.16, 0.16,])
    # resampled_ds = tf.data.experimental.sample_from_datasets(datasets[:2], weights=[0.1, 0.9])
    return resampled_ds


def get_dataset(config, img_paths, labels, dev=False):
    if dev:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    else:
        dataset = get_resampled_dataset(img_paths, labels)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.shuffle(config.data.shuffle_batch)
    if config.aug:
        Image_Data_Class = ImageData(config)
        dataset = dataset.map(Image_Data_Class.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=config.data.batch_size)
    return dataset


def get_test_dataset(config, img_paths):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(process_only_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=config.data.batch_size)
    return dataset


def get_train_data_loader(config, csv_path):
    csv_file = pd.read_csv(csv_path)
    img_paths = csv_file['filename'].to_numpy()
    labels = csv_file['label'].to_numpy()
    labels -= 1 # to set start index to zero
    indices = np.random.permutation(img_paths.shape[0])
    train_idx, dev_idx = indices[len(labels) // 10:], indices[:len(labels) // 10]
    train_img_paths, dev_img_paths = img_paths[train_idx], img_paths[dev_idx]
    train_labels, dev_labels = labels[train_idx], labels[dev_idx]
    config.data.dev_data_size = len(dev_idx)

    train_dataset = get_dataset(config, train_img_paths, train_labels).repeat()
    dev_dataset = get_dataset(config, dev_img_paths, dev_labels, dev=True).repeat()

    return train_dataset, dev_dataset


def get_test_data_loader(config, csv_path):
    csv_file = pd.read_csv(csv_path)
    test_img_paths = csv_file['filename'].to_numpy()
    test_dataset = get_test_dataset(config, test_img_paths)
    return test_dataset


if __name__=="__main__":
    config = get_config_from_json("config.json")
    train_data_loader, dev_data_loader = get_train_data_loader(config, "train_vision.csv")
    for train_x, y in train_data_loader:
        unique, counts = np.unique(y.numpy(), return_counts=True)
        print(dict(zip(unique, counts)))
    # test_data_loader = get_test_data_loader(config, "test_vision.csv")
    # for test_img_path in test_data_loader:
    #     print(test_img_path)