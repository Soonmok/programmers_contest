import numpy as np
import pandas as pd
import tensorflow as tf

from config import get_config_from_json
from data_loader import get_train_data_loader, get_test_data_loader
from fixresnet import resnet50
from model import get_se_model, get_resnet_model, get_sphere_model


def train():
    config = get_config_from_json("config.json")
    train_data_loader, dev_data_loader = get_train_data_loader(config, "train_vision.csv")
    test_data_loader = get_test_data_loader(config, "test_vision.csv")
    config.training = True
    if config.model == "baseline":
        model = get_resnet_model(config)
    elif config.model == "senet":
        model = get_se_model(config)
    elif config.model == "resnet":
        model = resnet50()
        model.build(input_shape=(128, 128, 3))
    elif config.model == "sphere":
        model = get_sphere_model(config)
    else:
        model = None
    print(model.summary())

    train_data_loader = train_data_loader.__iter__()
    dev_data_loader = dev_data_loader.__iter__()

    # define losses and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(x_train, y_train):
        with tf.GradientTape() as senet_tape:
            y_pred = model(x_train, training=True)
            loss = loss_object(y_train, y_pred)
        # tf.print(tf.argmax(y_pred, axis=-1))
        gradient_of_model = senet_tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient_of_model, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y_train, y_pred)

    @tf.function
    def dev_step(x_dev, y_dev):
        y_pred = model(x_dev, training=False)
        # tf.print(y_pred)
        t_loss = loss_object(y_dev, y_pred)

        test_loss(t_loss)
        test_accuracy(y_dev, y_pred)

    def test_step(x_test):
        y_pred = model(x_test, training=False)
        return y_pred

    default_dev_score = 88
    train_steps = 5500 // config.data.batch_size
    dev_steps = 500 // config.data.batch_size

    for epoch in range(config.trainer.epochs):
        config.training = True
        for idx in range(train_steps):
            x_train, y_train = next(train_data_loader)
            train_step(x_train, y_train)
            template = 'Epoch {}, Steps : {}/{}, Loss: {}, Accuracy: {}'
            # print(template.format(epoch+1, idx+1, train_steps, train_loss.result(), train_accuracy.result() * 100))

        config.training = False
        for idx in range(dev_steps):
            x_dev, y_dev = next(dev_data_loader)
            dev_step(x_dev, y_dev)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Dev Loss: {}, Dev Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        if test_accuracy.result() * 100 > default_dev_score:
            labels = []
            for idx, x_test in enumerate(test_data_loader):
                y_pred = test_step(x_test)
                labels.append(y_pred.numpy())
            labels = np.concatenate(labels, axis=0)
            labels = np.argmax(labels, axis=1)
            df = pd.read_csv("test_vision.csv")
            df['label'] = labels + 1
            df.drop(df.columns[0], axis=1)
            del df['filename']
            df.to_csv(f"test_result_{epoch + 1}.csv", mode='w')

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == "__main__":
    train()
