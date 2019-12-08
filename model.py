import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, add, Conv2D, MaxPooling2D, \
    BatchNormalization, Activation

from utils import stem_block, inception_resnet_A, squeeze_excitation_layer, reduction_A, inception_resnet_B, \
    reduction_B, inception_resnet_C


def build_senet(config, input_x):
    x = stem_block(input_x, scope='stem')

    for i in range(config.trainer.senet.inception_a_iter):
        x = inception_resnet_A(x, scope='Inception_A' + str(i))
        channel = x.shape[-1]
        x = squeeze_excitation_layer(x, out_dim=channel, ratio=config.trainer.senet.reduction_ratio, number=i)

    x = reduction_A(x, scope='Reduction_A')

    channel = x.shape[-1]
    x = squeeze_excitation_layer(x, out_dim=channel, ratio=config.trainer.senet.reduction_ratio, number=6)

    for i in range(config.trainer.senet.inception_b_iter):
        x = inception_resnet_B(x, scope='Inception_B' + str(i))
        channel = x.shape[-1]
        x = squeeze_excitation_layer(x, out_dim=channel, ratio=config.trainer.senet.reduction_ratio, number=7 + i)

    x = reduction_B(x, scope='Reduction_B')

    channel = x.shape[-1]
    x = squeeze_excitation_layer(x, out_dim=channel, ratio=config.trainer.senet.reduction_ratio, number=18)

    for i in range(config.trainer.senet.inception_c_iter):
        x = inception_resnet_C(x, scope='Inception_C' + str(i))
        channel = int(np.shape(x)[-1])
        x = squeeze_excitation_layer(x, out_dim=channel, ratio=config.trainer.senet.reduction_ratio, number=19 + i)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.2)(x, training=config.training)
    x = Flatten()(x)

    x = Dense(config.data.class_num, activation='softmax')(x)
    return x


def get_se_model(config):
    input = keras.Input(shape=(128, 128, 3))
    output = build_senet(config, input)
    se_model = keras.Model(input, output)
    config.training = True
    return se_model


def res_net_block(input_data, filters, conv_size, training):
    x = Conv2D(filters=filters, kernel_size=conv_size, activation='relu', padding='same')(input_data)
    x = BatchNormalization()(x, training=training)
    x = Conv2D(filters=filters, kernel_size=conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x, training=training)
    x = add([x, input_data])
    x = Activation('relu')(x)
    return x


def get_resnet_model(config):
    inputs = keras.Input(shape=(32, 32, 3))

    x = Conv2D(32, 3, activation='relu')(inputs)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D(3)(x)

    for i in range(config.trainer.resnet.num_blocks):
        x = res_net_block(x, 64, 3, training=config.training)

    x = Conv2D(64, 3, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x, training=config.training)
    outputs = Dense(config.data.class_num, activation='softmax')(x)
    res_net_model = keras.Model(inputs, outputs)
    return res_net_model
