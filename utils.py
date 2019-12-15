import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, \
    ReLU, BatchNormalization, multiply, add, GlobalAveragePooling2D, Dense, Activation, Reshape, PReLU


def stem_block(x, scope="stem", training=True):
    with tf.name_scope(scope):
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid')(x)
        x = PReLU()(x)
        x = Conv2D(filters=32, kernel_size=3, padding='valid')(x)
        x = PReLU()(x)
        block_1 = Conv2D(filters=32, kernel_size=3)(x)
        x = PReLU()(x)

        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(block_1)
        split_conv_x = Conv2D(filters=96, kernel_size=3, strides=2, padding='valid', activation='relu')(block_1)
        x = concatenate([split_max_x, split_conv_x], axis=3)

        split_conv_x1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu')(x)
        split_conv_x1 = Conv2D(filters=96, kernel_size=3, padding='valid', activation='relu')(split_conv_x1)

        split_conv_x2 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
        split_conv_x2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(split_conv_x2)
        split_conv_x2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same')(split_conv_x2)
        split_conv_x2 = Conv2D(filters=96, kernel_size=3, padding='valid', activation='relu')(split_conv_x2)

        x = concatenate([split_conv_x1, split_conv_x2], axis=-1)

        split_conv_x = Conv2D(filters=192, kernel_size=[3, 3], strides=2, padding='valid')(x)
        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

        x = concatenate([split_conv_x, split_max_x], axis=-1)
        x = BatchNormalization()(x, training=training)  # TODO: need to set trainable boolean
        x = ReLU()(x)

        return x


def inception_resnet_A(x, scope="inception_a", training=True):
    with tf.name_scope(scope):
        init_x = x
        split_conv_x1 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x1 = ReLU()(split_conv_x1)

        split_conv_x2 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x2 = ReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=32, kernel_size=3, padding='same')(split_conv_x2)
        split_conv_x2 = ReLU()(split_conv_x2)

        split_conv_x3 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x3 = ReLU()(split_conv_x3)
        split_conv_x3 = Conv2D(filters=48, kernel_size=3, padding='same')(split_conv_x3)
        split_conv_x3 = ReLU()(split_conv_x3)
        split_conv_x3 = Conv2D(filters=64, kernel_size=3, padding='same')(split_conv_x3)
        split_conv_x3 = ReLU()(split_conv_x3)

        x = concatenate([split_conv_x1, split_conv_x2, split_conv_x3], axis=-1)
        x = Conv2D(filters=384, kernel_size=1)(x)

        x = tf.math.scalar_mul(0.1, x)
        x = add([init_x, x])

        x = BatchNormalization()(x, training=training)
        x = ReLU()(x)

        return x


def inception_resnet_B(x, scope="inception_b", training=True):
    with tf.name_scope(scope):
        init = x

        split_conv_x1 = Conv2D(filters=192, kernel_size=1, padding='same')(x)
        split_conv_x1 = ReLU()(split_conv_x1)

        split_conv_x2 = Conv2D(filters=128, kernel_size=1, padding='same')(x)
        split_conv_x2 = ReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=160, kernel_size=(7, 1), padding='same')(split_conv_x2)
        split_conv_x2 = ReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=192, kernel_size=(1, 7), padding='same')(split_conv_x2)
        split_conv_x2 = ReLU()(split_conv_x2)

        x = concatenate([split_conv_x1, split_conv_x2], axis=-1)
        x = Conv2D(filters=1152, kernel_size=1, padding='same')(x)

        x = tf.math.scalar_mul(0.1, x)
        x = add([init, x])

        x = BatchNormalization()(x, training=training)
        x = ReLU()(x)

        return x


def inception_resnet_C(x, scope="inception_c", training=True):
    with tf.name_scope(scope):
        init = x

        split_conv_x1 = Conv2D(filters=192, kernel_size=1, padding='same')(x)

        split_conv_x2 = Conv2D(filters=192, kernel_size=1, padding='same')(x)
        split_conv_x2 = Conv2D(filters=224, kernel_size=(1, 3), padding='same')(split_conv_x2)
        split_conv_x2 = Conv2D(filters=256, kernel_size=(3, 1), padding='same')(split_conv_x2)

        x = concatenate([split_conv_x1, split_conv_x2], axis=-1)
        x = Conv2D(filters=2144, kernel_size=1, padding='same')(x)

        x = tf.math.scalar_mul(0.1, x)
        x = add([init, x])

        x = BatchNormalization()(x, training=training)
        x = ReLU()(x)

        return x


def reduction_A(x, scope="reduction_a", training=True):
    with tf.name_scope(scope):
        k = 256
        l = 256
        m = 384
        n = 384

        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

        split_conv_x1 = Conv2D(filters=n, kernel_size=3, strides=2, padding='valid', activation='relu')(x)

        split_conv_x2 = Conv2D(filters=k, kernel_size=1, padding='same', activation='relu')(x)
        split_conv_x2 = Conv2D(filters=l, kernel_size=3, padding='same', activation='relu')(split_conv_x2)
        split_conv_x2 = Conv2D(filters=m, kernel_size=3, strides=2, padding='valid', activation='relu')(split_conv_x2)

        x = concatenate([split_max_x, split_conv_x1, split_conv_x2], axis=-1)

        x = BatchNormalization()(x, training=training)
        x = ReLU()(x)

        return x


def reduction_B(x, scope="reduction_b", training=True):
    with tf.name_scope(scope):
        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

        split_conv_x1 = Conv2D(filters=256, kernel_size=[1, 1], padding='same', activation='relu')(x)
        split_conv_x1 = Conv2D(filters=384, kernel_size=[3, 3], strides=2, padding='valid', activation='relu')(split_conv_x1)

        split_conv_x2 = Conv2D(filters=256, kernel_size=[1, 1], padding='same', activation='relu')(x)
        split_conv_x2 = Conv2D(filters=288, kernel_size=[3, 3], strides=2, padding='valid', activation='relu')(split_conv_x2)

        split_conv_x3 = Conv2D(filters=256, kernel_size=[1, 1], padding='same', activation='relu')(x)
        split_conv_x3 = Conv2D(filters=288, kernel_size=[3, 3], padding='same', activation='relu')(split_conv_x3)
        split_conv_x3 = Conv2D(filters=320, kernel_size=[3, 3], strides=2, padding='valid', activation='relu')(split_conv_x3)

        x = concatenate([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3], axis=-1)

        x = BatchNormalization()(x, training=training)
        x = ReLU()(x)

        return x


def squeeze_excitation_layer(input_x, out_dim, ratio, scope="squeeze", number=1):
    with tf.name_scope(f"scope_{number}"):
        squeeze = GlobalAveragePooling2D()(input_x)

        excitation = Dense(out_dim / ratio)(squeeze)
        excitation = ReLU()(excitation)
        excitation = Dense(out_dim)(excitation)
        excitation = Activation(tf.nn.sigmoid)(excitation)

        excitation = Reshape((1, 1, out_dim))(excitation)
        scale = multiply([input_x, excitation])

        return scale
