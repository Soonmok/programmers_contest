import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, \
    ReLU, BatchNormalization, multiply, add, GlobalAveragePooling2D, Dense, Activation, Reshape, PReLU, Dropout


def stem_block(x, scope="stem", training=True):
    with tf.name_scope(scope):
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid')(x)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=3, padding='valid')(x)
        x = ReLU()(x)
        block_1 = Conv2D(filters=32, kernel_size=3)(x)
        block_1 = ReLU()(block_1)

        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(block_1)
        split_conv_x = Conv2D(filters=48, kernel_size=3, strides=2, padding='valid')(block_1)
        split_conv_x = PReLU()(split_conv_x)
        x = concatenate([split_max_x, split_conv_x], axis=3) # 32 + 48

        split_conv_x1 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x1 = PReLU()(split_conv_x1)
        split_conv_x1 = Conv2D(filters=32, kernel_size=3, padding='valid')(split_conv_x1)
        split_conv_x1 = PReLU()(split_conv_x1)

        split_conv_x2 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same')(split_conv_x2)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same')(split_conv_x2)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=32, kernel_size=3, padding='valid')(split_conv_x2)
        split_conv_x2 = PReLU()(split_conv_x2)

        x = concatenate([split_conv_x1, split_conv_x2], axis=-1) # 64

        split_conv_x = Conv2D(filters=64, kernel_size=[3, 3], strides=2, padding='valid')(x)
        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

        x = concatenate([split_conv_x, split_max_x], axis=-1)
        x = BatchNormalization()(x, training=training)  # TODO: need to set trainable boolean
        x = PReLU()(x)

        return x


def inception_resnet_A(x, scope="inception_a", training=True):
    with tf.name_scope(scope):
        init_x = x
        split_conv_x1 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x1 = BatchNormalization()(split_conv_x1, training=training)
        split_conv_x1 = PReLU()(split_conv_x1)

        split_conv_x2 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x2 = BatchNormalization()(split_conv_x2, training=training)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=32, kernel_size=3, padding='same')(split_conv_x2)
        split_conv_x2 = PReLU()(split_conv_x2)

        split_conv_x3 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x3 = BatchNormalization()(split_conv_x3, training=training)
        split_conv_x3 = PReLU()(split_conv_x3)
        split_conv_x3 = Conv2D(filters=48, kernel_size=3, padding='same')(split_conv_x3)
        split_conv_x3 = PReLU()(split_conv_x3)
        split_conv_x3 = Conv2D(filters=64, kernel_size=3, padding='same')(split_conv_x3)
        split_conv_x3 = PReLU()(split_conv_x3)

        x = concatenate([split_conv_x1, split_conv_x2, split_conv_x3], axis=-1)
        x = Conv2D(filters=128, kernel_size=1)(x)

        x = tf.math.scalar_mul(0.1, x)
        x = add([init_x, x])

        x = BatchNormalization()(x, training=training)
        x = PReLU()(x)

        return x


def inception_resnet_B(x, scope="inception_b", training=True):
    with tf.name_scope(scope):
        init = x

        split_conv_x1 = Conv2D(filters=64, kernel_size=1, padding='same')(x)
        split_conv_x1 = BatchNormalization()(split_conv_x1, training=training)
        split_conv_x1 = PReLU()(split_conv_x1)

        split_conv_x2 = Conv2D(filters=32, kernel_size=1, padding='same')(x)
        split_conv_x2 = BatchNormalization()(split_conv_x2, training=training)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same')(split_conv_x2)
        split_conv_x2 = BatchNormalization()(split_conv_x2, training=training)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same')(split_conv_x2)
        split_conv_x2 = PReLU()(split_conv_x2)

        x = concatenate([split_conv_x1, split_conv_x2], axis=-1)
        x = Conv2D(filters=256, kernel_size=1, padding='same')(x)

        x = tf.math.scalar_mul(0.1, x)
        x = add([init, x])

        x = BatchNormalization()(x, training=training)
        x = PReLU()(x)

        return x


def inception_resnet_C(x, scope="inception_c", training=True):
    with tf.name_scope(scope):
        init = x

        split_conv_x1 = Conv2D(filters=64, kernel_size=1, padding='same')(x)
        split_conv_x1 = BatchNormalization()(split_conv_x1, training=training)
        split_conv_x1 = PReLU()(split_conv_x1)

        split_conv_x2 = Conv2D(filters=64, kernel_size=1, padding='same')(x)
        split_conv_x2 = BatchNormalization()(split_conv_x2, training=training)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=96, kernel_size=(1, 3), padding='same')(split_conv_x2)
        split_conv_x2 = BatchNormalization()(split_conv_x2, training=training)
        split_conv_x2 = PReLU()(split_conv_x2)
        split_conv_x2 = Conv2D(filters=96, kernel_size=(3, 1), padding='same')(split_conv_x2)
        split_conv_x2 = PReLU()(split_conv_x2)

        x = concatenate([split_conv_x1, split_conv_x2], axis=-1)
        x = Conv2D(filters=448, kernel_size=1, padding='same')(x)

        x = tf.math.scalar_mul(0.1, x)
        x = add([init, x])

        x = BatchNormalization()(x, training=training)
        x = PReLU()(x)

        return x


def reduction_A(x, scope="reduction_a", training=True):
    with tf.name_scope(scope):
        k = 96
        l = 96
        m = 64
        n = 64

        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

        split_conv_x1 = Conv2D(filters=n, kernel_size=3, strides=2, padding='valid', activation='relu')(x)

        split_conv_x2 = Conv2D(filters=k, kernel_size=1, padding='same', activation='relu')(x)
        split_conv_x2 = Conv2D(filters=l, kernel_size=3, padding='same', activation='relu')(split_conv_x2)
        split_conv_x2 = Conv2D(filters=m, kernel_size=3, strides=2, padding='valid', activation='relu')(split_conv_x2)

        x = concatenate([split_max_x, split_conv_x1, split_conv_x2], axis=-1)

        x = BatchNormalization()(x, training=training)
        x = PReLU()(x)

        return x


def reduction_B(x, scope="reduction_b", training=True):
    with tf.name_scope(scope):
        split_max_x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

        split_conv_x1 = Conv2D(filters=48, kernel_size=[1, 1], padding='same', activation='relu')(x)
        split_conv_x1 = Conv2D(filters=64, kernel_size=[3, 3], strides=2, padding='valid', activation='relu')(split_conv_x1)

        split_conv_x2 = Conv2D(filters=48, kernel_size=[1, 1], padding='same', activation='relu')(x)
        split_conv_x2 = Conv2D(filters=64, kernel_size=[3, 3], strides=2, padding='valid', activation='relu')(split_conv_x2)

        split_conv_x3 = Conv2D(filters=48, kernel_size=[1, 1], padding='same', activation='relu')(x)
        split_conv_x3 = Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu')(split_conv_x3)
        split_conv_x3 = Conv2D(filters=64, kernel_size=[3, 3], strides=2, padding='valid', activation='relu')(split_conv_x3)

        x = concatenate([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3], axis=-1)

        x = BatchNormalization()(x, training=training)
        x = PReLU()(x)

        return x


def squeeze_excitation_layer(input_x, out_dim, ratio, scope="squeeze", number=1):
    with tf.name_scope(f"scope_{number}"):
        squeeze = GlobalAveragePooling2D()(input_x)

        excitation = Dense(out_dim / ratio)(squeeze)
        excitation = Dropout(0.5)(excitation)
        excitation = ReLU()(excitation)
        excitation = Dense(out_dim)(excitation)
        excitation = Activation(tf.nn.sigmoid)(excitation)

        excitation = Reshape((1, 1, out_dim))(excitation)
        scale = multiply([input_x, excitation])

        return scale
