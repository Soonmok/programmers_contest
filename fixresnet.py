import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, MaxPooling2D, \
    GlobalAveragePooling2D, Dense


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, filters, stride=1, training=True, downsample=None, groups=1,
                 base_width=64, dialation='same', norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1')
        self.conv1 = Conv2D(filters=filters, kernel_size=3, stride=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = Activation()
        self.conv2 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.bn2 = norm_layer(filters)
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = add([out, identity])
        out = self.relu(out)


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, stride=1, downsample=None, groups=1,
                 base_width=64, dilation='same', norm_layer=None, training=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization()
        width = int(filters * (base_width / 64.)) * groups
        self.conv1 = Conv2D(filters=width, kernel_size=1, strides=1)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=width, strides=stride, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=filters * 4, kernel_size=1, padding='same')
        self.bn3 = BatchNormalization()
        self.relu = Activation('relu')
        self.downsample = downsample
        self.stride = stride
        self.training = training

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=self.traininig)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = add([out, identity])
        out = self.relu(out)

        return out


class ResNet(tf.keras.Model):

    def __init__(self, block, layers, num_classes=6, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, training=True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization()
        self._norm_layer = norm_layer
        self.training = training

        self.filters = 64
        self.dilation = 'same'
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2D(filters=self.filters, kernel_size=7, strides=2, padding='same', input_shape=(128, 128, 3))

        self.bn1 = norm_layer
        self.relu = Activation('relu')
        self.maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, filters, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        model = Sequential()
        if dilate:
            stride = 1
        if stride != 1 or self.filters != filters * block.expansion:
            downsample = Sequential()
            downsample.add(Conv2D(filters=filters * block.expansion, kernel_size=1, strides=stride))
            downsample.add(norm_layer)

        layers = []
        layers.append(block(filters, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.filters = filters * block.expansion
        for _ in range(1, blocks):
            model.add(block(self.filters, filters, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer))
            # layers.append(block(self.inplanes, filters, groups=self.groups,
            #                     base_width=self.base_width, dilation=self.dilation,
            #                     norm_layer=norm_layer))

        return model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x, self.training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x1 = x.reshape(x.size(0), -1)
        x = self.fc(x1)

        return x


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


if __name__ == "__main__":
    model = resnet50()
