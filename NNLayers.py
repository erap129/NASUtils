import random
from config import config


class Layer():
    def __init__(self, name=None):
        self.name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__


class InputLayer(Layer):
    def __init__(self, shape_height, shape_width):
        Layer.__init__(self)
        self.shape_height = shape_height
        self.shape_width = shape_width


class SqueezeLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


class DropoutLayer(Layer):
    def __init__(self, rate=None):
        Layer.__init__(self)
        if rate is None:
            rate = random.uniform(0, config['dropout_max_rate'])
        self.rate = rate


class BatchNormLayer(Layer):
    def __init__(self, axis=3, momentum=0.1, epsilon=1e-5):
        Layer.__init__(self)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon


class ActivationLayer(Layer):
    def __init__(self, activation_type='elu'):
        Layer.__init__(self)
        self.activation_type = activation_type


class LinearLayer(Layer):
    # @initializer
    def __init__(self, output_dim=None, name=None):
        Layer.__init__(self, name)
        if output_dim is None:
            output_dim = random.randint(1, config['linear_max_dim'])
        self.output_dim = output_dim


class ConvLayer(Layer):
    # @initializer
    def __init__(self, height=None, width=None, channels=None, stride=None, name=None):
        Layer.__init__(self, name)
        if height is None:
            height = random.randint(1, config['conv_max_height'])
        if width is None:
            width = random.randint(1, config['conv_max_width'])
        if channels is None:
            channels = random.randint(1, config['conv_max_channels'])
        if stride is None:
            stride = random.randint(1, config['conv_max_stride'])
        self.height = height
        self.width = width
        self.channels = channels
        self.stride = stride


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, height=None, width=None, stride=None, mode='max'):
        Layer.__init__(self)
        if height is None:
            height = random.randint(1, config['pool_max_height'])
        if width is None:
            width = random.randint(1, config['pool_max_width'])
        if stride is None:
            stride = random.randint(1, config['pool_max_stride'])
        self.height = height
        self.width = width
        self.stride = stride
        self.mode = mode


class IdentityLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


class ZeroPadLayer(Layer):
    def __init__(self, height_pad_top, height_pad_bottom, width_pad_left, width_pad_right):
        Layer.__init__(self)
        self.height_pad_top = height_pad_top
        self.height_pad_bottom = height_pad_bottom
        self.width_pad_left = width_pad_left
        self.width_pad_right = width_pad_right


class ConcatLayer(Layer):
    def __init__(self, first_layer_index, second_layer_index):
        Layer.__init__(self)
        self.first_layer_index = first_layer_index
        self.second_layer_index = second_layer_index


class AveragingLayer(Layer):
    def __init__(self):
        pass
