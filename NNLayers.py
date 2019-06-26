import random


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


class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


class DropoutLayer(Layer):
    def __init__(self, rate=0.5):
        Layer.__init__(self)
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


class ConvLayer(Layer):
    # @initializer
    def __init__(self, height=None, width=None, channels=None, name=None):
        Layer.__init__(self, name)
        if height is None:
            height = random.randint(1, globals.get('kernel_height_max'))
        if width is None:
            width = random.randint(1, globals.get('kernel_time_max'))
        if channels is None:
            channels = random.randint(1, globals.get('filter_num_max'))
        if globals.get('channel_dim') == 'channels':
            height = 1
        self.kernel_eeg_chan = height
        self.kernel_time = width
        self.filter_num = channels


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, pool_time=None, stride_time=None, mode='max', stride_eeg_chan=1, pool_eeg_chan=1):
        Layer.__init__(self)
        if pool_time is None:
            pool_time = random.randint(1, globals.get('pool_time_max'))
        if stride_time is None:
            stride_time = random.randint(1, globals.get('pool_time_max'))
        self.pool_time = pool_time
        self.stride_time = stride_time
        self.mode = mode
        self.stride_eeg_chan = stride_eeg_chan
        self.pool_eeg_chan = pool_eeg_chan


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