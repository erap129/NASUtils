import numpy as np
from torch.nn import init
from pytorch_custom_layers import *
import copy
from NNLayers import *


def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
    print('-----------------------------------------------')


def fix_layers_dims(layer, prev_layer):
    # TODO - check layers dims with regard to previous layer and fix - Support Conv and max-pool
    print("ERRor - function not implemented")
    pass


def new_model_from_structure_pytorch(layer_collection, apply_fix=False):
    model = nn.Sequential()
    activations = {'relu': nn.ReLU, 'elu': nn.ELU, 'softmax': nn.Softmax, 'sigmoid': nn.Sigmoid}

    input_shape = (config['batch_size'], config['dataset_channels'], config['dataset_height'], config['dataset_width'])

    for i in range(len(layer_collection)):
        layer = layer_collection[i]
        if i > 0:
            out = model.forward(torch.ones(size=input_shape, dtype=torch.float32))
            prev_layer_shape = out.cpu().data.numpy().shape
        else:
            prev_layer_shape = input_shape

        if isinstance(layer, PoolingLayer):
            if apply_fix:
                fix_layers_dims(layer)
            model.add_module(name=f'{type(layer).__name__}_{i}',
                             module=nn.MaxPool2d(kernel_size=(int(layer.height), int(layer.width)),
                                                 stride=int(layer.stride)))

        elif isinstance(layer, ConvLayer):
            if apply_fix:
                fix_layers_dims(layer)
            model.add_module(f'{type(layer).__name__}_{i}', nn.Conv2d(in_channels=prev_layer_shape[1],
                                                                      out_channels=layer.channels,
                                                                      kernel_size=(layer.height, layer.width),
                                                                      stride=layer.stride))

        elif isinstance(layer, BatchNormLayer):
            model.add_module(f'{type(layer).__name__}_{i}', nn.BatchNorm2d(prev_layer_shape[1],
                                                                           affine=True, eps=1e-5))

        elif isinstance(layer, ActivationLayer):
            model.add_module(f'{type(layer).__name__}_{i}', activations[layer.activation_type]())

        elif isinstance(layer, DropoutLayer):
            model.add_module(f'{type(layer).__name__}_{i}', nn.Dropout(p=layer.rate))

        elif isinstance(layer, IdentityLayer):
            model.add_module(f'{type(layer).__name__}_{i}', IdentityModule())

        elif isinstance(layer, SqueezeLayer):
            model.add_module('squeeze', Squeeze_Layer())

        elif isinstance(layer, LinearLayer):
            model.add_module(f'{type(layer).__name__}_{i}', Flatten_Layer())
            model.add_module(f'{type(layer).__name__}_{i}', nn.Linear(in_features=np.prod(prev_layer_shape[1:])
                                                                      , out_features=layer.output_dim))

    # TODO - refactor weights init to another method - add support for choosing wether to init weights or not
    # init.xavier_uniform_(list(model._modules.items())[-3][1].weight, gain=1)
    # init.constant_(list(model._modules.items())[-3][1].bias, 0)
    return model


def check_legal_model(layer_collection):
    height = config['dataset_height']
    width = config['dataset_width']
    for layer in layer_collection:
        if type(layer) == ConvLayer or type(layer) == PoolingLayer:
            height = (height - layer.height)/layer.stride + 1
            width = (width - layer.width)/layer.stride + 1
        if height < 1 or width < 1:
            print(f"illegal model, height={height}, width={width}")
            return False
    return True


def random_layer():
    layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
    return layers[random.randint(0, len(layers) - 1)]()


def random_model():
    layer_collection = []
    for i in range(config['max_network_dims'][0]):
        layer_collection.append(random_layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return random_model(config['max_network_dims'])


def uniform_model(n_layers, layer_type):
    layer_collection = []
    for i in range(n_layers):
        layer = layer_type()
        layer_collection.append(layer)
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return uniform_model(n_layers, layer_type)


def custom_model(layers):
    layer_collection = []
    for layer in layers:
        layer_collection.append(layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return custom_model(layers)


def finalize_model(layer_collection):
    # TODO - add support for grid (parallel layers, skip connections)
    # if config['grid'):
    #     return ModelFromGrid(layer_collection)

    layer_collection = copy.deepcopy(layer_collection)

    # TODO - add support for cropping in BCI 2
    # if config['cropping']:
    #     final_conv_time = config['final_conv_size')

    linear_layer = LinearLayer(config['num_classes'])
    activation = ActivationLayer('softmax')
    layer_collection.append(linear_layer)
    layer_collection.append(activation)

    return new_model_from_structure_pytorch(layer_collection)
