import copy
from NNLayers import *


# ================================ UTILS ===============================================================================


def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
        print(attrs)
    print('-----------------------------------------------')


# ================================= Models Creation ====================================================================

def random_model(max_network_depth):
    layer_collection = []
    for i in range(max_network_depth):
        layer_collection.append(random_layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return random_model(max_network_depth)


def uniform_model(n_layers, layer_type):
    layer_collection = []
    for i in range(n_layers):
        layer = layer_type()
        layer_collection.append(layer)
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return uniform_model(n_layers, layer_type)


# TODO - how is this different then random_model?
def custom_model(layers):
    layer_collection = []
    for layer in layers:
        layer_collection.append(layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return custom_model(layers)


def random_layer():
    layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
    return layers[random.randint(0, len(layers) - 1)]()


# ========================== Model Validate and FIX ====================================================================

def fix_layers_dims(layer, prev_layer):
    # TODO - check layers dims with regard to previous layer and fix - Support Conv and max-pool
    print("ERRor - function not implemented")
    pass


def check_legal_model(layer_collection):
    # TODO - add checks to this method, doesnt validate network properly
    height = config['dataset_height']
    width = config['dataset_width']
    for layer in layer_collection:
        if type(layer) == ConvLayer or type(layer) == PoolingLayer:
            height = (height - layer.height) / layer.stride + 1
            width = (width - layer.width) / layer.stride + 1
        if height < 1 or width < 1:
            print(f"illegal model, height={height}, width={width}")
            return False
    return True


# TODO -  This method is not used in this class, being called from NASUTILS, maybe we need to move it to NASUTILS
def finalize_model(layer_collection):
    if config['grid']:
        # TODO - add support for grid (parallel layers, skip connections)
        # return ModelFromGrid(layer_collection)
        pass
    else:
        layer_collection = copy.deepcopy(layer_collection)
        output_layer = LinearLayer(config['num_classes'])
        activation = ActivationLayer('softmax')
        layer_collection.append(output_layer)
        layer_collection.append(activation)
    return layer_collection


# ========================== Population Creation =======================================================================

def initialize_population():
    population = []
    if config['grid']:
        # TODO - support parallel layers and skip connections
        # model_init = random_grid_model
        pass
    else:
        for i in range(config['population_size']):
            new_rand_model = random_model(config['max_network_depth'])
            population.append(new_rand_model)
    return population


def save_abstract_model_to_csv():
    # TODO - create a CSV file and save the models to it including all their meta-data
    pass
