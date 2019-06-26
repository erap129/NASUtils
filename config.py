config = {}

def init_configurations(grid, pop_size, max_network_dims, conv_max_height, conv_max_width, conv_max_filters,
                          conv_max_stride, pool_max_height, pool_max_width, pool_max_stride):
    global config
    for key, value in locals().items():
        config[key] = value
