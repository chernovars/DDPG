import tensorflow as tf
import numpy as np

def create_decay_scheme(decay, parameters, scheme="scalar"):
    '''
    :param decay:

        "scalar" - scalar

        "layer" - [ scalar, num_of_layer(starting from 1) ]

        "input" - [ scalar, num_of_input1, num_of_input2 ]

        "gradient" - [ scalar1, scalar2, ... ]

        "tree" - [ [ scalar, num_of_input ], matrix_of_decays_for_layer2, ... ]

    :param parameters:

        sorted list of parameter matrices: [ layer_1_weights, layer_2_weights, ... ]

    :param scheme:

        "scalar" - scalar multiplication for every weight parameter

        "layer" - scalar multiplication with specific layer's weght-matrix

        "input" - scalar multiplication with weights connected with specific input

        "gradient" - there is one decay factor for a layer, and later layers get more decay

        "tree" - decay only one weight in input layer, and then have an individual scalar decay for each layer after
    :return:
    '''

    with tf.variable_scope("param_decay"):
        if scheme == "scalar":
            scaling_factor = tf.constant(decay)
            return [param.assign(param * scaling_factor) for param in parameters]
        if scheme == "layer":
            scaling_factor = tf.constant(decay[0])
            layer = decay[1]-1
            param = parameters[layer]
            return [param.assign(param * scaling_factor)]
        if scheme == "input":
            param = parameters[0]
            decay_matrix = np.ones(param.get_shape().as_list())
            decay_matrix[decay[1], :] *= decay[0]
            decay_matrix[decay[2], :] *= decay[0]
            return [param.assign(param * decay_matrix)]
        if scheme == "output":
            param = parameters[-1]
            decay_matrix = np.ones(param.get_shape().as_list())
            decay_matrix[decay[1], :] *= decay[0]
            decay_matrix[decay[2], :] *= decay[0]
            return [param.assign(param * decay_matrix)]
        if scheme == "gradient":
            assert len(decay) == len(parameters)
            return [param.assign(param * tf.constant(d)) for param, d in zip(parameters, decay)]
        if scheme == "complex":
            ops = []
            assert len(decay) == len(parameters)
            for param, decay_matrix in zip(parameters, decay):
                assert param.get_shape() == decay_matrix.get_shape()
                ops.append(param.assign(param * decay_matrix))
            return ops
        if scheme == "tree":
            param = parameters[0]
            input_num = decay[0][1]
            decay_matrix = np.ones(param.get_shape().as_list())
            decay_matrix[input_num, :] *= decay[input_num]

            ops = [param.assign(param * decay_matrix)]
            assert len(decay) == len(parameters)
            for param, decay_matrix in zip(parameters[1:], decay[1:]):
                assert param.get_shape() == decay_matrix.get_shape()
                ops.append(param.assign(param * decay_matrix))
            return ops
