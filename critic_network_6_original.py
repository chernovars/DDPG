import tensorflow as tf
import numpy as np
import math
import os, errno
from network3 import *
#Deprecation

class CriticNetwork(Network):
    """docstring for CriticNetwork

        оригинальный critic поэтапно морфируемый

        всключает в себя create network из критика 4

        3 сета переменнызх

        6: removed copy
    """

    def __init__(self, sess, state_dim, action_dim, env_name, critic_settings, save_folder):
        Network.__init__(self, sess, state_dim, action_dim, "critic", critic_settings, save_folder)
        self.LAYER1_SIZE = critic_settings["layers"][0]
        self.LAYER2_SIZE = critic_settings["layers"][1]
        self.L2 = float(critic_settings["l2"]) #0.01


        # create q network
        self.state_input, \
        self.action_input, \
        self.q_value_output, \
        self.is_training = self.create_q_network(state_dim, action_dim, critic_settings)
        self.net = list(get_net_variables(self.name).values())
        print(get_net_variables(self.name).keys())
        self.parameters = get_net_variables2(self.name)

        # create target q network (the same structure with q network)
        self.target_state_input, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_update = self.create_target_q_network(state_dim, action_dim, self.net, self.parameters)

        self.create_training_method()

        self.post_init()
        self.sess.run(self.copy_op)


    def create_q_network(self, input_dim, action_dim, settings, scope="critic", parameters=None):
        self.actions_layer_number = transfer_parameter(settings, "actions_layer", not_found=2)
        make_new_var = False
        if parameters is None:
            make_new_var = True
            parameters = {}

        with tf.variable_scope(scope):
            state_input = tf.placeholder("float", [None, input_dim], name='state_input')
            actions_input = tf.placeholder("float", [None, action_dim], name='action_input')
            is_training = tf.placeholder(tf.bool, name='bn_is_training')
            layers = settings["layers"]

            def create_dense_layer(input, l_scope, shape, f, param_name="", use_bias=True):
                with tf.variable_scope(l_scope):
                    if make_new_var:
                        W = self.variable(shape, f, name='kernel_' + param_name)
                    else:
                        W = parameters['kernel_' + param_name]
                    if use_bias:
                        if make_new_var:
                            b = self.variable([shape[1]], f, name='bias_' + param_name)
                        else:
                            b = parameters['bias_' + param_name]

                _layer = tf.matmul(input, W)
                if use_bias:
                    _layer = _layer + b
                if self.BATCH_NORM:
                    _layer = tf.layers.batch_normalization(_layer, training=is_training, name=scope+"bn")
                return _layer


            #action input layer
            action_layer_units = layers[self.actions_layer_number-1]

            if self.actions_layer_number > 1:
                f = layers[self.actions_layer_number-1] + action_dim
            else:
                f = input_dim + action_dim
            actions_layer = create_dense_layer(actions_input, l_scope="action_input_layer",
                                               shape=[action_dim, action_layer_units], f=f,
                                               param_name='action', use_bias=False)

            #state input layer
            layer = create_dense_layer(state_input, l_scope="h_layer_1", shape=[input_dim, layers[0]],
                                       f=input_dim, param_name='1')
            if self.actions_layer_number == 1:
                layer = tf.add(layer, actions_layer, name="s_a_sum_1")
            layer = tf.nn.relu(layer, name='activation_input')

            #hidden layers
            for i in range(1, len(settings["layers"])):
                if i == (self.actions_layer_number-1):
                    f = layers[i-1] + action_dim
                else:
                    f = layers[i-1]

                num = str(i + 1)
                l_name = "h_layer_"+num

                layer = create_dense_layer(layer, l_scope="h_layer_" + num, shape=[layers[i-1], layers[i]],
                                           f=f, param_name=num)
                if i == (self.actions_layer_number-1):
                    layer = tf.add(layer, actions_layer, name="s_a_sum_"+l_name)
                layer = tf.nn.relu(layer, name="act_"+l_name)

            #output layer
            with tf.variable_scope("output_layer"):
                prev_l_dim = layers[len(layers)-1]
                if make_new_var:
                    W_out = tf.Variable(tf.random_uniform([prev_l_dim, 1], -3e-3, 3e-3), name='kernel_out')
                    b_out = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name='bias_out')
                else:
                    W_out = parameters['kernel_out']
                    b_out = parameters['bias_out']
                layer = tf.matmul(layer, W_out) + b_out
            output = tf.identity(layer, name='identity_output')

        return state_input, actions_input, output, is_training

    def create_target_q_network(self, state_dim, action_dim, net, parameters):
        with tf.variable_scope("t_critic"):
            state_input = tf.placeholder("float", [None, state_dim])
            action_input = tf.placeholder("float", [None, action_dim])
        with tf.variable_scope("critic_target_update"):
            ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
            target_update = ema.apply(net)
        with tf.variable_scope("t_critic", reuse=True):
            target_net = [ema.average(x) for x in net]

            layer1_size = self.LAYER1_SIZE
            layer2_size = self.LAYER2_SIZE

            with tf.variable_scope("h_layer_1"):
                W1 = self.variable([state_dim, layer1_size], state_dim, name='W1')
                b1 = self.variable([layer1_size], state_dim, name='b1')
                layer1 = tf.matmul(state_input, W1, name='layer1') + b1
            layer1 = tf.nn.relu(layer1, name='activation1')

            with tf.variable_scope("h_layer_2"):
                W2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim, name='W2')
                b2 = self.variable([layer2_size], layer1_size + action_dim, name='b2')
                layer2 = tf.matmul(layer1, W2, name='layer2') + b2

            with tf.variable_scope("h_layer_2_act"):
                W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim, name='W2_action')
                layer_2_a = tf.matmul(action_input, W2_action, name="action_layer0")

            layer2 = tf.nn.relu(layer2 + layer_2_a, name='activation2')

            with tf.variable_scope("output_layer"):
                W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3), name='W3')
                b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name='b3')
                layer3 = tf.matmul(layer2, W3, name='layer3') + b3

            self.copy_op = [W1.assign(parameters["kernel_1"]), b1.assign(parameters["bias_1"]),
                       W2.assign(parameters["kernel_2"]), W2_action.assign(parameters["kernel_action"]), b2.assign(parameters["bias_2"]),
                       W3.assign(parameters["kernel_out"]), b3.assign(parameters["bias_out"])]

            target_update = target_update

            q_value_output = tf.identity(layer3)

        return state_input, action_input, q_value_output, target_update

    def create_training_method(self):
        with tf.variable_scope("critic_train"):
            # Define training optimizer
            self.y_input = tf.placeholder("float", [None, 1])
            weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.net])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='critic')):
                self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
            self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })

