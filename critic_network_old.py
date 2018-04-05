import tensorflow as tf
import numpy as np
import math
import os, errno

#Deprecation

class CriticNetwork:
    """docstring for CriticNetwork
        original
    """

    def __init__(self, sess, state_dim, action_dim, env_name, critic_settings, save_folder):
        if len(critic_settings["layers"]) == 2:
            self.LAYER1_SIZE = critic_settings["layers"][0]
            self.LAYER2_SIZE = critic_settings["layers"][1]
        else:
            self.LAYER1_SIZE = 100
            self.LAYER2_SIZE = 40

        self.LEARNING_RATE = float(critic_settings["learning_rate"]) #1e-4
        self.TAU = float(critic_settings["tau"]) #0.008
        self.L2 = float(critic_settings["l2"]) #0.01

        self.time_step = 0
        self.sess = sess
        self.env_name = env_name
        self.save_folder = save_folder
        # create q network
        self.state_input, \
        self.action_input, \
        self.q_value_output, \
        self.net = self.create_q_network(state_dim, action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_update = self.create_target_q_network(state_dim, action_dim, self.net)

        self.create_training_method()

        # initialization
        with tf.variable_scope("init_critic"):
            self.sess.run(tf.global_variables_initializer())

        self.update_target()

        self.load_network()

    def create_training_method(self):
        with tf.variable_scope("critic_train"):
            # Define training optimizer
            self.y_input = tf.placeholder("float", [None, 1])
            weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.net])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
            self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
            self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim):
        with tf.variable_scope("critic"):
            # the layer size could be changed
            layer1_size = self.LAYER1_SIZE
            layer2_size = self.LAYER2_SIZE

            state_input = tf.placeholder("float", [None, state_dim])
            action_input = tf.placeholder("float", [None, action_dim])

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

            q_value_output = tf.identity(layer3, name='identity_output')

            return state_input, action_input, q_value_output, [W1, b1, W2, W2_action, b2, W3, b3]

    def create_target_q_network(self, state_dim, action_dim, net):
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

            copy_op = [W1.assign(target_net[0]), b1.assign(target_net[1]),
                       W2.assign(target_net[2]), W2_action.assign(target_net[3]), b2.assign(target_net[4]),
                       W3.assign(target_net[5]), b3.assign(target_net[6])]
            target_update = [target_update] + copy_op

            q_value_output = tf.identity(layer3)

        return state_input, action_input, q_value_output, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
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
            self.action_input: action_batch})

    # f fan-in size
    def variable(self, shape, f, name=None):
        if name is not None:
            return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)), name=name)
        else:
            return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

    def load_network(self, save_folder=None):
        with tf.variable_scope("save_critic"):
            self.saver = tf.train.Saver()
            path = self._get_path(save_folder)
            checkpoint = tf.train.get_checkpoint_state(path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

    def save_network(self, time_step, save_folder=None):
        print('saving critic-net...', time_step)
        path = self._get_path(save_folder)
        self.saver.save(self.sess, path, global_step=time_step)

    def _get_path(self, save_folder):
        if save_folder is not None:
            path = save_folder + '/saved_critic_networks/'
        elif self.save_folder is not None:
            path = self.save_folder + '/saved_critic_networks/'
        else:
            path = "experiments/" + self.env_name + "/saved_critic_networks/"
        return path
