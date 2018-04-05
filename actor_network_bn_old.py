import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math

#Deprecation

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self, sess, state_dim, action_dim, env_name, actor_settings, save_folder):
        # Hyper Parameters
        if len(actor_settings["layers"]) == 2:
            self.LAYER1_SIZE = actor_settings["layers"][0]
            self.LAYER2_SIZE = actor_settings["layers"][1]
        else:
            self.LAYER1_SIZE = 100
            self.LAYER2_SIZE = 40
        self.LEARNING_RATE = float(actor_settings["learning_rate"])#1e-3
        self.TAU = float(actor_settings["tau"])#0.008
        self.BATCH_SIZE = int(actor_settings["batch"])#64

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_settings = actor_settings
        self.save_folder = save_folder
        # create actor network
        self.state_input, self.action_output, self.net, self.is_training = self.create_network(state_dim, action_dim)

        # create target actor network
        self.target_state_input, self.target_action_output, self.target_update, self.target_is_training = self.create_target_network(
            state_dim, action_dim, self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()

        self.load_network()



    def create_network(self, state_dim, action_dim):
        layer1_size = self.LAYER1_SIZE
        layer2_size = self.LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = tf.Variable(tf.random_uniform([layer2_size, action_dim], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))

        layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training, scope_bn='batch_norm_0',
                                          activation=tf.identity)
        layer1 = tf.matmul(layer0_bn, W1) + b1
        layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training, scope_bn='batch_norm_1',
                                          activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, W2) + b2
        layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training, scope_bn='batch_norm_2',
                                          activation=tf.nn.relu)

        action_output = tf.tanh(tf.matmul(layer2_bn, W3) + b3)

        ##scaling_factor = tf.constant(float(self.actor_settings["decay"]))
        scaling_factor = tf.constant(0.999)
        decay = [param.assign(param * scaling_factor) for param in [W1, b1, W2, b2, W3, b3]]

        action_output = [action_output, decay]

        return state_input, action_output, [W1, b1, W2, b2, W3, b3], is_training

    def create_target_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        is_training = tf.placeholder(tf.bool)
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training, scope_bn='target_batch_norm_0',
                                          activation=tf.identity)

        layer1 = tf.matmul(layer0_bn, target_net[0]) + target_net[1]
        layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training, scope_bn='target_batch_norm_1',
                                          activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, target_net[2]) + target_net[3]
        layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training, scope_bn='target_batch_norm_2',
                                          activation=tf.nn.relu)

        action_output = tf.tanh(tf.matmul(layer2_bn, target_net[4]) + target_net[5])

        return state_input, action_output, target_update, is_training

    def update_target(self):
        self.sess.run(self.target_update)

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output[0], self.net, -self.q_gradient_input)
        self.normalized_parameters_gradients = list(
            map(lambda x: tf.div(x, self.BATCH_SIZE), self.parameters_gradients))
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(
            zip(self.normalized_parameters_gradients, self.net))
        #if self.actor_settings["decay"] is not None and float(self.actor_settings["decay"]) != 1:
        self.optimizer = [self.optimizer] + self.action_output[1]

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.is_training: True
        })

    def get_output_batch(self, state_batch):
        return self.sess.run(self.action_output[0], feed_dict={
            self.state_input: state_batch,
            self.is_training: False
        })

    def create_decay(self, net):
        scaling_factor = tf.constant(self.actor_settings["decay"])
        self.decay = [tf.scalar_mul(scaling_factor, param) for param in net]

    def get_output(self, state, is_training):
        return self.sess.run(self.action_output[0], feed_dict={
            self.state_input: [state],
            self.is_training: False
        })[0]

    def get_t_output_batch(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_is_training: True
        })

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(training_phase,
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5))

    def load_network(self, save_folder=None):
        self.saver = tf.train.Saver()
        path = self._get_path(save_folder)
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, save_folder=None):
        print('saving actor-net...', time_step)
        path = self._get_path(save_folder)
        self.saver.save(self.sess, path, global_step=time_step)

    def _get_path(self, save_folder):
        if save_folder is not None:
            path = save_folder + '/saved_actor_networks/'
        elif self.save_folder is not None:
            path = self.save_folder + '/saved_actor_networks/'
        else:
            path = "experiments/" + self.env_name + "/saved_actor_networks/"
        return path

