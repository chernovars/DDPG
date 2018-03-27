import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math

def key_exists(dict, key):
    try:
        if dict[key] is not None:
            return True
    except KeyError as e:
        return False

def transfer_parameter(dict, key, not_found):
    if key_exists(dict, key):
        param = dict[key]
    else:
        param = not_found
    return param

def get_net_variables(name):
    res = {}
    name_len = len(name)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var.name.startswith(name):
            res[var.name[name_len:]] = var
    return res

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self, sess, state_dim, action_dim, env_name, actor_settings, save_folder):
        # Hyper Parameters
        self.LEARNING_RATE = float(actor_settings["learning_rate"])#1e-3
        self.TAU = float(actor_settings["tau"])#0.008
        self.BATCH_SIZE = int(actor_settings["batch"])#64

        self.BATCH_NORM = transfer_parameter(actor_settings, "bn", False)
        self.DECAY = transfer_parameter(actor_settings, "decay", 1.0)

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_settings = actor_settings
        self.save_folder = save_folder
        # create actor network
        self.state_input, self.action_output, self.W, self.b, self.is_training = self.create_network(state_dim, action_dim, actor_settings)

        self.actor_net_params = get_net_variables("actor")
        self.t_actor_net_params = None
        self.actor_ema_params = None

        # create target actor network
        self.target_state_input, self.target_action_output, self.tW, self.tb, \
        self.target_is_training, self.copy_ops, self.ema_update, self.target_update, = \
            self.create_target_network(state_dim, action_dim, actor_settings, self.W, self.b)

        # create weight decay method
        self.decay_ops = self.create_decay(self.W)

        # define training rules
        self.create_training_method(post_train_ops=[self.ema_update] + self.target_update)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.copy_ops)

        self.update_target()
        self.load_network()

    def create_network(self, state_dim, action_dim, actor_settings, scope="actor"):
        with tf.variable_scope(scope):
            state_input = tf.placeholder("float", [None, state_dim])
            is_training = tf.placeholder(tf.bool)

            layer = tf.layers.dense(state_input, units=state_dim)
            bn = tf.layers.batch_normalization(layer, training=is_training)
            activation = tf.nn.relu(bn)

            for i in range(0, len(actor_settings["layers"])):
                layer = tf.layers.dense(activation, units=actor_settings["layers"][i])
                bn = tf.layers.batch_normalization(layer, training=is_training)
                activation = tf.nn.relu(bn)

            layer = tf.layers.dense(activation, units=action_dim)
            action_output = tf.nn.tanh(layer)

            W = []
            b = []
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if "kernel" in var.name:
                    W.append(var)
                if "bias" in var.name:
                    b.append(var)

        return state_input, action_output, W, b, is_training

    def create_target_network(self, state_dim, action_dim, actor_settings, W, b):

        state_input, action_output, tW, tb, is_training = self.create_network(state_dim, action_dim, actor_settings, scope="t_actor")

        self.t_actor_net_params = get_net_variables("t_actor")
        #op for copy weights from actor to target actor
        copy_ops = [self.t_actor_net_params[key].assign(self.actor_net_params[key]) for key in self.actor_net_params]

        ema = tf.train.ExponentialMovingAverage(decay= 1 - self.TAU)

        # op for calculating ema
        ema_update = ema.apply(list(self.actor_net_params.values()))

        self.actor_ema_params = {key: ema.average(self.actor_net_params[key]) for key in self.actor_net_params}
        # copy weights from running averages to target actor
        target_update = [self.t_actor_net_params[key].assign(self.actor_ema_params[key]) for key in self.actor_ema_params]

        return state_input, action_output, tW, tb, is_training, copy_ops, ema_update, target_update

    def create_decay(self, params):
        scaling_factor = tf.constant(self.DECAY)
        return [param.assign(param * scaling_factor) for param in params]

    def create_training_method(self, post_train_ops=[]):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.W + self.b, -self.q_gradient_input)
        self.normalized_parameters_gradients = list(
            map(lambda x: tf.div(x, self.BATCH_SIZE), self.parameters_gradients))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='actor')):
            self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(
                                                            zip(self.normalized_parameters_gradients, self.W + self.b))
        self.train_ops = [self.optimizer] + post_train_ops

        if self.DECAY != 1:
            self.train_ops += self.decay_ops


    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.train_ops, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.is_training: True
        })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch,
            self.is_training: True
        })

    def action(self, state, is_training):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state],
            self.is_training: is_training
        })[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_is_training: True #so that we always use batch mean and variance in batch norm for target network, because target net is used only for training
        })

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

