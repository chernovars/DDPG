import math

import tensorflow as tf
from utils import transfer_parameter

def get_net_variables(name):
    res = {}
    name_len = len(name)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var.name.startswith(name):
            res[var.name[name_len:]] = var
    return res

def get_net_variables2(name):
    res = {}
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var.name.startswith(name):
            short_name = var.name.split("/")[-1]
            short_name = short_name.split(":")[0]
            res[short_name] = var
    return res


class Network:
    def __init__(self, sess, input_dim, output_dim, name, settings, save_folder):
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.settings = settings
        self.save_folder = save_folder

        # Hyper Parameters
        self.LEARNING_RATE = float(settings["learning_rate"])
        self.TAU = float(settings["tau"])
        self.BATCH_NORM = transfer_parameter(settings, "bn", False)

        self.output = None
        self.input = None
        self.t_input = None
        self.t_output = None
        self.copy_ops = None
        self.ema_update = None
        self.target_update = None
        self.is_training = None
        self.t_is_training = None

        self.net_params = None
        self.ema_net_params = None
        self.t_net_params = None

    def post_init(self):
        with tf.variable_scope("init_" + self.name):
            self.sess.run(tf.global_variables_initializer())
        self.update_target()
        self.load_network()

    def create_network(self, input_dim, output_dim, settings, scope):
        with tf.variable_scope(scope):
            input = tf.placeholder("float", [None, input_dim], name="input")
            is_training = tf.placeholder(tf.bool, name="bn_is_training")

            layer = tf.layers.dense(input, units=settings["layers"][0], name="input_layer")
            if self.BATCH_NORM:
                layer = tf.layers.batch_normalization(layer, training=is_training, name="bn_input_layer")
            activation = tf.nn.relu(layer, name="activation_input")

            for i in range(1, len(settings["layers"])):
                l_name = "h_layer_" + str(i + 1)
                layer = tf.layers.dense(activation, units=settings["layers"][i], name=l_name)
                if self.BATCH_NORM:
                    layer = tf.layers.batch_normalization(layer, training=is_training, name="bn_"+l_name)
                activation = tf.nn.relu(layer, name="act_"+l_name)

            layer = tf.layers.dense(activation, units=output_dim, name="output_layer")
            output = tf.nn.tanh(layer, name='output_activation')
            W, b = self.get_W_b(scope=scope)

        return input, output, is_training, W, b

    def create_target_network(self, *args):
        if len(args) != 5:
            raise TypeError

        input_dim, output_dim, settings, net_params, old_scope = args

        self.t_input, self.t_output, self.t_is_training, _, _ \
            = self.create_network(input_dim, output_dim, settings, scope="t_" + old_scope)

        self.create_target_update(net_params)

    def create_target_update(self, net_params):
        with tf.variable_scope(self.name + "_target_update"):
            self.t_net_params = get_net_variables("t_" + self.name)
            # op for copy weights from actor to target actor
            self.copy_ops = [self.t_net_params[key].assign(net_params[key]) for key in net_params]

            ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)

            # op for calculating ema
            a = list(net_params.values())
            self.ema_update = ema.apply(a)

            self.ema_net_params = {key: ema.average(net_params[key]) for key in net_params}
            # copy weights from running averages to target actor
            self.target_update = [self.t_net_params[key].assign(self.ema_net_params[key]) for key in self.ema_net_params]

    def update_target(self):
        self.sess.run(self.target_update)

    def get_output_batch(self, input_batch):
        return self.sess.run(self.output, feed_dict={
            self.input: input_batch,
            self.is_training: False
        })

    def get_output(self, input, is_training):
        return self.sess.run(self.output, feed_dict={
            self.input: [input],
            self.is_training: False#is_training
        })[0]

    def get_t_output_batch(self, state_batch):
        return self.sess.run(self.t_output, feed_dict={
            self.t_input: state_batch,
            self.t_is_training: True    # so that we always use batch mean and variance in batch norm for target network, because target net is used only for training
        })

    # f fan-in size
    def variable(self, shape, f, name=None):
        if name is not None:
            return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)), name=name)
        else:
            return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

    '''def get_t_output_batch(self, state_batch):
        return self.sess.run(self.t_output, feed_dict={
            self.t_input: state_batch
        })'''

    def load_network(self, save_folder=None):
        with tf.variable_scope("save_" + self.name):
            self.saver = tf.train.Saver()
            path = self._get_path(save_folder)
            checkpoint = tf.train.get_checkpoint_state(path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

    def save_network(self, time_step, save_folder=None):
        print('saving '+ self.name + ' net...', time_step)
        path = self._get_path(save_folder)
        self.saver.save(self.sess, path, global_step=time_step)

    def _get_path(self, save_folder):
        first_folder = '/saved_' + self.name + '_networks/'
        if save_folder is not None:
            path = save_folder + first_folder
        elif self.save_folder is not None:
            path = self.save_folder + first_folder
        else:
            path = first_folder
        return path

    def get_W_b(self, scope):
        W = []
        b = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            if "kernel" in var.name:
                W.append(var)
            if "bias" in var.name:
                b.append(var)
        return W, b


