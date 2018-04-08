import math

import tensorflow as tf
from utils import transfer_parameter

def get_net_variables(scope):
    res = {}
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var.name.startswith(scope):
            short_name = var.name.split("/")
            for s in short_name:
                if "kernel" in s or "bias" in s:
                    s = s.split(":")[0]
                    res[s] = var
    return res

def get_params(net_scope, param_name):
    res = []
    vars = get_net_variables(net_scope)
    for v in vars:
        if param_name in v:
            res.append(vars[v])

    sorted_res = []
    for i in range(len(res)-1):
        for v in res:
            if str(i+1) in v.name:
                sorted_res.append(v)
                break
    for v in res:
        if "out" in v.name:
            sorted_res.append(v)
    return sorted_res

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

    def create_dense_layer(self, input, l_scope, shape, f, is_training=True, parameters=None, param_name="", use_bias=True, activate=True):
        with tf.variable_scope(l_scope):
            if parameters is None:
                W = self.variable(shape, f, name='kernel_' + param_name)
            else:
                W = parameters['kernel_' + param_name]
            if use_bias:
                if parameters is None:
                    b = self.variable([shape[1]], f, name='bias_' + param_name)
                else:
                    b = parameters['bias_' + param_name]

            _layer = tf.matmul(input, W)
            if use_bias:
                _layer = _layer + b
            if self.BATCH_NORM:
                _layer = tf.layers.batch_normalization(_layer, training=is_training, name=l_scope + "bn")
            if activate:
                _layer = tf.nn.relu(_layer, name='act_'+param_name)
        return _layer


    def create_network(self, input_dim, output_dim, settings, scope, parameters=None):
        with tf.variable_scope(scope):
            state_input = tf.placeholder("float", [None, input_dim], name='state_input')
            is_training = tf.placeholder(tf.bool, name='bn_is_training')
            layers = settings["layers"]

            # state input layer
            layer = self.create_dense_layer(state_input, l_scope="h_layer_1", shape=[input_dim, layers[0]],
                                       is_training=is_training, parameters=parameters, f=input_dim, param_name='1')
            # hidden layers
            for i in range(1, len(settings["layers"])):
                num = str(i + 1)
                layer = self.create_dense_layer(layer, l_scope="h_layer_" + num, shape=[layers[i - 1], layers[i]],
                                                is_training=is_training, parameters=parameters, f=layers[i - 1],
                                                param_name=num)
            # output layer
            with tf.variable_scope("output_layer"):
                prev_l_dim = layers[len(layers) - 1]
                if parameters is None:
                    W_out = tf.Variable(tf.random_uniform([prev_l_dim, output_dim], -3e-3, 3e-3), name='kernel_out')
                    b_out = tf.Variable(tf.random_uniform([output_dim], -3e-3, 3e-3), name='bias_out')
                else:
                    W_out = parameters['kernel_out']
                    b_out = parameters['bias_out']
                layer = tf.matmul(layer, W_out) + b_out
            output = tf.nn.tanh(layer, name='tanh_output')

        return state_input, output, is_training

    def create_target_network(self, *args):
        if len(args) != 5:
            raise TypeError

        input_dim, output_dim, settings, net_params, old_scope = args
        self.create_target_update(net_params)

        self.t_input, self.t_output, self.t_is_training, \
            = self.create_network(input_dim, output_dim, settings, scope="t_" + old_scope, parameters=self.ema_net_params)

    def create_target_update(self, net_params):
        with tf.variable_scope(self.name + "_target_update"):
            ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
            self.target_update = ema.apply(list(net_params.values()))
            self.ema_net_params = {key: ema.average(net_params[key]) for key in net_params}

    def update_target(self):
        self.sess.run(self.target_update)

    def get_output_batch(self, input_batch):
        return self.sess.run(self.output, feed_dict={
            self.input: input_batch,
            self.is_training: True #was False
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

