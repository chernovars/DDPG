from network import *
from utils import transfer_parameter
import math

class CriticNetwork(Network):
    """docstring for CriticNetwork"""

    def __init__(self, sess, state_dim, action_dim, env_name, critic_settings, save_folder):
        print('Running new critic')
        Network.__init__(self, sess, state_dim, action_dim, "critic", critic_settings, save_folder)
        self.L2 = float(critic_settings["l2"]) #0.01

        # create q network
        self.input, self.actions_input, self.output, self.is_training, self.W, self.b \
            = self.create_q_network(state_dim, action_dim, critic_settings, self.name)
        self.net_params = get_net_variables(self.name)

        # create target q network (the same structure with q network)
        self.create_target_network(state_dim, action_dim, critic_settings, self.net_params, self.name)

        self.create_training_method()

        self.post_init()

    def create_q_network(self, input_dim, action_dim, settings, scope):
        self.actions_layer_number = transfer_parameter(settings, "actions_layer", not_found=2)
        W = []
        b = []

        with tf.variable_scope(scope):
            state_input = tf.placeholder("float", [None, input_dim], name='state_input')
            actions_input = tf.placeholder("float", [None, action_dim], name='action_input')
            is_training = tf.placeholder(tf.bool, name='bn_is_training')
            layers = settings["layers"]

            #action input layer
            action_layer_units = layers[self.actions_layer_number-1]

            with tf.variable_scope("action_input_layer"):
                if self.actions_layer_number > 1:
                    f = layers[self.actions_layer_number-1] + action_dim
                else:
                    f = input_dim + action_dim

                W_action = self.variable([action_dim, action_layer_units], f, name='kernel_action')
                actions_layer = tf.matmul(actions_input, W_action)
                if self.BATCH_NORM:
                    actions_layer = tf.layers.batch_normalization(actions_layer, training=is_training, name='a_input_layer_bn')

            #state input layer
            with tf.variable_scope("h_layer_1"):
                W1 = self.variable([input_dim, layers[0]], input_dim, name='kernel_1')
                b1 = self.variable([layers[0]], input_dim, name='bias_1')
                layer = tf.matmul(state_input, W1) + b1
                if self.BATCH_NORM:
                    layer = tf.layers.batch_normalization(layer, training=is_training, name='bn_input_layer')

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

                with tf.variable_scope(l_name):
                    Wi = self.variable([layers[i-1], layers[i]], f, name='kernel_'+num)
                    bi = self.variable([layers[i]], f, name='bias_'+num)
                    layer = tf.matmul(layer, Wi) + bi
                    if self.BATCH_NORM:
                        layer = tf.layers.batch_normalization(layer, training=self.is_training, name='bn_'+l_name)

                if i == (self.actions_layer_number-1):
                    layer = tf.add(layer, actions_layer, name="s_a_sum_"+l_name)

                layer = tf.nn.relu(layer, name="act_"+l_name)

            #output layer
            with tf.variable_scope("output_layer"):
                prev_l_dim = layers[len(layers)-1]
                Wout = tf.Variable(tf.random_uniform([prev_l_dim, 1], -3e-3, 3e-3), name='kernel_out')
                bout = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name='bias_out')
                layer = tf.matmul(layer, Wout) + bout

            output = tf.identity(layer, name='identity_output')
            W, b = self.get_W_b(scope=scope)

        return state_input, actions_input, output, is_training, W, b

    def create_training_method(self):
        with tf.variable_scope("critic_train"):
            # Define training optimizer
            self.y_input = tf.placeholder("float", [None, 1])
            weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.W])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.output)) + weight_decay
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='critic')):
                self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
            self.action_gradients = tf.gradients(self.output, self.actions_input)

    def create_target_network(self, *args):
        if len(args) != 5:
            raise TypeError

        input_state_dim, input_actions_dim, settings, net_params, old_scope = args

        self.t_input, self.t_actions_input, self.t_output, self.t_is_training, _, _ \
            = self.create_q_network(input_state_dim, input_actions_dim, settings, scope="t_" + old_scope)

        self.create_target_update(net_params)

    def update_target(self):
        self.sess.run(self.ema_update)
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.input: state_batch,
            self.actions_input: action_batch,
            #self.is_training: True
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.input: state_batch,
            self.actions_input: action_batch,
            #self.is_training: True
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.t_output, feed_dict={
            self.t_input: state_batch,
            self.t_actions_input: action_batch,
            #self.t_is_training: False#True
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.output, feed_dict={
            self.input: state_batch,
            self.actions_input: action_batch,
            #self.is_training:False #True
        })
