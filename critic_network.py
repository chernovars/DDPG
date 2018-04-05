from network import *
from utils import transfer_parameter
import math

class CriticNetwork(Network):
    """docstring for CriticNetwork

        Critic4

        Uses tf.variables in a function, also variables in the target network are EMA of original net
        Almost the same as critic 4, but has create net funcrion in network2 file
     """

    def __init__(self, sess, state_dim, action_dim, env_name, critic_settings, save_folder):
        print('Running new critic')
        Network.__init__(self, sess, state_dim, action_dim, "critic", critic_settings, save_folder)
        self.L2 = float(critic_settings["l2"]) #0.01

        # create q network
        self.input, self.actions_input, self.output, self.is_training \
            = self.create_q_network(state_dim, action_dim, critic_settings, self.name)
        self.net_params = get_net_variables(self.name)

        # create target q network (the same structure with q network)
        self.create_target_network(state_dim, action_dim, critic_settings, self.net_params, self.name)

        self.create_training_method()

        self.post_init()

    def create_q_network(self, input_dim, action_dim, settings, scope, parameters=None):
        self.actions_layer_number = transfer_parameter(settings, "actions_layer", not_found=2)
        make_new_var = False
        if parameters is None:
            make_new_var = True

        with tf.variable_scope(scope):
            state_input = tf.placeholder("float", [None, input_dim], name='state_input')
            actions_input = tf.placeholder("float", [None, action_dim], name='action_input')
            is_training = tf.placeholder(tf.bool, name='bn_is_training')
            layers = settings["layers"]

            #action input layer
            action_layer_units = layers[self.actions_layer_number-1]

            if self.actions_layer_number > 1:
                f = layers[self.actions_layer_number-1] + action_dim
            else:
                f = input_dim + action_dim
            actions_layer = self.create_dense_layer(actions_input, l_scope="action_input_layer", is_training=is_training,
                                               shape=[action_dim, action_layer_units], f=f, parameters=parameters,
                                               param_name='action', use_bias=False, activate=False)

            #state input layer
            layer = self.create_dense_layer(state_input, l_scope="h_layer_1", shape=[input_dim, layers[0]],
                                       is_training=is_training, parameters=parameters, f=input_dim,
                                            param_name='1', activate=False)
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

                layer = self.create_dense_layer(layer, l_scope="h_layer_" + num, shape=[layers[i-1], layers[i]],
                                           is_training=is_training, parameters=parameters, f=f, param_name=num, activate=False)
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

    def create_training_method(self):
        with tf.variable_scope("critic_train"):
            # Define training optimizer
            self.y_input = tf.placeholder("float", [None, 1])
            weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in get_params(self.name, "kernel")])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.output)) + weight_decay
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='critic')):
                self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
            self.action_gradients = tf.gradients(self.output, self.actions_input)

    def create_target_network(self, *args):
        if len(args) != 5:
            raise TypeError

        input_state_dim, input_actions_dim, settings, net_params, old_scope = args

        self.create_target_update(net_params)

        self.t_input, self.t_actions_input, self.t_output, self.t_is_training \
            = self.create_q_network(input_state_dim, input_actions_dim, settings,
                                    scope="t_" + old_scope, parameters=self.ema_net_params)

    def train(self, y_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.input: state_batch,
            self.actions_input: action_batch,
            self.is_training: True
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.input: state_batch,
            self.actions_input: action_batch,
            self.is_training: True
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.t_output, feed_dict={
            self.t_input: state_batch,
            self.t_actions_input: action_batch,
            self.t_is_training: True
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.output, feed_dict={
            self.input: state_batch,
            self.actions_input: action_batch,
            self.is_training: True
        })
