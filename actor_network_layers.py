from network import *
from ddpg import BATCH_SIZE
from utils import transfer_parameter


class ActorNetwork(Network):
    """docstring for ActorNetwork"""
    def __init__(self, sess, state_dim, action_dim, env_name, actor_settings, save_folder):
        Network.__init__(self, sess, state_dim, action_dim, "actor", actor_settings, save_folder)
        self.DECAY = transfer_parameter(actor_settings, "decay", not_found=1.0)
        self.BATCH_SIZE = transfer_parameter(actor_settings, "batch", BATCH_SIZE)
        # create actor network
        self.input, self.output, self.is_training, self.W, self.b \
            = self.create_network(state_dim, action_dim, actor_settings, self.name)
        self.net_params = get_net_variables(self.name)

        # create target actor network
        self.create_target_network(state_dim, action_dim, actor_settings, self.net_params, self.name)

        # create weight decay method
        self.decay_ops = self.create_decay(self.W)

        # define training rules
        self.create_training_method(post_train_ops=[self.ema_update] + self.target_update)

        self.post_init()

    def create_decay(self, params):
        with tf.variable_scope("param_decay"):
            scaling_factor = tf.constant(self.DECAY)
            return [param.assign(param * scaling_factor) for param in params]

    def create_training_method(self, post_train_ops=[]):
        with tf.variable_scope("actor_train"):
            self.q_gradient_input = tf.placeholder("float", [None, self.output_dim])
            self.parameters_gradients = tf.gradients(self.output, self.W + self.b, -self.q_gradient_input)
            self.normalized_parameters_gradients = list(
                map(lambda x: tf.div(x, self.BATCH_SIZE), self.parameters_gradients))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='actor')):
                self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(
                                                                zip(self.normalized_parameters_gradients, self.W + self.b))
            self.train_ops = [self.optimizer] + post_train_ops

            if self.DECAY != 1:
                self.train_ops += self.decay_ops

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.train_ops, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.input: state_batch,
            self.is_training: True
        })