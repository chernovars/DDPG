from network import *
from utils import transfer_parameter
from ddpg import BATCH_SIZE
from decay import *

class ActorNetwork(Network):
    """docstring for CriticNetwork

        Critic4

        Uses tf.variables in a function, also variables in the target network are EMA of original net
    """

    def __init__(self, sess, state_dim, action_dim, env_name, actor_settings, save_folder, observations="states"):
        print('Running new actor')
        Network.__init__(self, sess, state_dim, action_dim, "actor", actor_settings, save_folder)
        self.DECAY = transfer_parameter(actor_settings, "decay", not_found=1.0)
        self.DECAY_SCHEME = transfer_parameter(actor_settings, "decay_scheme", "not_set")
        self.BATCH_SIZE = transfer_parameter(actor_settings, "batch", BATCH_SIZE)
        self.OBSERVATIONS = observations

        # create actor network
        self.input, self.output, self.is_training \
            = self.create_network(state_dim, action_dim, actor_settings, self.name)
        self.net_params = get_net_variables(self.name)

        # create target actor network
        self.create_target_network(state_dim, action_dim, actor_settings, self.net_params, self.name)

        # create weight decay method
        self.W = get_params(self.name, "kernel")
        self.decay_ops = self.create_decay(self.W)

        self.create_training_method()

        self.post_init()

    def create_decay(self, params):
        return create_decay_scheme(self.DECAY, params, scheme=self.DECAY_SCHEME)

    def create_training_method(self, post_train_ops=[]):
        with tf.variable_scope("actor_train"):
            self.q_gradient_input = tf.placeholder("float", [None, self.output_dim], name='q_gradient_input')
            params = list(self.net_params.values())
            self.parameters_gradients = tf.gradients(self.output, params, -self.q_gradient_input, name="sum")
            self.normalized_parameters_gradients = list(
                map(lambda x: tf.div(x, self.BATCH_SIZE), self.parameters_gradients))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='actor')):
                self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(
                                                                zip(self.normalized_parameters_gradients, self.net_params.values()))
            self.train_ops = [self.optimizer] + post_train_ops

            if self.DECAY_SCHEME is not "not_set":
                self.train_ops += self.decay_ops

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.train_ops, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.input: state_batch,
            self.is_training: True
        })