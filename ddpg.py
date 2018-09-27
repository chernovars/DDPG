# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung, Arseniy Chernov
# Date: 2017.11.29
# -----------------------------------
import tensorflow as tf
import numpy as np
from pycronos.agent.DDPG.ou_noise import OUNoise
from pycronos.agent.DDPG.replay_buffer import ReplayBuffer
import time
# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


# Debug Parameters:

SAVE_STEP_THRESHOLD = 3000


class DDPG:
    """docstring for DDPG"""
    def __init__(self, env, train, noise, env_name, actor_settings, critic_settings, save_folder):
        self.TRAIN = train
        self.name = 'DDPG' # name for uploading results
        self.env_name = env_name
        self.environment = env
        self.NOISE = noise
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0] #17
        self.action_dim = env.action_space.shape[0] #6
        self.time_step = 1
        self.sess = tf.InteractiveSession()

        if actor_settings["bn"] == 'True':
            from pycronos.agent.DDPG.actor_network_bn import ActorNetwork
        else:
            from pycronos.agent.DDPG.actor_network import ActorNetwork

        if critic_settings["bn"] == 'True':
            from pycronos.agent.DDPG.critic_network_bn import CriticNetwork
        else:
            from pycronos.agent.DDPG.critic_network import CriticNetwork

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.env_name, actor_settings, save_folder)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.env_name, critic_settings, save_folder)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, load=True, env_name=self.env_name, save_folder=save_folder)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        #print("train step",self.time_step)
        self.time_step += 1
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        #action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_target_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_target_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action+self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            #print("That's why it is slow!!!!")
            #time1= time.time()
            if self.TRAIN == True:
                self.train()
            #print("Training took ",time.time()-time1)
        '''if self.time_step % SAVE_STEP_THRESHOLD == 0:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)
'''
        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def save(self, episode_number, save_folder):
        self.actor_network.save_network(episode_number, save_folder)
        self.critic_network.save_network(episode_number, save_folder)
        self.replay_buffer.save_buffer(save_folder)

    def close(self):
        tf.reset_default_graph()
        self.sess.close()








