import numpy as np


# import gym

def makeFilteredEnv(env):
  """ Create a new environment class with actions and states normalized to [-1,1] """
  #env = gym.make(ENV_NAME)
  action_space = env.action_space
  observation_space = env.observation_space
  '''if not type(action_space) == gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(observation_space) == gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')
  '''

  env_type = type(env)

  class FilteredEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__) # transfer properties

      # Observation space
      if np.any(observation_space.high < 1e10):
        high = observation_space.high
        low = observation_space.low
        space = high-low
        self.o_space_mean = (high + low) / 2.
        self.o_half_space = space / 2.
      else:
        self.o_space_mean = np.zeros_like(observation_space.high)
        self.o_half_space = np.ones_like(observation_space.high)

      # Action space
      high = action_space.high
      low = action_space.low
      space = (high-low)
      self.a_space_mean = (high + low) / 2.
      self.a_half_space = space / 2.

      # Rewards
      self.r_sc = 0.1
      self.r_c = 0.0

      # Special cases
      '''
      if self.spec.id == "Reacher-v1":
        print "is Reacher!!!"
        self.o_sc[6] = 40.
        self.o_sc[7] = 20.
        self.r_sc = 200.
        self.r_c = 0.
    '''
      # Check and assign transformed spaces
      self.observation_space = gym.spaces.Box(self.filter_observation(observation_space.low),
                                              self.filter_observation(observation_space.high))

      self.action_space = gym.spaces.Box(-np.ones_like(action_space.high),np.ones_like(action_space.high))

      def assertEqual(a,b): assert np.all(a == b), "{} != {}".format(a,b)

      assertEqual(self.filter_action(self.action_space.low), action_space.low)
      assertEqual(self.filter_action(self.action_space.high), action_space.high)

    def filter_observation(self,obs):
      return (obs - self.o_space_mean) / self.o_half_space

    def filter_action(self,action):
      return self.a_half_space * action + self.a_space_mean

    def filter_reward(self,reward):
      ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
      return self.r_sc*reward+self.r_c

    def step(self,action):

      ac_normalized = np.clip(self.filter_action(action),self.action_space.low,self.action_space.high)

      obs, reward, term, info = env_type.step(self, ac_normalized) # super function

      obs_f = self.filter_observation(obs)

      return obs_f, reward, term, info

  fenv = FilteredEnv()

  print('True action space: ' + str(action_space.low) + ', ' + str(action_space.high))
  print('True state space: ' + str(observation_space.low) + ', ' + str(observation_space.high))
  print('Filtered action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
  print('Filtered state space: ' + str(fenv.observation_space.low) + ', ' + str(fenv.observation_space.high))

  return fenv