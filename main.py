import filter_env
from ddpg import *
import gc
import time
import datetime
import numpy as np
#import plot
import matplotlib.pyplot as plt
import os, errno

gc.enable()

ENV_NAME = 'InvertedDoublePendulum-v1'
EPISODES = 100000000  # 100000
TEST = 10
RENDER_STEP = False
RENDER_delay = 0  # 0.1
TRAIN = True
NOISE = False
TEST_ON_EPISODE = 10000
VIDEO_ON_EPISODE = 20000 #4*TEST_ON_EPISODE


def slow_render(env):
    env.render()
    time.sleep(0.015)

def renderEpisode(env, agent):
    state = env.reset()
    # print "episode:",episode
    # Train
    for step in range(env.spec.timestep_limit):
        env.render()
        action = agent.noise_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.perceive(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

def _plotting(x, data_collector, env_name):
    today = datetime.date.today().strftime("%d-%m-%Y")

    try:
        os.makedirs("./experiments/"+env_name+"/Pictures")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    plt.figure(1)
    plt.plot(x, np.array(data_collector.stepslist), 'b')
    plt.savefig("./experiments/"+env_name+"/Pictures/steps"+today)
    plt.figure(2)
    plt.plot(x, np.array(data_collector.rewardslist), 'r')
    plt.savefig("./experiments/"+env_name+"/Pictures/rewards"+today)
    plt.figure(3)
    plt.plot(x, np.array(data_collector.meanrewardslist))
    plt.savefig("./experiments/"+env_name+"/Pictures/filtered_rewards"+today)
    plt.show(block = False)

def _testing(env, agent, episode, data_collector, env_name):
    x = np.arange(1, episode + 2, 1)
    _plotting(x, data_collector, env_name)
    if agent.TRAIN == True:
        agent.save(episode)
    if not RENDER_STEP:
        renderEpisode(env, agent)
    total_reward = 0
    for i in range(TEST):
        state = env.reset()
        for j in range(env.spec.timestep_limit):
            action = agent.action(state)  # direct action for test
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    ave_reward = total_reward / TEST
    print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

class DataCollector:
    def __init__(self):
        self.time_point_first = time.time()
        self.time_point_last = time.time()
        self.timelist = []
        self.stepslist = []
        self.rewardslist = []
        self.mean_reward = 0
        self.meanrewardslist = []

    def _collect_data(self, step, episode_reward):
        self.time_point_last = time.time()
        print("time spent: ", self.time_point_last - self.time_point_first)
        self.timelist.append(self.time_point_last - self.time_point_first)
        print("steps made: ", step)
        self.stepslist.append(step)
        print("reward:", episode_reward)
        self.rewardslist.append(episode_reward)
        self.mean_reward = 0.5 * episode_reward + 0.95 * self.mean_reward
        self.meanrewardslist.append(self.mean_reward)
        self.time_point_first = self.time_point_last

def main():
    #myplot = plot.Plot() # TODO: real-time plotting for state analysys
    env_real = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env_real, TRAIN, NOISE, ENV_NAME)
    monitor = gym.wrappers.Monitor(env_real, 'experiments/' + ENV_NAME, (lambda i: (i % VIDEO_ON_EPISODE) == 0), resume=True)
    env = monitor
    data_collector = DataCollector()

    try:
        for episode in range(EPISODES):
            state = env.reset()
            print("episode:", episode)
            steps = 0
            episode_reward = 0
            # Simulate + train
            for step in range(env.spec.timestep_limit):
                if RENDER_STEP:
                    slow_render(env)
                    if RENDER_delay > 0:
                        time.sleep(RENDER_delay)
                steps += 1
                if NOISE:
                    action = agent.noise_action(state)
                else:
                    action = agent.action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                agent.perceive(state, action, reward, next_state, done)
                state = next_state
                if done:
                    data_collector._collect_data(steps, episode_reward)
                    break
            # Testing:
            if episode % TEST_ON_EPISODE == 0 and episode > 1:
                _testing(env, agent, episode, data_collector, ENV_NAME)
    except KeyboardInterrupt:
        monitor.close()

main()
