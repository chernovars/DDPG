import filter_env
from ddpg import *

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

import os, errno
import sys
import csv

import shutil

def generatePlot(x, y = None, title = "", labels = None, save_folder = None, show_picture = True, color = 'b' ):
    plt.figure()
    if x is None:
        print("def generatePlot: no x specified")
        raise AssertionError

    if y:
        plt.plot(x, y, color)
    else:
        plt.plot(x, color)

    if labels is None:
        plt.xlabel('x label')
        plt.ylabel('y label')
    else:
        plt.xlabel(labels[x])
        plt.ylabel(labels[y])

    if title:
        plt.title(title)

    if save_folder:
        plt.savefig(save_folder)

    if show_picture:
        plt.show(block=False)

def generatePlot(*y, x = None, title = "", labels = None, legend=None, save_folder = None, show_picture = True, color = 'b'):

    colors = ['b','y','r','c','m','g','k']

    plt.figure()
    plot_args = []

    legths_y = [len(i) for i in y]
    if min(legths_y) != max(legths_y):
        print("Lenghts of y-lists should be the same")
        raise AssertionError


    if x is None and len(y) > 0:
        x = list(range(0, len(y[0])))

    if legend is None or len(legend) != len(y):
        legend = ["y" + str(y.index(i)) for i in y]

    if len(y) > 1 :
        for i in range(0, len(y)):
            plt.plot(x, y[i], colors[i%(len(colors))], label = legend[i] )
    elif len(y) == 1 :
        plot_args += [x, y[0], color]

    plt.plot(plot_args)

    if labels is None:
        plt.xlabel('x label')
        plt.ylabel('y label')
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if title:
        plt.title(title)

    plt.legend()
    if save_folder:
        plt.savefig(save_folder)

    if show_picture:
        plt.show(block=False)


class DataCollector:
    def __init__(self, save_folder, env_name):
        self.time_point_first = time.time()
        self.time_point_last = time.time()
        self.time_list = []
        self.steps_list = []
        self.rewards_list = []
        self.EMA_reward = 0
        self.EMA_rewards_list = []
        self.save_folder = save_folder
        self.env_name = env_name
        if os.path.exists(self.save_folder + self.env_name):
            self.rewards_list = self.load_rewards_list(self.save_folder + self.env_name)
            self.EMA_rewards_list = self.listToEMA(self.rewards_list)


        self.solved = False

    def _collect_data(self, step, episode_reward):
        self.time_point_last = time.time()
        print("time spent: ", self.time_point_last - self.time_point_first)
        self.time_list.append(self.time_point_last - self.time_point_first)
        print("steps made: ", step)
        self.steps_list.append(step)
        print("reward:", episode_reward)
        self.rewards_list.append(episode_reward)
        self.EMA_reward = 0.5 * episode_reward + 0.95 * self.EMA_reward
        self.EMA_rewards_list.append(self.EMA_reward)
        self.time_point_first = self.time_point_last

    def save_rewards_list(self):
        with open(self.save_folder + self.env_name, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([len(self.rewards_list)])
            for val in self.rewards_list:
                writer.writerow([val])

    def load_rewards_list(self, file):
        if not os.path.isfile(file):
            print("def load_rewards_list: no path specified")
            exit(1)

        with open(file, "r") as f:
            n = f.readline()
            x = f.readlines()
            x = [float(i.strip()) for i in x ]
            self.rewards_list = x

        return self.rewards_list

    def listToEMA(self, orig, tau = 0.98):
        EMA = []
        EMA.append(orig[0] * (1 - tau))
        for i in range(1, len(orig)):
            EMA.append(EMA[i-1] * tau + orig[i] * (1 - tau))

        return EMA

class World:

    def __init__(self, RENDER_STEP=False, RENDER_delay=0, TRAIN=True, NOISE=True):
        self.ENV_NAME = 'InvertedDoublePendulum-v1'
        self.EPISODES = 100000000  # 100000
        self.TIME_LIMIT = 0
        self.TEST = 10
        self.RENDER_STEP = RENDER_STEP
        self.RENDER_TEST_EPISODE = False
        self.RENDER_delay = RENDER_delay  # 0.1
        self.TRAIN = TRAIN
        self.NOISE = NOISE
        self.TEST_ON_EPISODE = 200
        self.VIDEO_ON_EPISODE = 20000  # 4*TEST_ON_EPISODE
        self.SHOW_PLOT = False
        self.RECORD_VIDEO = False

        self.UNTIL_SOLVED = False
        self.AVG_REWARD = 0
        self.OVER_LAST = 0

        self.ACTOR_SETTINGS = []
        self.CRITIC_SETTINGS = []

    def main(self, save_folder, data_save=True):
        #myplot = plot.Plot() # TODO: real-time plotting for state analysys

        start_time = 0
        if self.TIME_LIMIT > 0:
            start_time = time.time()

        env_real = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))
        agent = DDPG(env_real, self.TRAIN, self.NOISE, self.ENV_NAME, self.ACTOR_SETTINGS, self.CRITIC_SETTINGS, save_folder)

        if self.RECORD_VIDEO:
            monitor = gym.wrappers.Monitor(env_real, 'experiments/' + self.ENV_NAME, (lambda i: (i % self.VIDEO_ON_EPISODE) == 0), resume=True)
            env = monitor
        else:
            env = env_real
        data_collector = DataCollector(save_folder, self.ENV_NAME)

        try:
            for episode in range(self.EPISODES):
                state = env.reset()
                print("episode:", episode)
                steps = 0
                episode_reward = 0
                # Simulate + train
                for step in range(env.spec.timestep_limit):
                    if self.RENDER_STEP:
                        self.slow_render(env)
                        if self.RENDER_delay > 0:
                            time.sleep(self.RENDER_delay)
                    steps += 1
                    if self.NOISE:
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
                if (episode % self.TEST_ON_EPISODE) == 0 and episode > self.TEST_ON_EPISODE:
                    if agent.TRAIN:
                        #agent.save(episode, save_folder)
                        if self._testing(env, agent, episode, data_collector, self.ENV_NAME):
                            self.finish(agent, env, episode, save_folder, data_collector)
                            return
                if self.TIME_LIMIT > 0 and (time.time() - start_time) > self.TIME_LIMIT:
                    self.finish(agent, env,episode, save_folder, data_collector, data_save)
                    return
            self.finish(agent, env, episode, save_folder, data_collector, data_save)
            return
        except KeyboardInterrupt:
            self.finish(agent, env, episode, save_folder, data_collector, data_save)
            return

    def finish(self, agent, env, episode, save_folder, data_collector, data_save=False):
        if not data_save:
            agent.save(episode, save_folder)
            data_collector.save_rewards_list()
        if self.RECORD_VIDEO:
            env.close()
        agent.close()


    def slow_render(self, env):
        env.render()
        time.sleep(0.015)

    def renderEpisode(self, env, agent):
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

    def _plotting(self, x, data_collector, env_name):
        today = datetime.date.today().strftime("%d-%m-%Y")

        try:
            os.makedirs("./experiments/" + env_name + "/Pictures")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        path = "./experiments/" + env_name + "/Pictures/"
        generatePlot(np.array(data_collector.stepslist), x=x, title=today, labels=["episodes", "steps"], save_folder=path + "steps" + today, show_picture=True)
        generatePlot(np.array(data_collector.rewardslist), x=x, title="", labels=["episodes", "steps"], save_folder=path + "rewards" + today,
                     show_picture=True, color='r')
        generatePlot(np.array(data_collector.EMA_rewards_list), x=x, title="", labels=["episodes", "steps"], save_folder=path + "filtered_rewards" + today,
                     show_picture=True, color='r')
        plt.show(block=False)

    def _testing(self, env, agent, episode, data_collector, env_name):
        x = np.arange(1, episode + 2, 1)
        if self.SHOW_PLOT:
            self._plotting(x, data_collector, env_name)
        if self.RENDER_TEST_EPISODE:
            self.renderEpisode(env, agent)
        total_reward = 0
        for i in range(self.TEST):
            state = env.reset()
            for j in range(env.spec.timestep_limit):
                action = agent.action(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / self.TEST
        print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        if self.UNTIL_SOLVED and ave_reward > self.AVG_REWARD:
            total_reward = 0
            for i in range(self.OVER_LAST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / self.OVER_LAST
            print('episode: ', episode, 'Evaluation Average Reward 2:', ave_reward)
            if ave_reward > self.AVG_REWARD:
                return True
        return False



if __name__ == '__main__':
    w = World
    w.main()
