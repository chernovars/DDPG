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


class DataCollector:
    def __init__(self, save_folder):
        self.time_point_first = time.time()
        self.time_point_last = time.time()
        self.timelist = []
        self.stepslist = []
        self.rewardslist = []
        self.EMA_reward = 0
        self.EMA_rewards_list = []
        self.save_folder = save_folder

        self.solved = False

    def _collect_data(self, step, episode_reward):
        self.time_point_last = time.time()
        print("time spent: ", self.time_point_last - self.time_point_first)
        self.timelist.append(self.time_point_last - self.time_point_first)
        print("steps made: ", step)
        self.stepslist.append(step)
        print("reward:", episode_reward)
        self.rewardslist.append(episode_reward)
        self.EMA_reward = 0.5 * episode_reward + 0.95 * self.EMA_reward
        self.EMA_rewards_list.append(self.EMA_reward)
        self.time_point_first = self.time_point_last

    def save_report(self, name):
        with open(self.save_folder + name, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([len(self.rewardslist)])
            for val in self.rewardslist:
                writer.writerow([val])

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
        data_collector = DataCollector(save_folder)

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
                if (episode % self.TEST_ON_EPISODE) == 0 and episode > 100:
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
            data_collector.save_report(self.ENV_NAME)
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

    def _plotting(x, data_collector, env_name):
        today = datetime.date.today().strftime("%d-%m-%Y")

        try:
            os.makedirs("./experiments/" + env_name + "/Pictures")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        plt.figure(1)
        plt.plot(x, np.array(data_collector.stepslist), 'b')
        plt.savefig("./experiments/" + env_name + "/Pictures/steps" + today)
        plt.figure(2)
        plt.plot(x, np.array(data_collector.rewardslist), 'r')
        plt.savefig("./experiments/" + env_name + "/Pictures/rewards" + today)
        plt.figure(3)
        plt.plot(x, np.array(data_collector.EMA_rewards_list))
        plt.savefig("./experiments/" + env_name + "/Pictures/filtered_rewards" + today)
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
