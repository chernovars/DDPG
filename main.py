import filter_env
from ddpg import *
import gc
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

import os, errno
import sys
import csv
import xml.etree.ElementTree as ET
import shutil

gc.enable()

ENV_NAME = 'InvertedDoublePendulum-v1'
EPISODES = 100000000  # 100000
TIME_LIMIT = 0
TEST = 10
RENDER_STEP = False
RENDER_TEST_EPISODE = False
RENDER_delay = 0  # 0.1
TRAIN = True
NOISE = False
TEST_ON_EPISODE = 500
VIDEO_ON_EPISODE = 20000 #4*TEST_ON_EPISODE
SHOW_PLOT = False
RECORD_VIDEO = False

UNTIL_SOLVED = False
AVG_REWARD = 0
OVER_LAST = 0

ACTOR_SETTINGS = []
CRITIC_SETTINGS = []

def scenario(_scenario):
    cur_time = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
    save_folder = "./experiments/"+_scenario+"_"+cur_time
    os.makedirs(save_folder, exist_ok=True)
    _scenario = _scenario + ".xml"
    if not os.path.isfile("./" + _scenario):
        print("Create and fill file xml scenario file.")
    else:
        tree = ET.parse(_scenario)
        root = tree.getroot()
        i = 0
        for t in root:
            i = i+1
            t_folder = save_folder + "/Task" + str(i)
            os.makedirs(t_folder, exist_ok=True)
            task(t, t_folder)

def task(_task, save_folder):
    global ENV_NAME
    global EPISODES
    global TIME_LIMIT
    global TEST
    global RENDER_STEP
    global RENDER_delay
    global TRAIN
    global NOISE
    global TEST_ON_EPISODE
    global UNTIL_SOLVED
    global ACTOR_SETTINGS
    global CRITIC_SETTINGS
    global AVG_REWARD
    global OVER_LAST

    if _task.attrib["type"] == "simulation":
        os.makedirs(save_folder + "/saved_actor_networks", exist_ok=True)
        os.makedirs(save_folder + "/saved_critic_networks", exist_ok=True)
        ENV_NAME = _task[0].attrib["name"]
        TEST = 10
        TEST_ON_EPISODE = 500

        el_actor = _task[0][0]
        el_critic = _task[0][1]
        el_end_criteria = _task[0][2]
        ACTOR_SETTINGS = el_actor.attrib
        CRITIC_SETTINGS = el_critic.attrib

        end_criteria = el_end_criteria.attrib["criteria"]
        if end_criteria == "episodes":
            EPISODES = int(el_end_criteria.text)
        elif end_criteria == "time": # TODO: Implement
            TIME_LIMIT = int(el_end_criteria.text) * 60  # minutes * seconds
        elif end_criteria == "solved":
            EPISODES = 10000000000000
            UNTIL_SOLVED = True
            AVG_REWARD = int(el_end_criteria[0].text)
            OVER_LAST = int(el_end_criteria[1].text)

        RENDER_STEP = False
        RENDER_delay = 0
        TRAIN = True
        NOISE = True
        main(save_folder)



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
    plt.plot(x, np.array(data_collector.EMA_rewards_list))
    plt.savefig("./experiments/"+env_name+"/Pictures/filtered_rewards"+today)
    plt.show(block = False)

def _testing(env, agent, episode, data_collector, env_name):
    x = np.arange(1, episode + 2, 1)
    if SHOW_PLOT:
        _plotting(x, data_collector, env_name)
    if RENDER_TEST_EPISODE:
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
        if UNTIL_SOLVED:
            last_rewards = self.rewardslist[-OVER_LAST:]
            if (sum(last_rewards) / len(last_rewards)) > AVG_REWARD:
                self.solved = True

    def save_report(self, name):
        with open(self.save_folder + name, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([len(self.rewardslist)])
            for val in self.rewardslist:
                writer.writerow([val])

def main(save_folder):
    global UNTIL_SOLVED
    #myplot = plot.Plot() # TODO: real-time plotting for state analysys
    env_real = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env_real, TRAIN, NOISE, ENV_NAME, ACTOR_SETTINGS, CRITIC_SETTINGS)

    if RECORD_VIDEO:
        monitor = gym.wrappers.Monitor(env_real, 'experiments/' + ENV_NAME, (lambda i: (i % VIDEO_ON_EPISODE) == 0), resume=True)
        env = monitor
    else:
        env = env_real
    data_collector = DataCollector(save_folder)

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
                    if UNTIL_SOLVED and data_collector.solved:
                        UNTIL_SOLVED = False
                        data_collector.save_report(ENV_NAME)
                        _testing(env, agent, episode, data_collector, ENV_NAME)
                        agent.close()
                        return
                    break

            # Testing:
            if episode % TEST_ON_EPISODE == 0 and episode > 1:
                if agent.TRAIN:
                    agent.save(episode, save_folder)
                _testing(env, agent, episode, data_collector, ENV_NAME)
        agent.close()

    except KeyboardInterrupt:
        agent.save(episode, save_folder)
        if RECORD_VIDEO:
            monitor.close()

if __name__ == '__main__':
    #main()
    scenario("scenario1")