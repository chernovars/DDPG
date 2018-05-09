import filter_env
from ddpg import DDPG
import roboschool
import gym
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from OpenGL import GLU
from math import isnan


from cv2 import resize, INTER_CUBIC
import os, errno
import csv

from utils import generatePlot


class DataCollector:
    def __init__(self, save_folder, env_name):
        self.time_point_first = time.time()
        self.time_point_last = time.time()
        self.time_list = []
        self.steps_list = []
        self.rewards_list = []
        self.test_list_y = []
        self.test_list_x = []
        self.start_test_x = 0

        self.EMA_reward = 0
        self.EMA_rewards_list = []
        self.save_folder = save_folder
        self.env_name = env_name
        if os.path.exists(self.save_folder + self.env_name):
            self.rewards_list = self.load_rewards_list(self.save_folder + self.env_name)
            self.EMA_rewards_list = self.listToEMA(self.rewards_list)
        if os.path.exists(self.save_folder + self.env_name + "_test"):
            self.load_test_list(self.save_folder + self.env_name + "_test")

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

    def save_test_list(self):
        with open(self.save_folder + self.env_name + "_test", "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([len(self.test_list_x)])
            for x, y in zip(self.test_list_x, self.test_list_y):
                writer.writerow([x, y])

    def load_rewards_list(self, file):
        if not os.path.isfile(file):
            print("def load_rewards_list: no path specified")
            exit(1)

        with open(file, "r") as f:
            n = f.readline()
            self.start_test_x = int(n)
            x = f.readlines()
            x = [float(i.strip()) for i in x]
            self.rewards_list = x

        return self.rewards_list

    def load_test_list(self, file):
        with open(file, "r") as f:
            n = f.readline()
            x = f.readlines()
            for line in x:
                l = line.split(',')
                self.test_list_x.append(float(l[0]))
                self.test_list_y.append(float(l[1]))
        return [self.test_list_x, self.test_list_y]

    def listToEMA(self, orig, tau=0.98):
        EMA = []
        EMA.append(orig[0] * (1 - tau))
        for i in range(1, len(orig)):
            if isnan(orig[i]):
                print("nan on ", i)
                raise EnvironmentError("Mujoco unstable")
            EMA.append(EMA[i - 1] * tau + orig[i] * (1 - tau))

        test =EMA[-50:]
        test2 = orig[-50:]
        return EMA


class World:

    def __init__(self, RENDER_STEP=True, RENDER_delay=0, TRAIN=True, NOISE=True):
        self.ENV_NAME = 'InvertedDoublePendulum-v1'
        self.EPISODES = 100000000  # 100000
        self.TIME_LIMIT = 0
        self.TEST_NUM = 10
        self.TEST = False
        self.TEST_SAVE = False

        self.RENDER_STEP = RENDER_STEP
        self.RENDER_TEST_EPISODE = False
        self.RENDER_delay = RENDER_delay  # 0.1
        self.TRAIN = TRAIN
        self.NOISE = NOISE
        self.NOISE_PERIOD = 100000000
        self.TEST_ON_EPISODE = 50 ################################# TODO: FIX IT !!!!
        self.VIDEO_ON_EPISODE =200000  # 4*TEST_ON_EPISODE
        self.SHOW_PLOT = False
        self.RECORD_VIDEO = False
        self.SAVE = True
        self.UNTIL_SOLVED = False
        self.AVG_REWARD = 0
        self.OVER_LAST = 0
        self.OBSERVATIONS = "state"
        self.ACTION_REPEATS = 2

        self.ACTOR_SETTINGS = []
        self.CRITIC_SETTINGS = []


    def main(self, save_folder, change_saved=True):
        # myplot = plot.Plot() # TODO: real-time plotting for state analysys
        observation = []
        start_time = 0
        if self.TIME_LIMIT > 0:
            start_time = time.time()

        env_real = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))

        agent = DDPG(env_real, self.TRAIN, self.NOISE, self.ENV_NAME, self.ACTOR_SETTINGS, self.CRITIC_SETTINGS,
                 save_folder, observations=self.OBSERVATIONS)
        self.agent = agent

        if self.RECORD_VIDEO:
            monitor = gym.wrappers.Monitor(env_real, save_folder,
                                           (lambda i: (i % self.VIDEO_ON_EPISODE) == 0), resume=True)
            env = monitor
        else:
            env = env_real
        data_collector = DataCollector(save_folder, self.ENV_NAME)

        episode = None

        try:
            for episode in range(self.EPISODES):
                state = env.reset()
                if self.OBSERVATIONS == "pixels":
                    #pic_batch = np.zeros(shape=env.render('rgb_array').shape[:-1])
                    self.pic_batch = np.zeros((64,64,2))
                    pic = resize(np.mean(env.render('rgb_array'), -1), dsize=(64, 64), interpolation=INTER_CUBIC)
                    for i in range(self.ACTION_REPEATS):
                        self.pic_batch[:,:,i] = pic
                    state = self.pic_batch
                print("episode:", episode)
                steps = 0
                episode_reward = 0
                # Simulate + train
                for step in range(env.spec.timestep_limit):
                    if self.RENDER_STEP:
                        self.slow_render(env)
                        if self.RENDER_delay > 0:
                            time.sleep(self.RENDER_delay)

                    if self.OBSERVATIONS == "pixels":
                        action, next_state, reward, done = self.get_next_pixel_state(agent, env)
                    else:
                        if self.NOISE:
                            action = agent.noise_action(state)
                        else:
                            action = agent.action(state)
                        next_state, reward, done, _ = env.step(action)
                        #observation.append(next_state[1])
                        #x = list(range(len(observation)))
                        #plt.plot(x, observation)
                        #plt.show(block=False)



                    episode_reward += reward
                    agent.perceive(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1
                    if done:
                        data_collector._collect_data(steps, episode_reward)
                        break
                # Testing:
                if self.TEST and (episode % self.TEST_ON_EPISODE) == 0 and episode > 0:
                        # agent.save(episode, save_folder)
                        if self._testing(env, agent, episode, data_collector, self.ENV_NAME):
                            self.finish(agent, env, episode, save_folder, data_collector, change_saved)
                            return

                if (episode % self.NOISE_PERIOD) == 0 and episode > 0:
                    self.NOISE = not self.NOISE

                if self.TIME_LIMIT > 0 and (time.time() - start_time) > self.TIME_LIMIT:
                    self.finish(agent, env, episode, save_folder, data_collector, change_saved)
                    return
            self.finish(agent, env, self.EPISODES, save_folder, data_collector, change_saved)
            return
        except KeyboardInterrupt:
            self.finish(agent, env, episode, save_folder, data_collector, change_saved)
            return

    def finish(self, agent=None, env=None, episode=None, save_folder=None, data_collector=None, change_saved=True):
        if not agent:
            agent = self.agent
        if change_saved:
            if self.SAVE:
                agent.save(episode, save_folder)
            else:
                agent.replay_buffer.erase()
                agent.replay_buffer.save_buffer(save_folder)
            data_collector.save_rewards_list()
            data_collector.save_test_list()
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
        generatePlot(np.array(data_collector.stepslist), x=x, title=today, labels=["episodes", "steps"],
                     save_folder=path + "steps" + today, show_picture=True)
        generatePlot(np.array(data_collector.rewardslist), x=x, title="", labels=["episodes", "steps"],
                     save_folder=path + "rewards" + today,
                     show_picture=True, color='r')
        generatePlot(np.array(data_collector.EMA_rewards_list), x=x, title="", labels=["episodes", "steps"],
                     save_folder=path + "filtered_rewards" + today,
                     show_picture=True, color='r')
        plt.show(block=False)

    def _testing(self, env, agent, episode, data_collector, env_name):
        save_dir, save_file = os.path.split(data_collector.save_folder)

        def check_append(text):
            if self.TEST_SAVE:
                with open(save_dir + "/TEST.csv", "a") as myfile:
                    myfile.write(text)

        x = np.arange(1, episode + 2, 1)
        if self.SHOW_PLOT:
            self._plotting(x, data_collector, env_name)
        if self.RENDER_TEST_EPISODE:
            self.renderEpisode(env, agent)
        total_reward = 0


        check_append(save_file + "  ")
        for i in range(self.TEST_NUM):
            ep_reward = 0
            state = env.reset()
            for j in range(env.spec.timestep_limit):
                if self.OBSERVATIONS == "pixels":
                    _, state, reward, done  = self.get_next_pixel_state(agent, env, noise=False)
                else:
                    action = agent.action(state, is_testing=True)  # direct action for test
                    state, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    check_append(str(ep_reward) + " ")
                    total_reward += ep_reward
                    break
        check_append("\n")


        ave_reward = total_reward / self.TEST_NUM
        data_collector.test_list_x.append(episode + data_collector.start_test_x)
        data_collector.test_list_y.append(ave_reward)
        print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        if self.UNTIL_SOLVED and ave_reward > self.AVG_REWARD:
            total_reward = 0
            for i in range(self.OVER_LAST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = agent.action(state, is_testing=True)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / self.OVER_LAST
            print('episode: ', episode, 'Evaluation Average Reward 2:', ave_reward)
            if ave_reward > self.AVG_REWARD:
                return True
        return False

    def get_next_pixel_state(self, agent, env, noise=None):
        if noise is None:
            noise = self.NOISE
        if noise:
            action = agent.noise_action(self.pic_batch)
        else:
            action = agent.action(self.pic_batch)
        for i in range(self.ACTION_REPEATS):
            _, reward, done, _ = env.step(action)
            self.pic_batch[:, :, i] = resize(np.mean(env.render('rgb_array'), -1), dsize=(64, 64), interpolation=INTER_CUBIC)
        return action, self.pic_batch, reward, done




'''if __name__ == '__main__':
    w = World
    w.main()
'''