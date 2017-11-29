import filter_env
from ddpg import *
import gc
import time

gc.enable()

ENV_NAME = 'InvertedPendulum-v1'
EPISODES = 1000#100000
TEST = 10

def renderEpisode(env, agent, state):
    state = env.reset()
    # print "episode:",episode
    # Train
    for step in range(env.spec.timestep_limit):
        env.render()
        time.sleep(0.2)
        action = agent.noise_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.perceive(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

def main():
    #env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    env = gym.make(ENV_NAME)
    agent = DDPG(env)
    monitor = gym.wrappers.Monitor(env, 'experiments/' + ENV_NAME, force=True)
    #env.monitor.start(, force=True)

    for episode in range(EPISODES):
        state = env.reset()
        # print "episode:",episode
        # Train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
    renderEpisode(env, agent, state)
    monitor.close()


if __name__ == '__main__':
    main()
