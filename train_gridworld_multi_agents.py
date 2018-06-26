import numpy as np
from game import mygym as gym
from training.ddqn_gw import DQNAgent
# import cv2
import threading
import pygame
import time as time_pg

EPISODES = 5000

visual = True
verbose = False
env = gym.make(visual=visual, game='GridWorld')
state_size = env.observation_space_shape
action_size = len(env.action_space)

done = False
batch_size = 32
max_step = 500

num_agents = 3
agents = []
curr_reward = {}
total_rewards = {}
MY_LOCK = threading.Lock()


def play(agent, init_state, results, e, time):
    state = init_state

    action = agent.act(state)
    MY_LOCK.acquire()
    next_state, reward, done, total_reward = env.step(action, agent)
    MY_LOCK.release()
    if curr_reward[agent.Id] == 0:
        curr_reward[agent.Id] = reward
    backup = reward
    reward = reward - curr_reward[agent.Id]
    curr_reward[agent.Id] = backup
    total_rewards[agent.Id] += reward
    next_state = np.reshape(next_state, [1, state_size])
    # if visual:
    # 	time_pg.sleep(20)
    #     cv2.imshow('state', next_state)
    #     cv2.waitKey(10)
    if verbose:
        if reward <= -1:
            print('agent:{} --> step reward: {}, total reward: {:2}, e: {:.2}'.
                  format(agent.Id, reward, round(total_rewards[agent.Id], 3), agent.epsilon))
    elif time == max_step - 1:
        print('agent:{} --> total reward: {:2}, e: {:.2}'.
              format(agent.Id, round(total_rewards[agent.Id], 3), agent.epsilon))

    agent.remember(state, action, reward, next_state, done)
    if e % 20 == 0:
    # if done:
        agent.update_target_model()
        print("agent:{} --> episode: {}/{}, e: {:.2}"
              .format(agent.Id, e + 1, time, agent.epsilon))
        results[agent.Id] = next_state, True
    else:
        if e % 10 == 0:
            agent.update_target_model()
        results[agent.Id] = next_state, False


if __name__ == "__main__":
    for i in range(num_agents):
        if i == 0:
            ran = True
        else:
            ran = False
        a = DQNAgent(state_size, action_size, myId=i, random_planing=ran)
        a.load("training/save/gw-ddqn-multiple-agents-{}.h5".format(i))
        agents.append(a)
        curr_reward[i] = 0
        total_rewards[i] = 0
    pygame.event.pump()
    for e in range(EPISODES):
        print('Episode {}/{}'.format(e + 1, EPISODES))
        states = env.reset(num_agents=num_agents, num_targets=0)
        done_agents = []

        for time in range(max_step):
            pygame.event.pump()
            threads = {}
            results = {}
            for agent in agents:
                if agent.Id not in done_agents:
                    state = np.reshape(states[agent.Id], [1, state_size])
                    threads[agent.Id] = threading.Thread(target=play, args=(agent, state, results, e, time))
                    threads[agent.Id].start()
                    # play(agent, state, results)

            for t in threads.values():
                t.join()

            for agent in agents:
                if agent.Id in results:
                    next_state, imdone = results[agent.Id]
                    states[agent.Id] = next_state
                    if imdone:
                        done_agents.append(agent.Id)
            if len(done_agents) == len(agents):
                print('all done')
                break
        for agent in agents:
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if e % 10 == 0:
                agent.save("training/save/gw-ddqn-multiple-agents-{}.h5".format(agent.Id))
