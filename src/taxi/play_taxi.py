import random
import gym
from gym import spaces
import numpy as np


def predict_reward(
        env,
        observation,
        a
):
    [row, col, passidx, destidx] = \
        env.decode(observation)
    nR = 5
    nC = 5
    maxR = nR-1
    maxC = nC-1
    taxiloc = (row, col)
    newrow, newcol, newpassidx = row, col, passidx
    reward = -1
    if a==0:
        newrow = min(row+1, maxR)
    elif a==1:
        newrow = max(row-1, 0)
    if a==2 and env.desc[1+row,2*col+2]==b":":
        newcol = min(col+1, maxC)
    elif a==3 and env.desc[1+row,2*col]==b":":
        newcol = max(col-1, 0)
    elif a==4: # pickup
        if (passidx < 4 and taxiloc == env.locs[passidx]):
            newpassidx = 4
        else:
            reward = -10
    elif a==5: # dropoff
        if (taxiloc == env.locs[destidx]) and passidx==4:
            newpassidx = destidx
            done = True
            reward = 20
        elif (taxiloc in env.locs) and passidx==4:
            newpassidx = env.locs.index(taxiloc)
        else:
            reward = -10
    newstate = env.encode(newrow, newcol, newpassidx, destidx)
    return reward, newstate


def max_a_for_q(nA, env, state_t, q_table):
    """

    :param nA:
    :param env:
    :param state_t:
    :param q_table:
    :return: max_q, selected_a
    """
    max_qp1 = -100
    selected_a = 0
    for a in range(nA):
        _, state_tp1 = predict_reward(
            env.env, state_t, a
        )
        q = q_table[state_tp1, a]
        if q > max_qp1:
            max_qp1 = q
            selected_a = a

    return max_qp1, selected_a

def choose_action(
        env,
        q_table:np.ndarray,
        state,
        gamma,
        t,
        i_episode,
        episode_size
):
    nA = 6
    thre = gamma ** (
            t + i_episode * episode_size)
    if random.random() < thre:  # beginning of the game
        # random action
        return env.action_space.sample()
    else:
        return max_a_for_q(nA, env, state, q_table)[1]


def update_q_table(
        env,
        observation,
        q_table:np.ndarray,
        lr,
        gamma
):
    """
    choose one action, and update q_table
    should add possibility to jump out
    :param action_dim:
    :param status_dim:
    :param q_table:
    :return:
    """

    nA = 6
    # look for Q table
    for action in range(nA):
        reward_t, state_t = predict_reward(
            env.env, observation, action
        )
        # find max reward
        max_qp1, selected_a = max_a_for_q(nA, env, state_t, q_table)

        q_table[state_t, action] = (1-lr) * q_table[state_t, action] +\
            lr * (reward_t + gamma * max_qp1)
    # reward_list = []
    # for a in range(nA):
    #     reward = predict_reward(
    #         env, row, col,
    #         passidx, destidx, a
    #     )
    #
    #     reward_list.append((a, reward))
    # [selected_action, max_reward] = sorted(reward_list, key=lambda x: x[1], reverse=True)[0]


def train():
    # env_name = 'MsPacman-v0'
    # env_name = 'CartPole-v1'
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    print(env.observation_space)
    print(env.action_space)
    status_dim = env.observation_space.n
    action_dim = env.action_space.n

    # init an [action x status] Q-table
    q_table = np.zeros(shape=(status_dim, action_dim))
    lr = 0.1
    gamma = 0.8

    # print(env.observation_space.high)
    # print(env.observation_space.low)
    for i_episode in range(3000):
        observation = env.reset()
        episode_size = 200
        for t in range(episode_size):
            env.render(mode='ansi')
            # env.render()
            # print("observation:",
            #       observation)
            # choose action
            update_q_table(env, observation, q_table,
                           lr, gamma)
            action = choose_action(
                env,
                q_table,
                observation,
                gamma,
                t,
                i_episode,
                episode_size
            )
            # action = env.action_space.sample()
            # print("taking action [{}]".format(action))
            observation, reward, done, info = \
                env.step(action)
            # print("info:", info)
            # print("getting reward [{}]".format(reward))
            # print(observation, reward, done, info)
            if done:
                print("Episode[{}] finished after {} timestamps".format(i_episode, t+1))
                break
    np.save("./q_table", q_table)


def test():
    # env_name = 'MsPacman-v0'
    # env_name = 'CartPole-v1'
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    print(env.observation_space)
    print(env.action_space)
    status_dim = env.observation_space.n
    action_dim = env.action_space.n

    # init an [action x status] Q-table
    q_table = np.load("./q_table.npy")
    # gamma = 1
    gamma = 0.8

    # print(env.observation_space.high)
    # print(env.observation_space.low)
    observation = env.reset()
    episode_size = 200
    i_episode = 3000
    for t in range(episode_size):
        env.render(mode='human')
        # env.render()
        # print("observation:",
        #       observation)
        # choose action
        action = choose_action(
            env,
            q_table,
            observation,
            gamma,
            t,
            i_episode,
            episode_size
        )
        # action = env.action_space.sample()
        # print("taking action [{}]".format(action))
        observation, reward, done, info = \
            env.step(action)
        # print("info:", info)
        # print("getting reward [{}]".format(reward))
        # print(observation, reward, done, info)
        if done:
            print("Episode[{}] finished after {} timestamps".format(i_episode, t+1))
            break


def main():
    test()
    # train()

if __name__ == "__main__":
    main()
