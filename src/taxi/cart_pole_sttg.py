import gym
from gym import spaces
def main():
    # env_name = 'MsPacman-v0'
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    for i_episode in range(20):
        observation = env.reset()
        angle = observation[2]
        for t in range(1000):
            if angle > 0:
                action = 1
            else:
                action = 0
            env.render()
            # print("observation:",
            #       observation)
            # action = env.action_space.sample()
            observation, reward, done, info = \
                env.step(action)
            angle = observation[2]
            # print(observation, reward, done, info)
            if done:
                print("Episode finished after {} timestamps".format(t+1))
                break


if __name__ == "__main__":
    main()
