import gym
import numpy as np
from collections import deque

def action_with_index(index):
    if index == 0:
        return 2
    elif index == 1:
        return 3
    else:
        NotImplementedError

def init_env():
    env = gym.make('PongNoFrameskip-v4')
    env.seed(0)
    env = MaxMergeSkipEnv(env, skip=4)
    env = FireResetEnv(env)
    env = BlackWhiteEnv(env)
    return env

class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GymWrapper, self).__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

class MaxMergeSkipEnv(GymWrapper):
    def __init__(self, env, skip=4):
        GymWrapper.__init__(self, env)
        self._observation_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._skip):
            observation, reward, done, info = self.env.step(action)
            self._observation_buffer.append(observation)
            total_reward += reward
            combined_info.update(info)
            if done:
                break

        max_frame = np.max(self._observation_buffer, axis=0)
        return max_frame, total_reward, done, combined_info

    def reset(self):
        self._observation_buffer.clear()
        observation = self.env.reset()
        self._observation_buffer.append(observation)
        return observation

class FireResetEnv(GymWrapper):
    def __init__(self, env):
        GymWrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        observation, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        observation, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return observation

class BlackWhiteEnv(GymWrapper):
    def __init__(self, env):
        GymWrapper.__init__(self, env)

    def preprocess(self, image):
        image = image[35:195]
        image = image[::2, ::2, 0]
        player = (image == 92) * 1
        ball = (image == 236) * 1
        enemy = (image == 213) * 1
        return player + ball + enemy

    def reset(self):
        return self.preprocess(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.preprocess(observation), reward, done, info