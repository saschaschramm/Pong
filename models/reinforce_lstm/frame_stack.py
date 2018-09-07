import numpy as np

class FrameStack():

    def __init__(self, env, num_stack):
        self.env = env
        self.stacked_observations = np.zeros((80, 80, num_stack))

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        self.stacked_observations = np.roll(self.stacked_observations, shift=-1, axis=-1)
        self.stacked_observations[..., -observation.shape[-1]:] = observation
        return self.stacked_observations, reward, done

    def reset(self):
        observation = self.env.reset()
        self.stacked_observations[...] = 0
        self.stacked_observations[..., -observation.shape[-1]:] = observation
        return self.stacked_observations