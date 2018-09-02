import numpy as np
import gym

def action_with_index(index):
    if index == 0:
        return 2
    elif index == 1:
        return 3
    else:
        NotImplementedError

def init_environment():
    environment = gym.make('PongNoFrameskip-v4')
    environment.seed(0)
    environment = EnvWrapper(environment, frame_skip=4)
    return environment

class EnvWrapper:

  def __init__(self, environment, frame_skip):
    self.environment = environment
    self.frame_skip = frame_skip
    self._observation_buffer = [
        np.empty((80, 80), dtype=np.uint8),
        np.empty((80, 80), dtype=np.uint8)
    ]

  @property
  def observation_space(self):
    return (80, 80)

  @property
  def action_space(self):
    return 2

  def reset(self):
        self.environment.reset()
        observation, _, terminal, _ = self.environment.step(1)
        if terminal:
            self.environment.reset()
        observation, _, terminal, _ = self.environment.step(2)
        if terminal:
            self.environment.reset()

        self._observation_buffer[0] = self.preprocess(observation)
        self._observation_buffer[1].fill(0)

        return self._max_merge()

  def step(self, action):
    total_reward = 0.0

    for time_step in range(self.frame_skip):
      observation, reward, terminal, _ = self.environment.step(action)
      total_reward += reward

      if terminal:
        break
      elif time_step >= self.frame_skip - 2:
        # 0 < 2
        # 1 < 2
        # 2 >= 2 -> observation_buffer[0]
        # 3 >= 2 -> observation_buffer[1]
        t = time_step - (self.frame_skip - 2)
        self._observation_buffer[t] = self.preprocess(observation)

    observation = self._max_merge()
    return observation, total_reward, terminal

  def preprocess(self, image):
    image = image[35:195]
    image = image[::2, ::2, 0]
    player = (image == 92) * 1
    ball = (image == 236) * 1
    enemy = (image == 213) * 1
    return (player + ball + enemy).reshape(80, 80)

  def _max_merge(self):
    if self.frame_skip > 1:
      max_frame = np.max(self._observation_buffer, axis=0)
    return max_frame