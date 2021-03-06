import numpy as np
from common.stats_recorder import StatsRecorder
from gym.envs.classic_control.rendering import SimpleImageViewer
from common.env_wrapper import action_with_index
from common.utilities import discount


class Runner:

    def __init__(self,
                 env,
                 model,
                 batch_size,
                 timesteps,
                 discount_rate,
                 summary_frequency,
                 performance_num_episodes,
                 summary_log_dir):
        self.env = env
        self.model = model
        self.timesteps = timesteps
        self.discount_rate = discount_rate
        self.observation = env.reset()
        self.batch_size = batch_size
        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)
        self.viewer = SimpleImageViewer()

    def render(self):
        columns = []
        for i in range(80):
            rows = []
            for j in range(80):
                if self.observation[i][j] == 1:
                    rows.append([255, 255, 255])
                else:
                    rows.append([0, 0, 0])
            columns.append(rows)
        self.viewer.imshow(np.asarray(columns, dtype=np.uint8))

    def run(self):
        observations = []
        rewards = []
        actions = []
        terminals = []

        for t in range(self.timesteps+1):
            action_index = self.model.predict_action([self.observation])
            observations.append(self.observation)
            action = action_with_index(action_index)

            self.observation, reward, terminal = self.env.step(action)
            self.stats_recorder.after_step(reward=reward, terminal=terminal)

            rewards.append(reward)
            actions.append(action_index)
            terminals.append(terminal)

            if len(rewards) == self.batch_size:
                discounted_rewards = discount(rewards, terminals, self.discount_rate)

                self.model.train(observations, discounted_rewards, actions)
                observations = []
                rewards = []
                actions = []
                terminals = []

            if terminal:
                self.observation = self.env.reset()

            if t % self.stats_recorder.summary_frequency == 0:
                self.model.save(0)