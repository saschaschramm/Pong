import numpy as np
from common.stats_recorder import StatsRecorder
from gym.envs.classic_control.rendering import SimpleImageViewer
from common.env_wrapper import action_with_index


def discount(rewards, dones, discount_rate):
    discounted = []
    total_return = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            total_return = reward
        else:
            total_return = reward + discount_rate * total_return
        discounted.append(total_return)
    return np.asarray(discounted[::-1])


class Runner:

    def __init__(self,
                 env,
                 model,
                 num_steps,
                 discount_rate,
                 summary_frequency,
                 performance_num_episodes,
                 summary_log_dir):
        self.env = env
        self.model = model
        self.discount_rate = discount_rate
        self.observation = env.reset()
        self.num_steps = num_steps
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
        values = []

        for _ in range(self.num_steps):
            action_index, value = self.model.predict([self.observation])
            observations.append(self.observation)
            action = action_with_index(action_index)
            values.append(value)

            self.observation, reward, terminal = self.env.step(action)
            self.stats_recorder.after_step(reward=reward, terminal=terminal)

            rewards.append(reward)
            actions.append(action_index)
            terminals.append(terminal)

            if terminal:
                self.observation = self.env.reset()

        if terminals[-1] == 0:
            next_value = self.model.predict_value([self.observation])[0]
            discounted_rewards = discount(rewards + [next_value], terminals + [False],
                                          self.discount_rate)[:-1]
        else:
            discounted_rewards = discount(rewards, terminals, self.discount_rate)

        self.model.train(observations, discounted_rewards, actions, values)
