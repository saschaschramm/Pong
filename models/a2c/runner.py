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

class Runner():

    def __init__(self,
                 env,
                 model,
                 num_steps,
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
        batch_observations = []
        batch_rewards = []
        batch_actions = []
        batch_dones = []
        batch_values = []

        print("self.observations", self.observation.shape)

        for t in range(self.timesteps+1):
            action_index, value = self.model.predict([self.observation])
            batch_observations.append(self.observation)
            action = action_with_index(action_index)
            batch_values.append(value)

            self.observation, reward, terminal = self.env.step(action)
            self.stats_recorder.after_step(reward=reward, done=terminal, t=t)

            batch_rewards.append(reward)
            batch_actions.append(action_index)
            batch_dones.append(terminal)

            if len(batch_rewards) == self.num_steps:

                if batch_dones[-1] == 0:
                    next_value = self.model.predict_value([self.observation])[0]
                    discounted_rewards = discount(batch_rewards + [next_value], batch_dones + [False],
                                                  self.discount_rate)[:-1]
                else:
                    discounted_rewards = discount(batch_rewards, batch_dones, self.discount_rate)

                self.model.train(batch_observations, discounted_rewards, batch_actions, batch_values)
                batch_observations = []
                batch_rewards = []
                batch_actions = []
                batch_dones = []
                batch_values = []

            if terminal:
                self.observation = self.env.reset()

            if t % self.stats_recorder.summary_frequency == 0:
                self.model.save(0)