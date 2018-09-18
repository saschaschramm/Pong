import numpy as np
from common.stats_recorder import StatsRecorder
from common.env_wrapper import action_with_index


class Runner:

    def __init__(self,
                 env,
                 model,
                 num_steps,
                 advantage_estimator_gamma,
                 advantage_estimator_lambda,
                 summary_frequency,
                 performance_num_episodes,
                 summary_log_dir
                 ):
        self.gae_lambda = advantage_estimator_lambda
        self.gae_gamma = advantage_estimator_gamma

        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)
        self.env = env
        self.model = model
        self.observation = env.reset()
        self.num_steps = num_steps
        self.terminal = False
        self.rewards = []
        self.values = []
        self.advantage_estimation = 0

    def estimate_advantage(self, t, terminal, next_value):
        if terminal:
            delta = self.rewards[t] + self.values[t]
            return delta
        else:
            delta = self.rewards[t] + self.gae_gamma * next_value - self.values[t]
            return delta + self.gae_gamma * self.gae_lambda * self.advantage_estimation

    def run(self):
        observations, actions, terminals, log_probs = [], [], [], []
        self.rewards = []
        self.values = []

        for _ in range(self.num_steps):
            action_index, value, log_prob = self.model.predict(self.observation)
            observations.append(self.observation)
            actions.append(action_index)
            self.values.append(value)
            log_probs.append(log_prob)
            terminals.append(self.terminal)

            action = action_with_index(action_index)
            self.observation, reward, self.terminal = self.env.step(action)

            if self.terminal:
                self.observation = self.env.reset()

            self.stats_recorder.after_step(reward, self.terminal)
            self.rewards.append(reward)

        actions = np.asarray(actions)
        self.values = np.asarray(self.values)
        log_probs = np.asarray(log_probs)
        last_value = self.model.predict_value(self.observation)

        advantage_estimations = np.zeros_like(self.rewards)
        self.advantage_estimation = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                self.advantage_estimation = self.estimate_advantage(t, self.terminal, last_value)
            else:
                self.advantage_estimation = self.estimate_advantage(t, terminals[t+1], self.values[t+1])

            advantage_estimations[t] = self.advantage_estimation

        return np.asarray(observations), advantage_estimations, terminals, actions, self.values, log_probs
