from common.stats_recorder import StatsRecorder
from common.utilities import discount
from common.env_wrapper import action_with_index


class Runner:

    def __init__(self,
                 env,
                 model,
                 batch_size,
                 discount_rate,
                 summary_frequency,
                 performance_num_episodes,
                 summary_log_dir
                 ):
        self.env = env
        self.model = model
        self.observation = env.reset()
        self.batch_size = batch_size
        self.states = model.initial_state
        self.terminal = False
        self.discount_rate = discount_rate

        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)

    def run(self):
        observations, batch_rewards, actions, terminals = [],[],[],[]
        states = self.states

        for n in range(self.batch_size):
            action_index, self.states = self.model.predict_action([self.observation], self.states, [self.terminal])
            action = action_with_index(action_index)
            observations.append(self.observation)
            actions.append(action_index)
            terminals.append(self.terminal)
            self.observation, reward, self.terminal = self.env.step(action)
            self.stats_recorder.after_step(reward=reward, terminal=self.terminal)

            if self.terminal:
                self.observation = self.env.reset()

            batch_rewards.append(reward)
        terminals.append(self.terminal)

        discounted_rewards = discount(batch_rewards, terminals[1:], self.discount_rate)
        return observations, states, discounted_rewards, terminals[:-1], actions