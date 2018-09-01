from common.stats_recorder import StatsRecorder
from common.utilities import discount
from common.env_wrapper import action_with_index

class Runner():

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
        self.done = False
        self.discount_rate = discount_rate

        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)
        self.t = 0

    def run(self):
        batch_observations, batch_rewards, batch_actions, batch_dones = [],[],[],[]
        mb_states = self.states
        for n in range(self.batch_size):
            action_index, states = self.model.predict_action([self.observation], self.states, [self.done])
            action = action_with_index(action_index)

            batch_observations.append(self.observation)
            batch_actions.append(action_index)
            batch_dones.append(self.done)
            self.observation, reward, self.done = self.env.step(action)

            self.stats_recorder.after_step(reward=reward,
                                           done=self.done,
                                           t=self.t)
            self.t += 1

            self.states = states

            if self.done:
                self.observation = self.env.reset()

            batch_rewards.append(reward)
        batch_dones.append(self.done)

        discounted_rewards = discount(batch_rewards, batch_dones[1:], self.discount_rate)
        return batch_observations, mb_states, discounted_rewards, batch_dones[:-1], batch_actions