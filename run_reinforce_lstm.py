from reinforce_lstm.policies import LstmPolicy
from common.env_wrapper import init_environment
from common.utilities import global_seed
from reinforce_lstm.model import Model
from reinforce_lstm.runner import Runner

def run():
    global_seed(0)
    env = init_environment()

    discount_rate = 0.99
    batch_size = 128

    dir = "reinforce_lstm"

    model = Model(policy=LstmPolicy,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  batch_size=batch_size
                  )

    runner = Runner(env=env,
                    model=model,
                    batch_size=batch_size,
                    discount_rate=discount_rate,
                    summary_frequency=20000,
                    performance_num_episodes=10,
                    summary_log_dir=dir)

    timesteps = int(1e6)

    for _ in range(0, timesteps//batch_size+1):
        observations, states, rewards, dones, actions = runner.run()
        model.train(observations, states, rewards, dones, actions)

run()