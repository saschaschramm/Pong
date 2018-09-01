from reinforce_lstm.policies import LstmPolicy

#from reinforce_lstm.policies2 import LstmPolicy
from reinforce_lstm.frame_stack import FrameStack
from common.env_wrapper import init_env
from common.utilities import global_seed
from reinforce_lstm.model import Model
from reinforce_lstm.runner import Runner

def run():
    global_seed(0)
    env = init_env()
    #env = FrameStack(env, 4)

    discount_rate = 0.99
    observation_space = (80, 80)
    action_space = 2
    batch_size = 128

    dir = "reinforce_lstm"

    model = Model(policy=LstmPolicy,
                  observation_space=observation_space,
                  action_space=action_space,
                  batch_size=batch_size
                  )

    runner = Runner(env=env,
                    model=model,
                    batch_size=batch_size,
                    discount_rate=discount_rate,
                    summary_frequency=40000,
                    performance_num_episodes=10,
                    summary_log_dir=dir)

    timesteps = int(1e6)

    for _ in range(0, timesteps//batch_size+1):
        observations, states, rewards, dones, actions = runner.run()
        model.train(observations, states, rewards, dones, actions)

run()

"""
1000 0.0 14.2
2000 -21.0 7.2
3000 -21.0 7.0
4000 -21.0 6.9
5000 -20.0 7.9
6000 -20.0 7.9
7000 -21.0 8.3
8000 -19.0 7.5
9000 -20.0 7.1
10000 -18.0 7.4
11000 -19.0 7.2
12000 -21.0 6.9
13000 -20.0 6.7
"""