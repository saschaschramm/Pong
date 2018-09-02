from common.env_wrapper import init_environment
from reinforce_fc.model import Model, PolicyFullyConnected
from reinforce_fc.runner import Runner
from common.utilities import global_seed

def run():
    global_seed(0)
    env = init_environment()
    dir = "reinforce_fc"

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=2e-4
    )

    runner = Runner(
        env = env,
        model = model,
        timesteps=int(1e6),
        batch_size= 128,
        discount_rate=0.99,
        summary_frequency=20000,
        performance_num_episodes=10,
        summary_log_dir=dir
    )
    runner.run()

run()
