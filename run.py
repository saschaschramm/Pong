from env_wrapper import init_env
from model import Model, PolicyFullyConnected
from runner import Runner
from common.utilities import global_seed

def run():
    global_seed(0)
    env = init_env()

    dir = "Reinforce"

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=(80, 80),
        action_space=2,
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
