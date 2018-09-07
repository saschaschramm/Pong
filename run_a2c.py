from common.env_wrapper import init_environment
from models.a2c.model import Model, PolicyFullyConnected
from models.a2c.runner import Runner
from common.utilities import global_seed

def run():
    global_seed(0)
    env = init_environment()

    dir = "a2c"

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
        num_steps= 5,
        discount_rate=0.99,
        summary_frequency=20000,
        performance_num_episodes=10,
        summary_log_dir=dir
    )
    runner.run()

run()