from common.env_wrapper import init_environment
from models.reinforce_fc.model import Model, PolicyFullyConnected
from models.reinforce_fc.runner import Runner
from common.utilities import global_seed
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', default=2e-4)
    parser.add_argument('--timesteps', default=int(1e6))
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--discount_rate', default=0.99)
    parser.add_argument('--summary_frequency', default=20000)
    parser.add_argument('--performance_num_episodes', default=10)
    parser.add_argument('--summary_log_dir', default="reinforce_fc")
    args = parser.parse_args()
    global_seed(0)
    env = init_environment()

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=args.learning_rate
    )

    runner = Runner(
        env=env,
        model=model,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        discount_rate=args.discount_rate,
        summary_frequency=args.summary_frequency,
        performance_num_episodes=args.performance_num_episodes,
        summary_log_dir=args.summary_log_dir
    )
    runner.run()


if __name__ == '__main__':
    main()
