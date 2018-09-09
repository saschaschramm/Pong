from models.reinforce_lstm.policies import LstmPolicy
from common.env_wrapper import init_environment
from common.utilities import global_seed
from models.reinforce_lstm.model import Model
from models.reinforce_lstm.runner import Runner
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--timesteps', default=int(1e6))
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--discount_rate', default=0.99)
    parser.add_argument('--summary_frequency', default=20000)
    parser.add_argument('--performance_num_episodes', default=10)
    parser.add_argument('--summary_log_dir', default="reinforce_lstm")
    args = parser.parse_args()
    global_seed(0)
    env = init_environment()

    model = Model(policy=LstmPolicy,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  batch_size=args.batch_size
                  )

    runner = Runner(env=env,
                    model=model,
                    batch_size=args.batch_size,
                    discount_rate=args.discount_rate,
                    summary_frequency=args.summary_frequency,
                    performance_num_episodes=args.performance_num_episodes,
                    summary_log_dir=args.summary_log_dir)

    for _ in range(0, args.timesteps//args.batch_size+1):
        observations, states, rewards, terminals, actions = runner.run()
        model.train(observations, states, rewards, terminals, actions)


if __name__ == '__main__':
    main()