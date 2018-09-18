from common.env_wrapper import init_environment
from common.utilities import global_seed
from models.ppo.model import Model
from models.ppo.policies import PolicyFullyConnected
from models.ppo.runner import Runner
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--total_timesteps', default=int(1000000))
    parser.add_argument('--num_steps', default=128)
    parser.add_argument('--ent_coef', default=0.01)
    parser.add_argument('--learning_rate', default=3e-4)
    parser.add_argument('--vf_coef', default=0.5)
    parser.add_argument('--gae_gamma', default=0.99)
    parser.add_argument('--gae_lambda', default=0.95)
    parser.add_argument('--num_batches', default=4)
    parser.add_argument('--num_training_epochs', default=4)
    parser.add_argument('--clip_range', default=0.2)
    parser.add_argument('--summary_frequency', default=20000)
    parser.add_argument('--performance_num_episodes', default=10)
    parser.add_argument('--summary_log_dir', default="ppo_fc")
    args = parser.parse_args()


    global_seed(0)
    env = init_environment()
    batch_size = args.num_steps // args.num_batches

    model = Model(policy=PolicyFullyConnected,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  batch_size=batch_size,
                  ent_coef=args.ent_coef,
                  vf_coef=args.vf_coef)

    runner = Runner(env=env,
                    model=model,
                    num_steps=args.num_steps,
                    advantage_estimator_gamma=args.gae_gamma,
                    advantage_estimator_lambda=args.gae_lambda,
                    summary_frequency=args.summary_frequency,
                    performance_num_episodes=args.performance_num_episodes,
                    summary_log_dir=args.summary_log_dir)

    for _ in range(0, (args.total_timesteps // args.num_steps) + 1):
        assert args.num_steps % args.num_batches == 0
        observations, advantages, masks, actions, values, log_probs = runner.run()
        indexes = np.arange(args.num_steps)  # [0,1,2 ..., 127]

        for _ in range(args.num_training_epochs):
            np.random.shuffle(indexes)

            for i in range(0, args.num_steps, batch_size):
                # 0
                # 32
                # 64
                # 96
                shuffled_indexes = indexes[i:i + batch_size]

                model.train(learning_rate=args.learning_rate,
                            clip_range=args.clip_range,
                            observations=observations[shuffled_indexes],
                            advantages=advantages[shuffled_indexes],
                            actions=actions[shuffled_indexes],
                            values=values[shuffled_indexes],
                            log_probs=log_probs[shuffled_indexes]
                            )


if __name__ == '__main__':
    main()
