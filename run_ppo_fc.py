from common.env_wrapper import init_environment
from common.utilities import global_seed
from models.ppo.model import Model
from models.ppo.policies import PolicyFullyConnected
from models.ppo.runner import Runner
import numpy as np

def run():
    global_seed(0)
    total_timesteps = int(1000000)
    env = init_environment()
    dir = "ppo_fc"

    num_steps = 128
    ent_coef = 0.01
    learning_rate = 3e-4
    vf_coef = 0.5
    gae_gamma = 0.99
    gae_lambda = 0.95
    num_batches = 4
    num_training_epochs = 4
    clip_range = 0.2
    batch_size = num_steps // num_batches

    model = Model(policy=PolicyFullyConnected,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  batch_size=batch_size,
                  ent_coef=ent_coef,
                  vf_coef=vf_coef)

    runner = Runner(env=env,
                    model=model,
                    num_steps=num_steps,
                    advantage_estimator_gamma=gae_gamma,
                    advantage_estimator_lambda=gae_lambda,
                    summary_log_dir=dir)

    for _ in range(0, (total_timesteps // num_steps) + 1):
        assert num_steps % num_batches == 0
        observations, returns, masks, actions, values, log_probs = runner.run()
        indexes = np.arange(num_steps)  # [0,1,2 ..., 127]

        for _ in range(num_training_epochs):
            np.random.shuffle(indexes)

            for i in range(0, num_steps, batch_size):
                # 0
                # 32
                # 64
                # 96
                shuffled_indexes = indexes[i:i + batch_size]

                model.train(learning_rate=learning_rate,
                            clip_range=clip_range,
                            observations=observations[shuffled_indexes],
                            returns=returns[shuffled_indexes],
                            actions=actions[shuffled_indexes],
                            values=values[shuffled_indexes],
                            log_probs=log_probs[shuffled_indexes]
                            )

run()