from common.env_wrapper import init_env, action_with_index
from common.model import Model, PolicyFullyConnected
import time

from common.utilities import global_seed


def test():
    global_seed(0)

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=(80, 80),
        action_space=2,
        learning_rate=2e-4
    )

    model.load(0)
    env = init_env()
    observation = env.reset()

    for t in range(int(1e6)):
        action_index = model.predict_action([observation])[0]
        action = action_with_index(action_index)
        observation, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)

        if done:
            observation = env.reset()

test()