import tensorflow as tf


class PolicyFullyConnected:
    def __init__(self, observation_space, action_space, batch_size, reuse):
        height = observation_space[0]
        width = observation_space[1]
        self.observations = tf.placeholder(shape=(batch_size, height, width), dtype=tf.float32)

        with tf.variable_scope(name_or_scope="model", reuse=reuse):
            reshaped_observations = tf.reshape(tensor=tf.to_float(self.observations),
                                               shape=(batch_size, height * width))

            self.hidden = tf.layers.dense(inputs=reshaped_observations,
                                          units=256,
                                          activation=tf.nn.relu)

            logits = tf.layers.dense(inputs=self.hidden, units=action_space)

            self.probs = tf.nn.softmax(logits)
            self.values = tf.layers.dense(inputs=self.hidden, units=1)[:, 0]