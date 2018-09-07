import tensorflow as tf

def sample(probs):
    random_uniform = tf.random_uniform(tf.shape(probs))
    scaled_random_uniform = tf.log(random_uniform) / probs
    return tf.argmax(scaled_random_uniform, axis=1)

class PolicyFullyConnected:

    def __init__(self, observation_space, action_space, batch_size, reuse):
        height = observation_space[0]
        width = observation_space[1]
        self.observations = tf.placeholder(shape=(batch_size, height, width), dtype=tf.float32)

        with tf.variable_scope(name_or_scope="model", reuse=reuse):
            reshaped_observations = tf.reshape(tensor=tf.to_float(self.observations),
                             shape=(batch_size, height * width))

            hidden = tf.layers.dense(inputs=reshaped_observations,
                                     units=256,
                                     activation=tf.nn.relu)

            logits = tf.layers.dense(inputs=hidden, units=action_space)

            self.probs = tf.nn.softmax(logits)
            self.action = sample(self.probs)

            action_mask = tf.one_hot(self.action, action_space)

            self.log_probs = -tf.reduce_sum(action_mask * tf.log(self.probs + 1e-13),
                                            axis=1)

            self.value = tf.layers.dense(inputs=hidden, units=1)[:, 0]