import tensorflow as tf


class PolicyFullyConnected:
    def __init__(self, observation_space, action_space):
        height, width = observation_space
        self.inputs = tf.placeholder(tf.float32, (None, height, width))
        inputs_reshaped = tf.reshape(self.inputs, [tf.shape(self.inputs)[0], width * height])
        hidden = tf.layers.dense(inputs=inputs_reshaped, units=256, activation=tf.nn.relu)
        logits_policy = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        logits_values = tf.layers.dense(inputs=hidden, units=1, activation=None)

        self.values = logits_values[:, 0]
        self.policy = tf.nn.softmax(logits_policy)


def sample(probs):
    random_uniform = tf.random_uniform(tf.shape(probs))
    scaled_random_uniform = tf.log(random_uniform) / probs
    return tf.argmax(scaled_random_uniform, axis=1)


class Model:
    def __init__(self, policy, observation_space, action_space, learning_rate):
        self.session = tf.Session()

        self.actions = tf.placeholder(tf.uint8, [None], name="action")
        self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
        self.advantage = tf.placeholder(tf.float32, [None])

        self.learning_rate = tf.Variable(trainable=False, initial_value=learning_rate)
        self.model = policy(observation_space, action_space)
        self.sampled_actions = sample(self.model.policy)

        action_mask = tf.one_hot(self.actions, action_space)
        log_probs = -tf.reduce_sum(action_mask * tf.log(self.model.policy + 1e-13), axis = 1)
        policy_loss = tf.reduce_mean(log_probs * self.advantage)

        value_loss = tf.reduce_mean(tf.squared_difference(self.model.values, self.rewards))
        loss = policy_loss + value_loss

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99).minimize(loss)
        self.session.run(tf.global_variables_initializer())

    def train(self, inputs, rewards, actions, values):

        advantage = rewards - values
        self.session.run(self.optimizer, feed_dict={
            self.model.inputs: inputs,
            self.rewards: rewards,
            self.actions: actions,
            self.advantage: advantage
        })

    def predict_action(self, observations):
        actions = self.session.run(self.sampled_actions, feed_dict={self.model.inputs: observations})
        return actions

    def predict_value(self, observations):
        return self.session.run(self.model.values, feed_dict={self.model.inputs: observations})

    def predict(self, observations):
        actions, values = self.session.run([self.sampled_actions, self.model.values],
                                          feed_dict={self.model.inputs: observations})
        return actions[0], values[0]

    def save(self, id):
        variables = tf.trainable_variables()
        saver = tf.train.Saver(variables)
        saver.save(self.session, "saver/model_{}.ckpt".format(id), write_meta_graph=False)

    def load(self, id):
        variables = tf.trainable_variables()
        saver = tf.train.Saver(variables)
        saver.restore(self.session, "saver/model_{}.ckpt".format(id))