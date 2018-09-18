import tensorflow as tf


def sample(probs):
    random_uniform = tf.random_uniform(tf.shape(probs))
    scaled_random_uniform = tf.log(random_uniform) / probs
    return tf.argmax(scaled_random_uniform, axis=1)

"""
gumbel trick
def sample(logits):
    u = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
"""


class Model:

    def __init__(self, policy, observation_space, action_space, batch_size):
        self.session = tf.Session()
        self.actions = tf.placeholder(tf.int32, [batch_size])
        self.rewards = tf.placeholder(tf.float32, [batch_size])

        lstm_units = 256
        self.predict_model = policy(observation_space, action_space,
                                    batch_size=1,
                                    units=lstm_units,
                                    reuse=False)

        self.train_model = policy(observation_space, action_space,
                                  batch_size=batch_size,
                                  units=lstm_units,
                                  reuse=True)

        action_mask = tf.one_hot(self.actions, action_space)
        log_probs = -tf.reduce_sum(action_mask * tf.log(self.train_model.probs + 1e-13),
                                   axis=1)

        self.loss = tf.reduce_mean(self.rewards * log_probs)

        self.action = sample(self.predict_model.probs)
        self.optimize = tf.train.RMSPropOptimizer(learning_rate=2e-4, decay=0.99).minimize(self.loss)

        self.initial_state = self.predict_model.initial_state
        tf.global_variables_initializer().run(session=self.session)

    def train(self, observations, states, rewards, masks, actions):
        feed_dict = {self.train_model.inputs: observations,
                     self.actions: actions,
                     self.rewards: rewards,
                     self.train_model.states: states,
                     self.train_model.masks: masks
                     }
        loss, _ = self.session.run([self.loss, self.optimize],
                                          feed_dict=feed_dict)
        return loss

    def predict_action(self, observations, states, masks):

        feed_dict = {self.predict_model.inputs: observations,
                     self.predict_model.states: states,
                     self.predict_model.masks: masks
                     }

        actions, states = self.session.run([self.action, self.predict_model.new_states],
                                           feed_dict=feed_dict)
        return actions[0], states