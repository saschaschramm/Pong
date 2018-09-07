import tensorflow as tf

class Model:
    def __init__(self, policy, observation_space, action_space, batch_size, ent_coef, vf_coef):
        self.session = tf.Session()

        self.model_predict = policy(observation_space, action_space, 1, False)
        self.model_train = policy(observation_space, action_space, batch_size, True)
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.returns = tf.placeholder(tf.float32, [None])
        self.old_log_probs = tf.placeholder(tf.float32, [None])
        self.old_values = tf.placeholder(tf.float32, [None])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.clip_range = tf.placeholder(tf.float32, [])

        self.entropy = -tf.reduce_mean(tf.reduce_sum(self.model_train.probs * tf.log(self.model_train.probs + 1e-13),
                                                     axis=1,
                                                     keepdims=True))

        # Value
        values = self.model_train.value
        values_clipped = self.old_values + tf.clip_by_value(values - self.old_values, -self.clip_range, self.clip_range)
        value_losses = tf.square(values - self.returns)
        value_losses_clipped = tf.square(values_clipped - self.returns)
        self.value_loss = .5 * tf.reduce_mean(tf.maximum(value_losses, value_losses_clipped))

        # Policy
        action_mask = tf.one_hot(self.actions, action_space)
        log_probs = -tf.reduce_sum(action_mask * tf.log(self.model_train.probs + 1e-13),
                                   axis=1)

        ratio = tf.exp(self.old_log_probs - log_probs)
        ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        policy_losses = -self.advantages * ratio
        policy_losses_clipped = -self.advantages * ratio_clipped
        self.policy_loss = tf.reduce_mean(tf.maximum(policy_losses, policy_losses_clipped))

        loss = self.policy_loss - self.entropy * ent_coef + self.value_loss * vf_coef
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        self.session.run(tf.global_variables_initializer())

    def predict(self, observation):
        actions, values, log_probs = self.session.run([self.model_predict.action,
                                                       self.model_predict.value,
                                                       self.model_predict.log_probs],
                                                      feed_dict={self.model_predict.observations: [observation]})
        return actions[0], values[0], log_probs[0]

    def predict_value(self, observation):
        return self.session.run(self.model_predict.value,
                                feed_dict={self.model_predict.observations: [observation]})[0]

    def train(self,
              learning_rate,
              clip_range,
              observations,
              returns,
              actions,
              values,
              log_probs):

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        feed_dict = {self.model_train.observations: observations,
                     self.actions: actions,
                     self.advantages: advantages,
                     self.returns: returns,
                     self.learning_rate: learning_rate,
                     self.clip_range: clip_range,
                     self.old_log_probs: log_probs,
                     self.old_values: values}

        self.session.run([self.optimizer], feed_dict)