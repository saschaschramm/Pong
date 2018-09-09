import tensorflow as tf

class PolicyFullyConnected():
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

"""
0 0.0 0.0
20000 -21.0 69.7
40000 -20.2 70.5
60000 -19.5 71.3
80000 -18.6 71.2
100000 -17.9 68.4
120000 -17.0 67.6
140000 -18.6 67.6
160000 -18.7 69.1
180000 -17.3 69.6
200000 -18.2 75.6
220000 -19.0 72.9
240000 -17.2 76.1
260000 -14.8 72.7
280000 -17.1 70.4
300000 -18.9 70.8
320000 -17.2 67.5
340000 -15.6 67.0
360000 -17.1 68.3
380000 -15.9 69.0
400000 -17.5 68.0
420000 -17.6 69.6
440000 -15.2 69.9
460000 -14.9 67.4
480000 -17.9 67.1
500000 -18.5 69.2
520000 -15.9 69.3
540000 -17.9 68.5
560000 -17.5 71.1
580000 -16.4 66.9
600000 -15.0 66.1
620000 -13.4 65.9
640000 -13.8 68.1
660000 -13.4 68.1
680000 -11.1 67.0
700000 -8.0 66.3
720000 -5.9 66.4
740000 -5.7 67.1
760000 -4.3 66.7
780000 -9.0 66.9
800000 -8.7 68.2
820000 1.6 67.2
840000 5.8 69.8
860000 2.0 68.2
880000 2.8 66.6
900000 1.5 68.5
920000 1.0 67.3
940000 1.4 67.4
960000 -0.8 66.5
980000 0.0 66.4
1000000 3.0 69.0
"""



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
        policy_loss = -tf.reduce_mean(tf.reduce_sum(action_mask * tf.log(self.model.policy + 1e-13), axis = 1) *
                                      self.advantage)

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