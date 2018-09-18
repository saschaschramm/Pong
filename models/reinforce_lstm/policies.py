import numpy as np
import tensorflow as tf


def lstm(input_sequence, mask_sequence, state, units):
    input_weights = tf.get_variable(name="input_weights", shape=[units, units * 4])
    hidden_weights = tf.get_variable(name="hidden_weights", shape=[units, units * 4])
    bias = tf.get_variable(name="bias_weights",
                           shape=[units * 4],
                          initializer=tf.constant_initializer(0.0))

    sigmoid = tf.nn.sigmoid
    tanh = tf.tanh

    cell, hidden = tf.split(axis=1, num_or_size_splits=2, value=state)

    for index, (input, done) in enumerate(zip(input_sequence, mask_sequence)):
        cell = cell * (1-done)
        hidden = hidden * (1-done)
        gate_inputs = tf.matmul([input], input_weights) + tf.matmul(hidden, hidden_weights) + bias
        input_gate, forget_gate, output_gate, new_input = tf.split(value=gate_inputs,
                                                                   num_or_size_splits=4,
                                                                   axis=1
                                                                   )
        cell = sigmoid(forget_gate)*cell + sigmoid(input_gate)*tanh(new_input)
        hidden = sigmoid(output_gate)*tanh(cell)
        input_sequence[index] = hidden
    state = tf.concat(axis=1, values=[cell, hidden])
    return input_sequence, state


class LstmPolicy:

    def __init__(self, observation_space, action_space, batch_size, units, reuse=False):
        height = observation_space[0]
        width = observation_space[1]
        self.inputs = tf.placeholder(tf.float32, (batch_size, height, width))
        self.masks = tf.placeholder(tf.float32, [batch_size])
        self.states = tf.placeholder(tf.float32, [1, units * 2])

        with tf.variable_scope("model", reuse=reuse):
            inputs_reshaped = tf.reshape(self.inputs, [batch_size, height * width])
            hidden = tf.layers.dense(inputs=inputs_reshaped, units=256, activation=tf.nn.relu)

            # (128, 256) -> [256, 256, ... ]
            hidden_sequence = tf.unstack(hidden, axis=0)
            mask_sequence = tf.unstack(self.masks, axis=0)
            hidden2, self.new_states = lstm(hidden_sequence, mask_sequence, self.states, units=units)
            hidden2=tf.concat(axis=0, values=hidden2)
            logits = tf.layers.dense(inputs=hidden2,
                                     units=action_space,
                                     activation=None)

            self.probs = tf.nn.softmax(logits)

        self.initial_state = np.zeros((1, units * 2), dtype=np.float32)