import numpy as np
import tensorflow as tf

def sequence_to_batch(input):
    units = input[0].get_shape()[-1].value
    return tf.reshape(tf.concat(axis=1, values=input), [-1, units])

def batch_to_sequence(input, batch_size):
    input = tf.reshape(input, [1, batch_size, -1])
    sequence = []
    for element in tf.split(axis=1, num_or_size_splits=batch_size, value=input):
        sequence.append(tf.squeeze(element, [1]))
    return sequence

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
        cell = cell*(1-done)
        hidden = hidden*(1-done)
        gate_inputs = tf.matmul(input, input_weights) + tf.matmul(hidden, hidden_weights) + bias
        input_gate, forget_gate, output_gate, new_input = tf.split(value=gate_inputs,
                                                                   num_or_size_splits=4,
                                                                   axis=1
                                                                   )
        cell = sigmoid(forget_gate)*cell + sigmoid(input_gate)*tanh(new_input)
        hidden = sigmoid(output_gate)*tanh(cell)
        input_sequence[index] = hidden
    state = tf.concat(axis=1, values=[cell, hidden])
    return input_sequence, state

class LstmPolicy():

    def __init__(self, observation_space, action_space, batch_size, units, reuse=False):
        self.inputs = tf.placeholder(tf.float32, (batch_size, 80, 80, 4))
        self.masks = tf.placeholder(tf.float32, [batch_size])
        self.states = tf.placeholder(tf.float32, [1, units * 2])
        with tf.variable_scope("model", reuse=reuse):
            inputs_reshaped = tf.reshape(self.inputs, [batch_size, 80 * 80 * 4])
            hidden = tf.layers.dense(inputs=inputs_reshaped, units=256, activation=tf.nn.relu)
            hidden_sequence = batch_to_sequence(hidden, batch_size)
            mask_sequence = batch_to_sequence(self.masks, batch_size)
            hidden2, self.new_states = lstm(hidden_sequence, mask_sequence, self.states, units=units)
            hidden2 = sequence_to_batch(hidden2)
            logits = tf.layers.dense(inputs=hidden2,
                                     units=action_space,
                                     activation=None)
            self.probs = tf.nn.softmax(logits)
        self.initial_state = np.zeros((1, units * 2), dtype=np.float32)