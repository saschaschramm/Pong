import collections
import tensorflow as tf

LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class BasicLSTMCell():

  def zero_state(self, batch_size):
    c = tf.zeros([batch_size] + [self._state_size.c])
    h = tf.zeros([batch_size] + [self._state_size.h])
    return LSTMStateTuple(c=c, h=h)

  def __init__(self, num_units, forget_bias):
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_size = LSTMStateTuple(num_units, num_units)

    self._bias = tf.get_variable(shape=[4 * self._num_units],
                                 name="bias",
                                 initializer=tf.constant_initializer(0.0))

    self._kernel = tf.get_variable("kernel", dtype=tf.float32, shape=[self._num_units, self._num_units * 4])

  def __call__(self, inputs, state_tuple):

      sigmoid = tf.sigmoid
      tanh = tf.tanh
      cell_state, hidden_state = state_tuple

      gate_inputs = tf.matmul(tf.concat([inputs, hidden_state], 1), self._kernel) + self._bias
      gate_inputs = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

      input_gate = sigmoid(gate_inputs[0])
      new_input = tanh(gate_inputs[1])
      forget_gate = sigmoid(gate_inputs[2] + self._forget_bias)
      output_gate = sigmoid(gate_inputs[3])

      new_cell_state = cell_state * forget_gate + input_gate * new_input
      new_hidden_state = tanh(new_cell_state) * output_gate

      return LSTMStateTuple(new_cell_state, new_hidden_state)