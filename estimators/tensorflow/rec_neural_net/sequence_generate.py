# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import sets
import random
import functools
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from tensorflow.examples.tutorials.mnist import input_data


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceGenerate(object):
    def __init__(self, x, y, lstm_init_value, lstm_size, num_layers, num_chars):
        self.x = x
        self.y = y
        self.lstm_init_value = lstm_init_value
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_chars = num_chars

        self.x_enc = tf.one_hot(x, depth=self.num_chars)
        self.y_enc = tf.one_hot(self.y, depth=self.num_chars)

        # self.state = None
        # self.out = None

        self.prediction
        self.probs
        self.cost
        self.optimize

    @lazy_property
    def prediction(self):
        def make_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)

        network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)], state_is_tuple=False)
        output, state = tf.nn.dynamic_rnn(network, self.x_enc, initial_state=self.lstm_init_value, dtype=tf.float32)
        self.out = tf.shape(output)
        self.state = state

        outputs_reshaped = tf.reshape(output, [-1, self.lstm_size])
        return tf.layers.dense(outputs_reshaped, self.num_chars, activation=None)

    @lazy_property
    def probs(self):
        return tf.reshape(tf.nn.softmax(self.prediction), (self.out[0], self.out[1], self.num_chars))

    @lazy_property
    def cost(self):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=tf.reshape(self.y_enc, [-1, self.num_chars]))
        return tf.reduce_mean(xentropy)

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9)
        return optimizer.minimize(self.cost)

def data_create():
    txt = open('../data_files/strata_abstracts.txt', 'r').read().lower()
    data_index = list(''.join(OrderedDict.fromkeys(txt).keys()))
    data = [data_index.index(c) for c in txt]
    return data_index, data

def _get_next_batch(batch_size, time_steps, data):
    '''Returns batch of covariates and corresponding outcomes of size "batch_size" and length "time_steps"'''
    x_batch = np.zeros((batch_size, time_steps))
    y_batch = np.zeros((batch_size, time_steps))

    batch_ids = range(len(data) - time_steps - 1)
    batch_id = random.sample(batch_ids, batch_size)

    for t in range(time_steps):
        x_batch[:, t] = [data[i + t] for i in batch_id]
        y_batch[:, t] = [data[i + t + 1] for i in batch_id]

    return x_batch, y_batch


def run_step(sess, model, x, lstm_init_value, seed, chars, init_value):
    test_data = [[chars.index(c) for c in seed]]

    out, next_lstm_state = sess.run([model.probs, model.state], {x: test_data, lstm_init_value: [init_value]})

    return out[0][0], next_lstm_state[0]


def generate_text(sess, model, x, lstm_init_value, seed, n_layers, lstm_size, chars, len_test_txt=500):
    seed = seed.lower()

    lstm_last_state = np.zeros((n_layers * 2 * lstm_size,))
    for c in seed:
        out, lstm_last_state = run_step(sess, model, x, lstm_init_value, c, chars, lstm_last_state)

    gen_str = seed
    for i in range(len_test_txt):
        ele = np.random.choice(range(len(chars)), p=out)
        gen_str += chars[ele]
        out, lstm_last_state = run_step(sess, model, x, lstm_init_value, chars[ele], chars, lstm_last_state)

    return gen_str


def main():
    data_idx, data = data_create()  # data is 1525206 characters long. For training, I'll randomly select 50 sequences of length 100.
    n_chars = len(data_idx)

    batch_size = 50
    num_iterations = 10000

    time_steps = 100  # sequence length

    num_layers = 2
    lstm_size = 256

    display_step = 50

    x = tf.placeholder(tf.int32, shape=(None, None), name="x")
    y_true = tf.placeholder(tf.int32, (None, None))
    lstm_init_value = tf.placeholder(tf.float32, shape=(None, num_layers * 2 * lstm_size), name="lstm_init_value")  # there are two state variables...I need to learn more about lstm cells
    model = SequenceGenerate(x, y_true, lstm_init_value, lstm_size, num_layers, n_chars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(num_iterations):
        x_batch, y_true_batch = _get_next_batch(batch_size, time_steps, data)

        init_value = np.zeros((x_batch.shape[0], num_layers * 2 * lstm_size))
        sess.run(model.optimize, feed_dict={x: x_batch, y_true: y_true_batch, lstm_init_value: init_value})

        if (i % display_step == 0) or (i == num_iterations - 1):
            l = sess.run(model.cost, feed_dict={x: x_batch, y_true: y_true_batch, lstm_init_value: init_value})

            gen_str = generate_text(sess, model, x, lstm_init_value, 'We', num_layers, lstm_size, data_idx, 50)

            # # todo: go through generate text code, clean things up.
            # len_test_txt = 50
            # seed = 'we'
            # lstm_last_state = np.zeros((num_layers * 2 * lstm_size,))
            # for c in seed:
            #     test_data = [[data_idx.index(see) for see in c]]
            #     out, next_lstm_state = sess.run([model.probs, model.state], {x: test_data, lstm_init_value: [lstm_last_state]})
            #     out = out[0][0]
            #     lstm_last_state = next_lstm_state[0]
            #
            # gen_str = seed
            # for i in range(len_test_txt):
            #     ele = np.random.choice(range(len(data_idx)), p=out)
            #     gen_str += data_idx[ele]
            #
            #     test_data = [[data_idx.index(c) for c in data_idx[ele]]]
            #
            #     out, next_lstm_state = sess.run([model.probs, model.state], {x: test_data, lstm_init_value: [lstm_last_state]})
            #     out = out[0][0]
            #     lstm_last_state = next_lstm_state[0]

            msg = "Optimization Iteration: {0:>6}, Training Loss: {1:>6}"
            print(msg.format(i, l))
            print("  " + gen_str)



if __name__ == '__main__':
    main()



# todo: get this working in this format
# todo: get the engine one working
# todo: get the math exercise prediction one working
