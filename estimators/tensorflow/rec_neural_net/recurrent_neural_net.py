import sys
import matplotlib
import numpy as np
import tensorflow as tf

import os

import time
import random
from datetime import datetime, timedelta

from collections import OrderedDict


def data_create():
    txt = open('data_files/strata_abstracts.txt', 'r').read().lower()
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


def _train(placeholders, outputs, loss, optimizer):
# def _train(x, y_true, y_pred, lstm_new_state, lstm_init_value, loss, optimizer):
    x = placeholders['x']
    y_true = placeholders['y_true']
    lstm_init_value = placeholders['lstm_init_value']

    prob = outputs['prob']
    state = outputs['state']

    start_time = time.time()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(init)

        # Create summary writers
        logs_path = datetime.now().strftime("%Y%m%d-%H%M%S") + '/summaries'
        train_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())

        step = 1
        for i in range(num_iterations):

            x_batch, y_true_batch = _get_next_batch(batch_size, time_steps, data)

            init_value = np.zeros((x_batch.shape[0], n_layers * 2 * lstm_size))
            sess.run(optimizer, feed_dict={x: x_batch, y_true: y_true_batch, lstm_init_value: init_value})

            if (i % display_step == 0) or (i == num_iterations - 1):
                saver.save(sess, './strata_model', global_step=i)

                summary, l = sess.run([merged, loss], feed_dict={x: x_batch, y_true: y_true_batch, lstm_init_value: init_value})
                train_writer.add_summary(summary, step)

                msg = "Optimization Iteration: {0:>6}, Training Loss: {1:>6}"
                print(msg.format(i, l))
                print("  " + generate_text(sess, {'x': x, 'lstm_init_value': lstm_init_value}, {'prob': prob, 'state': state}, 'We', 60))
                # print("  " + generate_text(sess, prob, state, x, lstm_init_value, 'We', 60))

                step += 1

        time_dif = time.time() - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        train_writer.close()


def rnn_architecture(x, lstm_init_value):
    # LSTM
    lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False) for _ in range(n_layers)], state_is_tuple=False)

    # Iteratively compute output of recurrent network
    out, lstm_new_state = tf.nn.dynamic_rnn(lstm, x, initial_state=lstm_init_value, dtype=tf.float32)

    # Linear activation (FC layer on top of the LSTM net)
    out_reshaped = tf.reshape(out, [-1, lstm_size])
    y = tf.layers.dense(out_reshaped, n_chars, activation=None)

    return y, tf.shape(out), lstm_new_state


def rnn_run():
    x = tf.placeholder(tf.int32, shape=(None, None), name="x")
    y_true = tf.placeholder(tf.int32, (None, None))
    lstm_init_value = tf.placeholder(tf.float32, shape=(None, n_layers * 2 * lstm_size), name="lstm_init_value")  # there are two state variables...I need to learn more about lstm cells

    x_enc = tf.one_hot(x, depth=n_chars)
    y_true_enc = tf.one_hot(y_true, depth=n_chars)

    y_pred, out, lstm_new_state = rnn_architecture(x_enc, lstm_init_value)

    final_out = tf.reshape(tf.nn.softmax(y_pred), (out[0], out[1], n_chars), name='pred')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.reshape(y_true_enc, [-1, n_chars])))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.RMSPropOptimizer(0.003, 0.9).minimize(loss)

    _train(
        {'x': x, 'y_true': y_true, 'lstm_init_value': lstm_init_value},
        {'prob': final_out, 'state': lstm_new_state},
        loss,
        optimizer
    )
    # _train(x, y_true, final_out, lstm_new_state, lstm_init_value, loss, optimizer)


# todo: these three functions can be improved
def run_step(sess, placeholders, outcomes, seed, chars, init_value):
    x = placeholders['x']
    lstm_init_value = placeholders['lstm_init_value']

    prob = outcomes['prob']
    state = outcomes['state']

    test_data = [[chars.index(c) for c in seed]]

    out, next_lstm_state = sess.run([prob, state], {x: test_data, lstm_init_value: [init_value]})
    # I was thinking the output and the state should look the same at the last line.
    print(out)
    print(out.shape)
    print(out[0][0])
    print(next_lstm_state)
    print(next_lstm_state[0])
    print(next_lstm_state.shape)

    sys.exit()
    return out[0][0], next_lstm_state[0]

def generate_text(sess, placeholders, outcomes, seed, len_test_txt=500):
# def generate_text(sess, pred, lstm_new_state, x, lstm_init_value, seed, len_test_txt=500):

    seed = seed.lower()

    lstm_last_state = np.zeros((n_layers * 2 * lstm_size,))
    for c in seed:
        out, lstm_last_state = run_step(sess, placeholders, outcomes, c, data_idx, lstm_last_state)

    gen_str = seed
    for i in range(len_test_txt):
        ele = np.random.choice(range(len(data_idx)), p=out)
        gen_str += data_idx[ele]
        out, lstm_last_state = run_step(sess, placeholders, outcomes, data_idx[ele], data_idx, lstm_last_state)
    return gen_str


def generate(seed, len_test_txt=500):
    seed = seed.lower()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('strata_model-{}.meta'.format(num_iterations - 1))
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        prob = graph.get_tensor_by_name('pred:0')
        state = graph.get_tensor_by_name('rnn/while/Exit_3:0')
        x = graph.get_tensor_by_name('x:0')
        lstm_init_value = graph.get_tensor_by_name('lstm_init_value:0')

        return generate_text(sess, {'x': x, 'lstm_init_value': lstm_init_value}, {'prob': prob, 'state': state}, seed, len_test_txt)



if __name__ == '__main__':
    data_idx, data = data_create()
    time_steps = 100
    batch_size = 50
    num_iterations = 10000
    n_layers = 2
    n_chars = len(data_idx)
    lstm_size = 256
    display_step = 50

    rnn_run()

    print(generate('Hey there!'))
    # print(generate('Hel'))

# todo: https://www.tensorflow.org/tutorials/recurrent#lstm



# todo: what going on here: https://stackoverflow.com/questions/42440565/how-to-feed-back-rnn-output-to-input-in-tensorflow
