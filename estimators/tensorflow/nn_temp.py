import sys
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

import time
from datetime import datetime, timedelta

np.random.seed(42)


N_PIXELS=28 * 28
N_CLASSES=10
dropout = 0.2
hidden_size = 5
BATCH_SIZE = 100
n_steps = 500
alpha = 0.0001
# learning_rate = 0.5
early_stopping_num = 5

def data_create():
    return input_data.read_data_sets('/tmp/data', one_hot=True)

def _train_mnist(data, x, y_label, loss, accuracy, train, training, steps_total, steps_print):
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        best_model_dict = {'loss': 10 ** 5 }
        loss_lags = [10 ** 5] * early_stopping_num
        for i in range(steps_total):
            print(i)
            batch_x, batch_y = data.train.next_batch(BATCH_SIZE)
            sess.run(train, feed_dict={x: batch_x, y_label: batch_y, training: True})

            loss_current = sess.run(loss, feed_dict={x: batch_x, y_label: batch_y, training: True})
            print(loss_current)
            if loss_current <= best_model_dict['loss']:
                best_model_dict['loss'] = loss_current
                saver.save(sess, './mnis_model')  #, global_step=i)

            early_stopping_value = loss_lags.pop()
            loss_lags = [loss_current] + loss_lags
            if np.all(early_stopping_value < np.array(loss_lags)):
                l, a = sess.run([loss, accuracy], feed_dict={x: data.test.images, y_label: data.test.labels, training: False})
                metrics['test_loss'].append(l)
                metrics['test_acc'].append(a)
                print("Test:  %0.5f, %0.5f" % (l, a))

                l, a = sess.run([loss, accuracy], feed_dict={x: data.train.images, y_label: data.train.labels, training: False})
                metrics['train_loss'].append(l)
                metrics['train_acc'].append(a)
                print("Train: %0.5f, %0.5f" % (l, a))
                break

            if i % steps_print == 0 or i == steps_total - 1:
                l, a = sess.run([loss, accuracy], feed_dict={x: data.test.images, y_label: data.test.labels, training: False})
                metrics['test_loss'].append(l)
                metrics['test_acc'].append(a)
                print("Test:  %0.5f, %0.5f" % (l, a))

                l, a = sess.run([loss, accuracy], feed_dict={x: data.train.images, y_label: data.train.labels, training: False})
                metrics['train_loss'].append(l)
                metrics['train_acc'].append(a)
                print("Train: %0.5f, %0.5f" % (l, a))

                print('\n')

    return metrics

def hierarchical_neural_net_dropout(data):
    x = tf.placeholder(tf.float32, [None, N_PIXELS], name="pixels")
    y_label = tf.placeholder(tf.float32, [None, 10], name="labels")
    training = tf.placeholder(tf.bool, name="training")

    drop1 = tf.layers.dropout(
        x,
        dropout,
        training=training
    )
    hidden1 = tf.layers.dense(
        drop1,
        hidden_size,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=N_PIXELS**-0.5)
    )

    drop2 = tf.layers.dropout(
        hidden1,
        dropout,
        training=training
    )
    hidden2 = tf.layers.dense(
        drop2,
        hidden_size,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=hidden_size**-0.5)
    )

    drop3 = tf.layers.dropout(
        hidden2,
        dropout,
        training=training
    )
    y = tf.layers.dense(
        drop3,
        10,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=hidden_size ** -0.5)
    )

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                                  labels=y_label))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1)), tf.float32))

    _train_mnist(data, x, y_label, loss, accuracy, train, training, n_steps, 1000)

def hierarchical_neural_net_regularize(data):
    x = tf.placeholder(tf.float32, [None, N_PIXELS], name="pixels")
    y_label = tf.placeholder(tf.float32, [None, 10], name="labels")
    training = tf.placeholder(tf.bool, name="training")

    hidden1 = tf.layers.dense(
        x,
        hidden_size,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=N_PIXELS ** -0.5),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha)
    )
    hidden2 = tf.layers.dense(
        hidden1,
        hidden_size,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=N_PIXELS ** -0.5),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha)
    )
    y = tf.layers.dense(
        hidden2,
        10,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=hidden_size**-0.5),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha)
    )

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss + tf.add_n(reg_loss))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1)), tf.float32))

    _train_mnist(data, x, y_label, loss, accuracy, train, training, n_steps, 1000)

if __name__ == '__main__':
    df = data_create()
    hierarchical_neural_net_dropout(df)
    # hierarchical_neural_net_regularize(df)

# todo: normally with early stopping do we look after we slipped through all of the observations? ...not a single batch.