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

def data_create():
    return input_data.read_data_sets('/tmp/data', one_hot=True)

def hierarchical_neural_net(data):
    batch_size = 100
    learning_rate = 0.5
    n_features = 28 * 28

    hidden_size = 256
    alpha = 0.0001
    dropout = 0.2

    x = tf.placeholder(tf.float32, [None, n_features], name="pixels")
    y_label = tf.placeholder(tf.float32, [None, 10], name="labels")
    # training = tf.placeholder(tf.bool, name="training")

    # sys.exit()
    drop1 = tf.layers.dropout(x, dropout)  #, training=training)
    hidden1 = tf.layers.dense(
        drop1,
        hidden_size,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=n_features ** -0.5)  #,
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha)
    )

    drop2 = tf.layers.dropout(hidden1, dropout)  #, training=training)
    hidden2 = tf.layers.dense(
        drop2,
        hidden_size,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=n_features ** -0.5)  #,
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha)
    )

    drop3 = tf.layers.dropout(hidden2, dropout)  #, training=training)
    y = tf.layers.dense(
        drop3,
        10,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=hidden_size**-0.5)  #,
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha)
    )

    # reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss + tf.add_n(reg_loss))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1)), tf.float32))


    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        early_stopping_num = 5
        burn_in = 200
        steps_total = 3000
        for i in range(steps_total):
            batch_x, batch_y = data.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y_label: batch_y})  #, training: True})

            if i <= burn_in:
                if i == 0:  # initialize saved models
                    saver.save(sess, './mnis_model')  # , global_step=i)
                    best_model_dict = {'best_model': 'mnis_model.meta', 'loss': sess.run(loss, feed_dict={x: batch_x, y_label: batch_y})}

                loss_current = sess.run(loss, feed_dict={x: batch_x, y_label: batch_y})
                if loss_current <= best_model_dict['loss']:
                    best_model_dict['best_model'] = 'mnis_model-{}.meta'.format(i)
                    best_model_dict['loss'] = loss_current
                    saver.save(sess, './mnis_model')  # , global_step=i)
                if i == burn_in:
                    early_stopping_counter = 0
            else:
                loss_current = sess.run(loss, feed_dict={x: batch_x, y_label: batch_y})
                if loss_current <= best_model_dict['loss']:
                    best_model_dict['best_model'] = 'mnis_model-{}.meta'.format(i)
                    best_model_dict['loss'] = loss_current
                    saver.save(sess, './mnis_model')  #, global_step=i)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print('counter: {}'.format(early_stopping_counter))
                    print('current loss: {}'.format(loss_current))
                    print('best model loss: {}'.format(best_model_dict['loss']))

                if ((early_stopping_counter == early_stopping_num) and (i > burn_in)):
                    print(sess.run([loss, accuracy], feed_dict={x: batch_x, y_label: batch_y}))
                    break

            print('iteration {0}: \n\t{1}'.format(i, sess.run([loss, accuracy], feed_dict={x: batch_x, y_label: batch_y})))
            # if i == 0:  # initialize saved models
            #     saver.save(sess, './mnis_model')  #, global_step=i)
            #     best_model_dict = {'best_model': 'mnis_model.meta', 'loss': sess.run(loss, feed_dict={x: batch_x, y_label: batch_y})}
            #     # early_stopping_counter = 0
            #
            # loss_current = sess.run(loss, feed_dict={x: batch_x, y_label: batch_y})
            # if loss_current <= best_model_dict['loss']:
            #     best_model_dict['best_model'] = 'mnis_model-{}.meta'.format(i)
            #     best_model_dict['loss'] = loss_current
            #     saver.save(sess, './mnis_model')  #, global_step=i)
            #     early_stopping_counter = 0
            # else:
            #     early_stopping_counter += 1
            #     print('counter: {}'.format(early_stopping_counter))
            #     print('current loss: {}'.format(loss_current))
            #     print('best model loss: {}'.format(best_model_dict['loss']))
            #
            # if ((early_stopping_counter == early_stopping_num) and (i > burn_in)):
            #     print(sess.run([loss, accuracy], feed_dict={x: batch_x, y_label: batch_y}))
            #     break
            #
            # print('iteration {0}: \n\t{1}'.format(i, sess.run([loss, accuracy], feed_dict={x: batch_x, y_label: batch_y})))

            # if i % 300 == 0 or i == steps_total - 1:
            #     print(sess.run([loss, accuracy], feed_dict={x: batch_x, y_label: batch_y}))


            # saver = tf.train.import_meta_graph(best_model)
            # saver.restore(sess, tf.train.latest_checkpoint('./'))


def train_mnist(x, y_label, loss, accuracy, train, training, steps_total, steps_print):
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    reset_vars()
    for i in xrange(steps_total):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train,
                 feed_dict={x: batch_x, y_label: batch_y, training: True})
        if i % steps_print == 0 or i == steps_total - 1:
            l, a = sess.run([loss, accuracy],
                            feed_dict={x: mnist.test.images,
                                       y_label: mnist.test.labels,
                                       training: False})
            metrics['test_loss'].append(l)
            metrics['test_acc'].append(a)
            print
            "Test:  %0.5f, %0.5f" % (l, a)
            l, a = sess.run([loss, accuracy],
                            feed_dict={x: mnist.train.images,
                                       y_label: mnist.train.labels,
                                       training: False})
            metrics['train_loss'].append(l)
            metrics['train_acc'].append(a)
            print
            "Train: %0.5f, %0.5f" % (l, a)
    return metrics

if __name__ == '__main__':
    df = data_create()
    hierarchical_neural_net(df)

#     todo: implement early stopping, dropout, regularization, train/test
# what should early stopping do? I'm wondering if my implementation of it is good or not.
# when use dropout?
# when use regularization?
