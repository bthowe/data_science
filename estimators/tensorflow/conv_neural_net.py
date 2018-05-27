import matplotlib
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

import time
from datetime import datetime, timedelta

# import pylib.conv_widget as cw
# from pylib.tensorboardcmd import tensorboard_cmd


img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

n_classes = 10
n_channels = 1
filt_size = [5, 5] # 5x5 pixel filters

batch_size = 50
num_iterations = 400 #1500
display_step = 100

now = datetime.now()
logs_path = now.strftime("%Y%m%d-%H%M%S") + '/summaries'
print(logs_path)

def data_create():
    return input_data.read_data_sets('/tmp/data/', one_hot=True)

def _optimize(data, x, y_true, loss, accuracy, optimizer, num_iterations):
    # (data, x, y_label, loss, accuracy, train, training, n_steps, 1000)
    # Start-time used for printing time-usage below.
    start_time = time.time()

    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(init)
        train_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(logs_path + '/test', graph=tf.get_default_graph())

        step = 1
        for i in range(num_iterations):

            # Get a batch of training examples.
            x_batch, y_true_batch = data.train.next_batch(batch_size)

            # ---------------------- TRAIN -------------------------
            # Optimize model
            sess.run(optimizer, feed_dict={x: x_batch, y_true: y_true_batch})

            # Print status every 100 iterations.
            if (i % display_step == 0) or (i == num_iterations - 1):
                summary = sess.run(merged, feed_dict={x: x_batch, y_true: y_true_batch})
                train_writer.add_summary(summary, step)

                # ----------------------- TEST ---------------------------
                # Test model
                summary, l, acc = sess.run([merged, loss, accuracy], feed_dict={x: data.test.images,
                                                                                y_true: data.test.labels})
                test_writer.add_summary(summary, step)

                # Message for network evaluation
                msg = "Optimization Iteration: {0:>6}, Test Loss: {1:>6}, Test Accuracy: {2:>6.1%}"
                print(msg.format(i, l, acc))

                step += 1

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        train_writer.close()
        test_writer.close()

def conv_neural_net(data):
    data.test.cls = np.argmax(data.test.labels, axis=1)

    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    x_reshape = tf.reshape(x, shape=[-1, img_size, img_size, 1])
    out_conv = tf.layers.conv2d(x_reshape, 16, filt_size, padding='same', activation=tf.nn.relu, name="convolution")
    out_pool = tf.layers.max_pooling2d(out_conv, pool_size=(2, 2), strides=(2,2), padding='same')
    out_pool_reshape = tf.reshape(out_pool, [-1, out_pool.shape[1:].num_elements()])
    out = tf.layers.dense(out_pool_reshape, 100, activation=tf.nn.relu)
    y_pred = tf.layers.dense(out, n_classes, activation=None)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # _optimize(data, num_iterations)
    _optimize(data, x, y_true, loss, accuracy, optimizer, num_iterations)


def prediction():
    prediction = tf.argmax(y_pred, 1)

    def predict(idx):
        image = data.test.images[idx]
        return sess.run(prediction, feed_dict={x: [image]})

    idx = 0
    actual = np.argmax(data.test.labels[idx])
    print("Predicted: %d, Actual: %d" % (predict(idx), actual))
    plt.imshow(data.test.images[idx].reshape((28, 28)), cmap=plt.cm.gray_r)

if __name__ == '__main__':
    df = data_create()
    conv_neural_net(df)


# todo: get tensorboard working: make sure you're in the correct directory, type tensorboard --logdir=20180526-231700/summaries, and then go to localhost:6006
# todo: why not train and test like in the hierarchical nn
# todo: change the architecture to be like in the notes.
