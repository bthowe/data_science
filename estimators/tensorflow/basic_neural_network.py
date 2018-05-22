import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)

def data_create(plot=False):
    centers = np.array([[0, 0]] * 100 + [[1, 1]] * 100 + [[0, 1]] * 100 + [[1, 0]] * 100)
    data = np.random.normal(0, 0.2, (400, 2)) + centers
    labels = np.array([0] * 200 + [1] * 200)

    if plot:
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.RdYlBu)
        plt.colorbar()
        plt.show()

    return pd.concat(
        [
            pd.DataFrame(data, columns=['x1', 'x2']),
            pd.DataFrame(labels, columns=['y'])
        ],
        axis=1
    )

def neural_net(data, activation_function, hidden_size):
    x = data
    y = x.pop('y').values.reshape(len(x), 1)
    X = x.values

    x = tf.placeholder(tf.float32, [None, 2], name="features")
    y_label = tf.placeholder(tf.float32, [None, 1], name="labels")

    W1 = tf.Variable(tf.random_normal([2, hidden_size], seed=42), name="weight1")
    b1 = tf.Variable(tf.zeros([hidden_size]), name="bias1")

    hidden = tf.nn.sigmoid(tf.matmul(x, W1) + b1, name="hidden")

    W2 = tf.Variable(tf.random_normal([hidden_size, 1], seed=24), name="weight2")
    b2 = tf.Variable(tf.zeros([1]), name="bias2")

    y_pred = tf.matmul(hidden, W2) + b2

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_label))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    predicted = tf.cast(activation_function(y_pred) > 0.5, np.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_label), np.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(3000):
            sess.run(train, feed_dict={x: X, y_label: y})
            if i % 300 == 0:
                print(sess.run([loss, accuracy], feed_dict={x: X, y_label: y}))

def leaky_relu(x):
    return tf.maximum(0.01 * x, x)

if __name__ == '__main__':
    # tf.nn.sigmoid()
    # tf.nn.tanh()
    # tf.nn.relu()
    # leaky_relu()
    # tf.nn.elu()

    activation_function = tf.nn.elu
    hidden_size = 4
    data_create().pipe(neural_net, activation_function, hidden_size)
