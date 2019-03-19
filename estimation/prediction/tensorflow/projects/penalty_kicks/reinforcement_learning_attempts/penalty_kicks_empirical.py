import sys
import sets
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from itertools import product
from scipy.linalg import solve
from scipy.optimize import minimize
from tensorflow.python import debug as tf_debug

count = 0

p_miss = {
    'upper_left': 16 / 101,
    'upper_middle': 9 / 40,
    'upper_right': 16 / 101,
    'lower_left': 14 / 246,
    'lower_middle': 0 / 46,
    'lower_right': 14 / 246
}
p_block = {
    'upper_left': .7,
    'upper_middle': .8,
    'upper_right': .7,
    'lower_left': .7,
    'lower_middle': 1,
    'lower_right': .7
}  # conditional on the same keeper action
# nash equilibrium: [0.17548387, 0.15354839, 0.17548387, 0.17548387, 0.14451613, 0.17548387]


p_shot = {
    'upper_left': .01,
    'upper_middle': .01,
    'upper_right': .01,
    'lower_left': .01,
    'lower_middle': .95,
    'lower_right': .01
}
# what is the theoretical best response in this instance?


def kicker_location():
    return np.random.choice(list(p_shot.keys()), p=list(p_shot.values()))

def outcome(action):
    dic = {
        'upper_left': 0,
        'upper_middle': 1,
        'upper_right': 2,
        'lower_left': 3,
        'lower_middle': 4,
        'lower_right': 5}
    k = kicker_location()
    if (np.random.uniform(0, 1) < p_miss[k]):
        return 1
    if (dic[k] == action):
        if (np.random.uniform(0, 1) < p_block[k]):
            return 1
    return -1


def train():
    n_inputs = 6
    n_hidden = 6
    n_outputs = 6

    initializer = tf.contrib.layers.variance_scaling_initializer()

    learning_rate = 0.01
    X = tf.placeholder(tf.float32, shape=[1, n_inputs], name='X')
    # X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')

    # hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)

    logits = tf.layers.dense(X, n_outputs, kernel_initializer=initializer)

    # logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    outputs = tf.nn.softmax(logits, name='outputs')
    # p = tf.concat(axis=1, values=[outputs, 1 - outputs], name='p_left_and_right')

    action = tf.multinomial(tf.log(outputs), num_samples=1, name='action')
    # action = tf.multinomial(tf.log(p_left_and_right), num_samples=1, name='action')

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=action, logits=logits)

    print(action)
    print([0 if i != action[0][0] else 1 for i in range(6)])
    print(logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=[0, 0, 0, 0, 1, 0], logits=logits)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=[0 if i != action else 1 for i in range(6)], logits=logits)
    # y = 1. - tf.to_float(action)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    print(grads_and_vars)
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    def discount_rewards(rewards, discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(all_rewards, discount_rate):
        all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    # n_iterations = 10
    n_iterations = 750
    n_max_steps = 100
    n_games_per_update = 100
    save_iterations = 10
    # discount_rate = 1
    discount_rate = 0.95

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        init.run()
        for iteration in range(n_iterations):
            print("iteration: {}".format(iteration))
            all_rewards = []
            all_gradients = []
            for game in range(n_games_per_update):
                current_rewards = []
                current_gradients = []

                # obs = env.reset()
                obs = np.array([1] * n_inputs)
                # obs = np.array([1 / n_inputs] * n_inputs)
                for step in range(n_max_steps):
                    action_val, gradients_val, out_val = sess.run([action, gradients, outputs], feed_dict={X: obs.reshape(1, n_inputs)})
                    print(out_val)
                    print(action_val)

                    reward = outcome(action_val[0][0])
                    print(reward)  # todo: what should the reward values be?
                    print('\n')
                    # obs should just be the predicted values?

                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    # if done:
                    #     break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
                # env.render()

            all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
            feed_dict = {}
            for var_index, grad_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean(
                    [
                        reward * all_gradients[game_index][step][var_index] for game_index, rewards in
                        enumerate(all_rewards) for step, reward in enumerate(rewards)
                        ], axis=0
                )
                feed_dict[grad_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            print(out_val)
            # sys.exit()
            if iteration % save_iterations == 0:
                saver.save(sess, "./penalty_kicks.ckpt")


def test():
    n_inputs = 4
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('penalty_kicks.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()

        # for op in graph.get_operations():
        #     print(op.name)
        # sys.exit()

        # action = graph.get_operation_by_name('action/Multinomial_1:0')
        X = graph.get_tensor_by_name('X:0')
        prob = graph.get_tensor_by_name('p_left_and_right:0')

        obs = env.reset()
        for step in range(1000):
            print(step)
            prob_val = sess.run(prob, feed_dict={X: obs.reshape(1, n_inputs)})
            action_val = sess.run(tf.multinomial(tf.log(prob_val), num_samples=1, name='action'))[0][0]
            obs, reward, done, info = env.step(action_val)
            if done:
                break
            # print(obs.reshape(1, n_inputs))
            # sys.exit()

            env.render()
    env.close()


if __name__ == '__main__':
    train()
    # test()

    # result_dic = {
    #     'upper_left': 0,
    #     'upper_middle': 0,
    #     'upper_right': 0,
    #     'lower_left': 0,
    #     'lower_middle': 0,
    #     'lower_right': 0
    # }
    # for i in range(100000):
    #     k = kicker_location()
    #     result_dic[k] += 1
    #
    # print(result_dic)



# create data generator?



# todo: I don't understand why it's not making the 4 higher probability



# todo: instead of using a neural network, why don't I just focus on the parameters, which define the probability distribution over the actions.
# - use stochastic gradient descent
# - how would I do this given there isn't a label, though? todo


# baseline value would be, given the state, simulate what happens.
# total_return would be the discounted reward for the values in the rest of the episode

# for every episode in the group calculate
# 1. the total reward by discounting the future stream of payoffs
# 2. the reward for that episode by


# todo: is there something to tweak in my original code?
# todo: see if I can get that guys code to work here
