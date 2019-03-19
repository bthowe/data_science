import sys
import functools
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import gym
env = gym.make('CartPole-v0').env


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification(object):
    def __init__(self, data, initalizer, num_outputs, num_hidden):
        self.data = data
        self.initializer = initalizer
        self._num_outputs = num_outputs
        self._num_hidden = num_hidden

        self.prediction
        self.target
        self.optimize

    @lazy_property
    def prediction(self):
        hidden = tf.layers.dense(self.data, self._num_hidden, activation=tf.nn.elu, kernel_initializer=self.initializer)
        return tf.layers.dense(hidden, self._num_outputs, kernel_initializer=self.initializer)

    @lazy_property
    def target(self):
        outputs = tf.nn.sigmoid(self.prediction, name='outputs')
        p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs], name='p_left_and_right')
        return tf.multinomial(tf.log(p_left_and_right), num_samples=1, name='action')
        # return 1. - tf.to_float(action)

    @lazy_property
    def cost(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=1. - tf.to_float(self.target), logits=self.prediction)

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return self.optimizer.compute_gradients(self.cost)


def main():
    n_inputs = 4
    n_outputs = 1
    n_hidden = 4
    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
    model = SequenceClassification(X, initializer, n_outputs, n_hidden)

    grads_and_vars = model.optimize
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = model.optimizer.apply_gradients(grads_and_vars_feed)

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

    n_iterations = 2000
    n_max_steps = 1000
    n_games_per_update = 10
    save_iterations = 10
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

                obs = env.reset()
                for step in range(n_max_steps):
                    action_val, gradients_val = sess.run([model.target, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
                # env.render()

            all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
            feed_dict = {}
            for var_index, grad_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean(
                    [
                        reward * all_gradients[game_index][step][var_index] for game_index, rewards in enumerate(all_rewards) for step, reward in enumerate(rewards)
                    ], axis=0
                )
                feed_dict[grad_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % save_iterations == 0:
                saver.save(sess, "./my_policy_net_pg.ckpt")

    env.close()

def test():
    n_inputs = 4
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('my_policy_net_pg.ckpt.meta')
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
    # main()
    test()


# todo: this is not updating like it should