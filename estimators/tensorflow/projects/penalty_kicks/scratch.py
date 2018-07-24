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
                action_val, gradients_val, out_val = sess.run([action, gradients, outputs],
                                                              feed_dict={X: obs.reshape(1, n_inputs)})
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

