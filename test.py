import itertools

import tensorflow as tf

from actor_critic_player import ActorCriticPlayer
from game_controller import GameController
from policy_estimator import PolicyEstimator
from reinforce_player import ReinforcePlayer
from value_estimator import ValueEstimator


player1 = ActorCriticPlayer(1.0, "Policy_Estimator_Black", "Value_Estimator_Black")
player2 = ActorCriticPlayer(-1.0, "Policy_Estimator_White", "Value_Estimator_White")
# player1 = ReinforcePlayer(1.0, "Policy_Estimator_Black", discount_factor=0.99)
# player2 = ReinforcePlayer(-1.0, "Policy_Estimator_White", discount_factor=0.99)
# policy_estimator = PolicyEstimator("policy_estimator")
# value_estimator = ValueEstimator("policy_estimator", discount_factor=0.99)
# player1 = ActorCriticPlayer(1.0, policy_estimator, value_estimator)
# player2 = ActorCriticPlayer(-1.0, policy_estimator, value_estimator)

game_controller = GameController(player1, player2)

write_summary_every = 100
print_wins_and_draws_every = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary = tf.summary.merge_all()
    tensorflow_writer = tf.summary.FileWriter("/tmp/gomoku/0")
    tensorflow_writer.add_graph(sess.graph)

    draws = 0
    white_wins = 0
    black_wins = 0
    # for i in itertools.count():
    for i in range(11):
        player1_starts = False
        if i % 2 == 0:
            player1_starts = True

        print_board = (i % 10 == 0)
        if print_board:
            print("Playing game", i)

        winner_value = game_controller.play_game(sess, player_one_starts=player1_starts, print_board=print_board)

        if winner_value == 0.0:
            draws += 1
        elif winner_value == -1.0:
            white_wins += 1
        else:
            black_wins += 1

        if i % write_summary_every == 0:
            print("Writing summary")
            s = sess.run(summary)
            tensorflow_writer.add_summary(s, i)

        if i % print_wins_and_draws_every == 0:
            print("Black wins: {}, White wins: {}, Draws: {}".format(black_wins, white_wins, draws))

    # save model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/tmp/gomoku/model.ckpt")
    print("Model saved in file: %s" % save_path)
