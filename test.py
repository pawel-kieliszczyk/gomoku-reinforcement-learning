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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tensorflow_writer = tf.summary.FileWriter("/tmp/gomoku/0")
    tensorflow_writer.add_graph(sess.graph)

    for i in itertools.count():
    # for i in range(1):
        print_board = (i % 10 == 0)
        if print_board:
            print("Playing game", i)
        game_controller.play_game(sess, print_board=print_board)
