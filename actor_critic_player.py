import random
import numpy as np

from policy_estimator import PolicyEstimator
from state_builder import StateBuilder
from sum_tree import SumTree
from value_estimator import ValueEstimator


class ActorCriticPlayer(object):
    def __init__(self, value, policy_estimator_name, value_estimator_name):
    # def __init__(self, value, policy_estimator, value_estimator):
        self.value = value

        self.policy_estimator = PolicyEstimator(policy_estimator_name)
        self.value_estimator = ValueEstimator(value_estimator_name, discount_factor=0.99)
        # self.policy_estimator = policy_estimator
        # self.value_estimator = value_estimator

    def initialize_state(self, board):
        self.state = StateBuilder().build(board)
        # self.state = StateBuilder().build(board, self.value)

    def observe(self, session, board):
        next_state = StateBuilder().build(board)
        # next_state = StateBuilder().build(board, self.value)

        # Update policy estimator
        advantage = self.value_estimator.td_error(session, self.state, 0.0, next_state)
        self.policy_estimator.update(session, self.state, advantage, self.action)

        # Update value estimator
        self.value_estimator.update(session, self.state, 0.0, next_state, False)

        self.state = next_state

    def observe_finished(self, session, board):
        next_state = StateBuilder().build(board)
        # next_state = StateBuilder().build(board, self.value)

        reward = -1.0
        if board.get_winner_value() == 0.0:
            reward = 0.0
        elif board.get_winner_value() == self.value:
            reward = 1.0

        # Update policy estimator
        advantage = self.value_estimator.td_error(session, self.state, reward, next_state)
        self.policy_estimator.update(session, self.state, advantage, self.action)

        # Update value estimator
        self.value_estimator.update(session, self.state, reward, next_state, True)

    def predict_move(self, session):
        action_probs = self.policy_estimator.predict(session, self.state)
        # print(action_probs)

        x, y = self._pick_action_from_probs(action_probs)

        self.action = 15 * x + y
        return x, y

    def _pick_action_from_probs(self, probs):
        tree = SumTree(15 * 15)
        for action, action_prob in enumerate(probs):
            x = action // 15
            y = action % 15
            if self.state[x, y, 2] == 1.0:
                tree.add(action_prob, (x, y))

        total = tree.total()
        s = random.uniform(0, total)
        _, _, data = tree.get(s)

        return data
