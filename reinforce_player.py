import random
import numpy as np

from policy_estimator_reinforce import PolicyEstimatorReinforce
from state_builder import StateBuilder
from sum_tree import SumTree


class ReinforcePlayer(object):
    def __init__(self, value, policy_estimator_name, discount_factor):
        self.value = value
        self.discount_factor = discount_factor

        self.policy_estimator_reinforce = PolicyEstimatorReinforce(policy_estimator_name)
        self.episode = []

    def initialize_state(self, board):
        self.state = StateBuilder().build(board)
        self.episode = []

    def observe(self, session, board):
        next_state = StateBuilder().build(board)

        self.episode.append((self.state, self.action, 0.0))

        self.state = next_state

    def observe_finished(self, session, board):
        reward = -1.0
        if board.get_winner_value() == self.value:
            reward = 1.0

        self.episode.append((self.state, self.action, reward))

        # Update policy estimator
        for t, transition in enumerate(self.episode):
            total_return = sum(self.discount_factor ** i * t[2] for i, t in enumerate(self.episode[t:]))
            self.policy_estimator_reinforce.update(session, transition[0], total_return, transition[1])

    def predict_move(self, session):
        action_probs = self.policy_estimator_reinforce.predict(session, self.state)

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
