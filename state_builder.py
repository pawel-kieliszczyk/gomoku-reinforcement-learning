import numpy as np


class StateBuilder(object):
    def build(self, board):
    # def build(self, board, value):
        # channels:
        # 1. black stones
        # 2. white stones
        # 3. valid moves
        state = np.zeros([board.get_size(), board.get_size(), 3])
        self._build_state_black_stones(board, state, 0)
        self._build_state_white_stones(board, state, 1)
        self._build_state_valid_moves(board, state, 2)
        # self._build_state_next_move(board, value, state, 3)

        return state

    def _build_state_black_stones(self, board, state, channel_number):
        for x in range(board.get_size()):
            for y in range(board.get_size()):
                if board.get_value(x, y) == 1.0:
                    state[x, y, channel_number] = 1.0

    def _build_state_white_stones(self, board, state, channel_number):
        for x in range(board.get_size()):
            for y in range(board.get_size()):
                if board.get_value(x, y) == -1.0:
                    state[x, y, channel_number] = 1.0

    def _build_state_valid_moves(self, board, state, channel_number):
        for x in range(board.get_size()):
            for y in range(board.get_size()):
                if board.get_value(x, y) == 0.0:
                    state[x, y, channel_number] = 1.0

    # def _build_state_next_move(self, board, value, state, channel_number):
    #     for x in range(board.get_size()):
    #         for y in range(board.get_size()):
    #             if value == 1.0:
    #                 state[x, y, channel_number] = 1.0
    #             else:
    #                 state[x, y, channel_number] = 0.0
