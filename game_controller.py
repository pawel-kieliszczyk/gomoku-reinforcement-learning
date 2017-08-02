from board import Board


class GameController(object):
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def play_game(self, session, player_one_starts=True, print_board=False):
        board = Board()

        if player_one_starts:
            self._play_game_if_player_one_starts(session, board, print_board)
        else:
            self._play_game_if_player_two_starts(session, board, print_board)

    def _play_game_if_player_one_starts(self, session, board, print_board):
        self.player1.initialize_state(board)
        x, y = self.player1.predict_move(session)
        board.put_value(x, y, 1.0)

        self.player2.initialize_state(board)
        x, y = self.player2.predict_move(session)
        board.put_value(x, y, -1.0)

        while True:
            self.player1.observe(session, board)
            x, y = self.player1.predict_move(session)
            board.put_value(x, y, 1.0)

            if board.game_finished():
                self.player1.observe_finished(session, board)
                self.player2.observe_finished(session, board)
                if print_board:
                    board.draw()
                    print(board.get_winner_value())
                break

            self.player2.observe(session, board)
            x, y = self.player2.predict_move(session)
            board.put_value(x, y, -1.0)

            if board.game_finished():
                self.player1.observe_finished(session, board)
                self.player2.observe_finished(session, board)
                if print_board:
                    board.draw()
                    print(board.get_winner_value())
                break

    def _play_game_if_player_two_starts(self, session, board, print_board):
        self.player2.initialize_state(board)
        x, y = self.player2.predict_move(session)
        board.put_value(x, y, -1.0)

        self.player1.initialize_state(board)
        x, y = self.player1.predict_move(session)
        board.put_value(x, y, 1.0)

        while True:
            self.player2.observe(session, board)
            x, y = self.player2.predict_move(session)
            board.put_value(x, y, -1.0)

            if board.game_finished():
                self.player1.observe_finished(session, board)
                self.player2.observe_finished(session, board)
                if print_board:
                    board.draw()
                    print(board.get_winner_value())
                break

            self.player1.observe(session, board)
            x, y = self.player1.predict_move(session)
            board.put_value(x, y, 1.0)

            if board.game_finished():
                self.player1.observe_finished(session, board)
                self.player2.observe_finished(session, board)
                if print_board:
                    board.draw()
                    print(board.get_winner_value())
                break
