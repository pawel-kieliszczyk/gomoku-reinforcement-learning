from board import Board


class GameController(object):
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def play_game(self, session, print_board=False):
        board = Board()

        self.player1.initialize_state(board)
        x, y = self.player1.predict_move(session)
        board.put_value(x, y, 1.0)

        self.player2.initialize_state(board)
        x, y = self.player2.predict_move(session)
        board.put_value(x, y, -1.0)

        while True:
            self.player1.observe(session, board)
            x, y = self.player1.predict_move(session)
            # print("Putting 1:", x, y, board.data)
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
            # print("Putting -1:", x, y, board.data)
            board.put_value(x, y, -1.0)

            if board.game_finished():
                self.player1.observe_finished(session, board)
                self.player2.observe_finished(session, board)
                if print_board:
                    board.draw()
                    print(board.get_winner_value())
                break
