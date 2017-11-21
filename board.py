import numpy as np


class Board(object):
    def __init__(self):
        self.size = 15
        self.data = np.zeros([self.size, self.size])

        self.moves_left = self.size ** 2
        self.finished = False
        self.winner_value = 0.0 # not defined yet

    def get_size(self):
        return self.size

    def put_value(self, x, y, value):
        self.data[x, y] = value
        self.moves_left -= 1
        self._check_if_finished_after_move(x, y, value)

    def get_value(self, x, y):
        return self.data[x, y]

    def game_finished(self):
        return self.finished

    def get_winner_value(self):
        return self.winner_value

    def draw(self):
        print(self.data)
        # b = np.chararray([self.size, self.size])
        # for x in range(self.size):
        #     for y in range(self.size):
        #         if self.data[x, y] == 0.0:
        #             b[x, y] = '_'
        #         elif self.data[x, y] == 1.0:
        #             b[x, y] = 'x'
        #         else:
        #             b[x, y] = 'o'
        #
        # print(b)

    def _check_if_finished_after_move(self, x, y, value):
        if self.moves_left == 0:
            self.finished = True

        self._check_if_finished_vertically(x, y, value)
        self._check_if_finished_horizontally(x, y, value)
        self._check_if_finished_diagonals(x, y, value)

    def _check_if_finished_vertically(self, x, y, value):
        self._check_if_finished(x, y, value, 0, 1)

    def _check_if_finished_horizontally(self, x, y, value):
        self._check_if_finished(x, y, value, 1, 0)

    def _check_if_finished_diagonals(self, x, y, value):
        self._check_if_finished(x, y, value, 1, 1)
        self._check_if_finished(x, y, value, 1, -1)

    def _check_if_finished(self, x, y, value, xOffset, yOffset):
        sameStonesLeft = 0
        for i in range(1, 6):
            xx = x - i * xOffset
            yy = y - i * yOffset

            if xx < 0 or xx >= self.size:
                break

            if yy < 0 or yy >= self.size:
                break

            if self.data[xx, yy] == 0.0:
                break

            if self.data[xx, yy] == value:
                sameStonesLeft += 1
            else:
                break

        sameStonesRight = 0
        for i in range(1, 5):
            xx = x + i * xOffset
            yy = y + i * yOffset

            if xx < 0 or xx >= self.size:
                break

            if yy < 0 or yy >= self.size:
                break

            if self.data[xx, yy] == 0.0:
                break

            if self.data[xx, yy] == value:
                sameStonesRight += 1
            else:
                break

        sameStones = sameStonesLeft + sameStonesRight + 1
        if sameStones >= 5:
            self.finished = True
            self.winner_value = value
