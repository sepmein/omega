# coding:utf-8
"""
class Board
"""
import numpy as np


class Board:
    """omega board
        rules:
            black is -1 / white is 1
            black(-1) take the first step
    """

    def __init__(self, n):
        board = np.zeros((2 * n, 2 * n), np.int8)
        board[n - 1, n - 1] = 1
        board[n - 1, n] = -1
        board[n, n - 1] = -1
        board[n, n] = 1
        self.board = board
        self.sequece = []
        self.color = -1
        self.step = 0
        self.n = 2 * n

    def searchToEdge(self, position):
        """search in 8 direction
        batch search algorithm"""
        results = []
        p = np.array(position)
        direction = np.array(
            [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
        n = self.n
        for j in range(8):
            differentColorAppeared = False
            d = direction[j]
            for i in range(n):
                nextSearchStep = p + (i + 1) * np.array(d)
                x, y = nextSearchStep
                # 超过边
                if x < 0 | y < 0 | x > self.n - 1 | y > self.n - 1:
                    break
                # 空白的格子
                elif self.board[x, y] == 0:
                    if i == 0:
                        break
                    elif differentColorAppeared == False:
                        break
                    else:
                        results.append(nextSearchStep)
                        break
                # 与当前要下的棋子不同色
                elif self.board[x, y] != self.color:
                    # 如果找到和当前要下的棋子一样的，那么就把当中所有棋子都变成这种颜色
                    # for u in range(i):
                    #    nextChangePosition = p + u * np.array(d)
                    #    self.board[nextChangePosition[0],
                    #               nextChangePosition[1]] = self.color
                    differentColorAppeared = True
                    continue
                elif self.board[x, y] == self.color:
                    if differentColorAppeared == True:
                        break
                    else:
                        continue
        return results

    def findPossibleStep(self):
        """find the possible step of next step"""
        print('find possible step')
        color = self.color

    def getAllSameColoredPieces(self):
        return np.argwhere(self.board == self.color)

    def play(self, position):
        """play at position"""
        if self.step == self.n**2:
            self.endGame()
        else:
            self.board[position[0], position[1]] = self.color
            self.searchToEdge(position)
            self.sequece.append([self.board, position])
            self.step = self.step + 1
            self.color = -1 * self.color

    def endGame(self):
        """end game"""
        print('end game')
        self.export()

    def export(self):
        """export board"""
        print('export')

    def printBoard(self):
        """print out board"""
        print(self.board)
