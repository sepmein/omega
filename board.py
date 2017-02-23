#coding:utf-8
import numpy as np

class Board:
    def __init__(self,n):
        self.board = np.zeros((n,n),np.int8)
        self.sequece = []
        self.nextMove = 1
        self.step = 0
        self.n = n

    def searchToEdge(self,position):
        """search in 8 direction"""
        """batch search algorithm"""
        direction = np.array([[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[-1,0],[-1,-1]])
        n = self.n
        for i in range(n):
            for j in range(8):
                d = direction[j]
                nextSearchStep = position + i * np.array(d)
                print(nextSearchStep)
                if nextSearchStep[0] < 0 | nextSearchStep[0] > n | nextSearchStep[1] < 0 | nextSearchStep[1] >n:
                    return
                elif self.board[nextSearchStep[0], nextSearchStep[1]] == 0:
                    return
                elif self.board[nextSearchStep[0], nextSearchStep[1]] == self.nextMove:
                    """如果找到和当前要下的棋子一样的，那么就把当中所有棋子都变成这种颜色"""
                    for u in range(i):
                        nextChangePosition = position + u * np.array(d)
                        self.board[nextChangePosition[0], nextChangePosition[1]] = self.nextMove

    def play(self,position):
        if self.step == self.n**2:
            self.endGame()
        else:
            self.board[position[0],position[1]] = self.nextMove
            self.searchToEdge(position)
            self.sequece.append([self.board, position])
            self.step = self.step + 1
            self.nextMove = -1 * self.nextMove

    def nextStep(self):
        print('next step')

    def endGame(self):
        print('end game')
        self.export()

    def export(self):
        print('export')

    def printBoard(self):
        print('board')

