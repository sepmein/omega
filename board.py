# coding:utf-8
"""
class Board
"""
import numpy as np
import random


class Board:
    """omega board
        rules:
            black is -1 / white is 1
            black(-1) take the first step
            state 1 for ing / 0 for ended
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
        self.state = 1
        self.noMoreStepsCount = 0
        self.directions = np.array(
            [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

    def searchPossibleStepsToEdge(self, position):
        """search in 8 directions
        batch search algorithm
        position is python list
        """
        results = []
        p = np.array(position)
        n = self.n
        for j in range(8):
            differentColorAppeared = False
            d = self.directions[j]
            for i in range(n):
                nextSearchStep = p + (i + 1) * np.array(d)
                x, y = nextSearchStep
                # 超过边
                if x < 0 or y < 0 or (x > self.n - 1) or (y > self.n - 1):
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

    def flipToEdge(self, position):
        """flip to edge"""
        color = self.color
        p = position
        n = self.n
        for j in range(8):
            differentColorAppeared = False
            d = self.directions[j]
            for i in range(n):
                nextSearchStep = p + (i + 1) * np.array(d)
                x, y = nextSearchStep
                # 超过边
                if x < 0 or y < 0 or (x > self.n - 1) or (y > self.n - 1):
                    break
                # 空白的格子
                elif self.board[x, y] == 0:
                    break
                # 与当前要下的棋子不同色
                elif self.board[x, y] != self.color:
                    differentColorAppeared = True
                    continue
                # 与当前棋子相同的颜色
                elif self.board[x, y] == self.color:
                    # flip
                    if differentColorAppeared == True:
                        for o in range(i):
                            positionToBeChanged = p + (o + 1) * np.array(d)
                            x, y = positionToBeChanged
                            self.board[x, y] = self.color
                        break
                    else:
                        break

    def findAllPossibleSteps(self):
        """find the possible step of next step"""
        pieces = self.getAllSameColoredPieces()
        length = pieces.shape[0]
        result = []
        for i in range(length):
            result.extend(self.searchPossibleStepsToEdge(pieces[i]))
        result = np.array(result)
        # return unique rows
        if result.shape[0] == 0:
            return np.array([])
        else:
            return np.vstack({tuple(row) for row in result})

    def getAllSameColoredPieces(self):
        """get all same colored pieces"""
        return np.argwhere(self.board == self.color)

    def play(self, position):
        """play at position"""
        allPossibleSteps = self.findAllPossibleSteps()
        length = allPossibleSteps.shape[0]
        if type(position) == np.ndarray:
            p = position.tolist()
        else:
            p = position
        # 没有可下的棋
        if length == 0:
            self.noMoreStepsCount = self.noMoreStepsCount + 1
            # 如果连续两步没有可以下的棋，那么就结束比赛
            if self.noMoreStepsCount >= 2:
                self.endGame()
                return
            # 否则将棋盘控制权交给另一方，结束当前回合
            else:
                self.flipSide()
        else:
            # 要下的位置是否在可下的位置
            if p in allPossibleSteps.tolist():
                self.board[p[0], p[1]] = self.color
                self.flipToEdge(p)
                self.sequece.append([self.board, p])
                self.color = -1 * self.color
                # print(self.board)
            else:
                print('Error, not possible')

    def flipSide(self):
        """flip turn, eg: white to black / black to white"""
        self.color = -1 * self.color

    def endGame(self):
        """end game"""
        self.state = 0
        # winner = self.judgeWinner()
        # print(winner)
        # print('end game')

    def count(self):
        blackNumber = (self.board == -1).sum()
        whiteNubmer = (self.board == 1).sum()
        # blankNumber = self.n ** 2 - blackNumber - whiteNubmer
        # print('1/', whiteNubmer, '-1/', blackNumber)
        return (blackNumber, whiteNubmer)

    def judgeWinner(self):
        """judge winner
        count black number or white number
        """
        (blackNumber, whiteNubmer) = self.count()
        if blackNumber > whiteNubmer:
            return -1
        elif blackNumber < whiteNubmer:
            return 1
        else:
            return 0

    def reset(self):
        """reset board"""
        n = self.n / 2
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
        self.state = 1
        self.noMoreStepsCount = 0
        self.directions = np.array(
            [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

    def generateAndExportGame(self, times, fname):
        """
            generete full played game several times
            export four things:
            current board, player color, position to played, final result
            consider parameters steps, possible steps(均可从board衍生而出)
        """
        if self.n % 2 != 0 or self.n <= 6:
            print('board dimension too small or board dimension error')
            return
        export = []
        for i in range(times):
            if i % 1000 == 0:
                print('Cycle:', i)
            export.append([])
            u = 0
            while self.state == 1:
                possibleSteps = self.findAllPossibleSteps()
                if possibleSteps.shape[0] == 0:
                    self.play(possibleSteps)
                else:
                    export[i].append([])
                    randomIndex = int(random.random() * possibleSteps.shape[0])
                    position = possibleSteps[randomIndex]
                    export[i][u].append(np.copy(self.board))
                    export[i][u].append(self.color)
                    export[i][u].append(position)
                    self.play(position)
                    u = u + 1
            winner = self.judgeWinner()
            # print('w/', winner)
            for j in range(len(export[i])):
                export[i][j].append(winner)
            self.reset()
        self.export(export, fname)
        return export

    def export(self, e, fname):
        """export board"""
        np.save(fname, e)

    def printBoard(self):
        """print out board"""
        print(self.board)

    def loadData(self, fname):
	data = np.load(fname)
	return data

    def formatData(self, data):
        d = 0
        u = False
        for i in range(data.shape[0]):
            for j in range(len(data[i])):
                # the board
                # reshape to
                board = data[i][j][0].reshape(64)
                currentColor = [data[i][j][1]]
                position = data[i][j][2]
                result = [data[i][j][3]]
                row = np.concatenate((board, currentColor, position, result))
                if u == False:
                    d = row
                    u = True
                else:
                    d = np.vstack((d, row))
        return d
