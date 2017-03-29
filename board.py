# r: coding:utf-8
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
            reward for black win is 1
            reward for white win is -1
            reward for draw is 0
    """

    def __init__(self, n=4):
        self.board = np.zeros((2 * n, 2 * n), np.int8)
        self.board[n - 1, n - 1] = 1
        self.board[n - 1, n] = -1
        self.board[n, n - 1] = -1
        self.board[n, n] = 1
        self.sequece = []
        self.color = -1
        self.step = 0
        self.n = 2 * n
        self.ended = 1
        self.noMoreStepsCount = 0
        self.possibleSteps = np.array([[2, 3], [3, 2], [5, 4], [4, 5]])
        self.directions = np.array(
            [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

    def flipToEdge(self, position, state=None):
        """
            flip to edge
            using self.board or providing a specific state
            state should be an instance of numpy array
            position should be a 2*1 list
            return an numpy array
        """
        color = self.color
        p = position
        n = self.n
        if type(state) == np.ndarray:
            board = state
            (x, y) = position
            board[x][y] = color
        else:
            board = self.board
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
                elif board[x, y] == 0:
                    break
                # 与当前要下的棋子不同色
                elif board[x, y] != self.color:
                    differentColorAppeared = True
                    continue
                # 与当前棋子相同的颜色
                elif board[x, y] == self.color:
                    # flip
                    if differentColorAppeared == True:
                        for o in range(i):
                            positionToBeChanged = p + (o + 1) * np.array(d)
                            x, y = positionToBeChanged
                            board[x, y] = self.color
                        break
                    else:
                        break
        return board

    def getAllSameColoredPieces(self, state=None, color=None):
        """get all same colored pieces"""
        if type(state) == np.ndarray:
            board = state
        else:
            board = self.board
        return np.argwhere(board == color)

    def findAllPossibleSteps(self, state=None, color=None):
        """
            Find all possible steps for next step
            Return a numpy array as result
        """
        # if type(state) == np.ndarry:
        #     s = state
        # else:
        #     s = self.board
        pieces = self.getAllSameColoredPieces(state, color)
        results = self.searchPossibleStepsToEdge(pieces, state, color)
        # return unique rows
        if results.shape[0] == 0:
            return results
        else:
            return np.vstack({tuple(row) for row in results})

    def searchPossibleStepsToEdge(self, positions, state=None,  color=None):
        """
            search in 8 directions
            batch search algorithm
            positions is python list
        """
        if color:
            c = color
        else:
            c = self.color
        if type(state) == np.ndarray:
            b = state
        else:
            b = self.board
        n = self.n
        searched = np.empty((0, 3), np.int8)
        result = []
        for position in positions:
            p = np.array(position, np.int8)
            for j in range(n):
                differentColorAppeared = False
                d = self.directions[j]
                for i in range(n):
                    nextSearchStep = p + (i + 1) * np.array(d, np.int8)
                    x, y = nextSearchStep
                    # if too many steps
                    # start reduce function
                    if position.shape[0] >= 5:
                        if np.any(np.all(searched == [x, y, j], axis=1)):
                            # searched before, don't search any more
                            break
                        else:
                            # never searched before, push result to searched
                            # array
                            searched = np.append(searched, [[x, y, j]], axis=0)
                    # 超过边
                    if x < 0 or y < 0 or (x > self.n - 1) or (y > self.n - 1):
                        break
                    # 空白的格子
                    elif b[x, y] == 0:
                        if i == 0:
                            break
                        elif differentColorAppeared == False:
                            break
                        else:
                            result.append(nextSearchStep)
                            break
                    # 与当前要下的棋子不同色
                    elif b[x, y] != c:
                        # 如果找到和当前要下的棋子一样的，那么就把当中所有棋子都变成这种颜色
                        differentColorAppeared = True
                        continue
                    elif b[x, y] == c:
                        if differentColorAppeared == True:
                            break
                        else:
                            continue
        return np.array(result)

    def play(self, position=None):
        """play at position"""
        # if (self.possibleSteps, np.ndarray) == True:
        #     allPossibleSteps = self.possibleSteps
        # else:
        #     # never calculated before
        self.possibleSteps = self.findAllPossibleSteps(color=self.color)
        # print('\nBefore played')
        # print('Current possible steps:')
        # print self.possibleSteps
        # print('Position to be played: ', position)
        # print('Color to be played: ', self.color)
        # print('==================')
        # length = allPossibleSteps.shape[0]
        if position is None:
            self.flipSide()
            self.possibleSteps = self.findAllPossibleSteps(color=self.color)
            return
        if type(position) == np.ndarray:
            p = position.tolist()
        else:
            p = position
        if p in self.possibleSteps.tolist():
            self.board[p[0], p[1]] = self.color
            self.flipToEdge(p)
            self.step = self.step + 1
            # update possible steps
            # self.sequece.append([self.board, p])
            # print('\n BOARD:')
            # self.printBoard()
            if self.step < 52:
                ended = False
            else:
                winner, ended = self.judge_terminal()
            if ended:
                self.endGame()
            else:
                self.flipSide()
                self.possibleSteps = self.findAllPossibleSteps(
                    color=self.color)
                # reset no_more_steps_count to 0 if one has played a step
                # self.noMoreStepsCount = 0
        else:
            print('Error, not possible')

    def flipSide(self):
        """flip turn, eg: white to black / black to white"""
        self.color = -1 * self.color

    def endGame(self):
        """end game"""
        # print('\n')
        # print('++++++++++++++GAME ENDED+++++++++++++++')
        # print('\n')
        self.ended = 0

    def count(self, state=None):
        if type(state) == np.ndarray:
            s = state
        else:
            s = self.board
        blackNumber = (s == -1).sum()
        whiteNubmer = (s == 1).sum()
        # blankNumber = self.n ** 2 - blackNumber - whiteNubmer
        # print('1/', whiteNubmer, '-1/', blackNumber)
        return (blackNumber, whiteNubmer)

    def judgeWinner(self, state=None):
        """judge winner
        count black number or white number
        """
        if type(state) == np.ndarray:
            s = state
        else:
            s = self.board
        (blackNumber, whiteNubmer) = self.count(s)
        if blackNumber > whiteNubmer:
            return -1
        elif blackNumber < whiteNubmer:
            return 1
        else:
            return 0

    def reset(self):
        """reset board"""
        self = self.__init__()

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
            while self.ended == 1:
                possibleSteps = self.findAllPossibleSteps(color=self.color)
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

    def generateGameUsingModelOnce(self, model):
        if self.n % 2 != 0 or self.n <= 6:
            print('board dimension too small or board dimension error')
            return
        export = []
        u = 0
        while self.ended == 1:
            possibleSteps = self.findAllPossibleSteps(color=self.color)
            if possibleSteps.shape[0] == 0:
                self.play(possibleSteps)
            else:
                board = np.copy(self.board).reshape(64)
                color = [self.color]
                nextStep = self.pickBestMoveWithRandomness(model, 0.5)
                export.append(np.concatenate((board, color, nextStep)))
                self.play(nextStep)
                u = u + 1
        winner = self.judgeWinner()
        # print('w/', winner)
        for j in range(len(export)):
            export[j] = np.concatenate((export[j], [winner]))
        self.reset()
        export = np.array(export)
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

    def predictWinningProbabilityForNextStep(self, model):
        """using keras"""
        allPossibleSteps = self.findAllPossibleSteps(color=self.color)
        data = []
        for i in range(allPossibleSteps.shape[0]):
            boardReshape = self.board.reshape(64)
            currentColor = [self.color]
            step = allPossibleSteps[i]
            data.append(np.concatenate((boardReshape, currentColor, step)))
        datanp = np.vstack(step for step in data)
        predictions = model.predict(datanp, batch_size=1, verbose=0)
        return (allPossibleSteps, predictions)

    def pickBestMove(self, model):
        (steps, predictions) = self.predictWinningProbabilityForNextStep(model)
        if self.color == -1:
            return steps[predictions[:, 2:].argmax()]
        else:
            return steps[predictions[:, 1:2].argmax()]

    def pickBestMoveWithRandomness(self, model, randomRate):
        (steps, predictions) = self.predictWinningProbabilityForNextStep(model)

        def softmax(w, t=1.0):
            e = np.exp(np.array(w) / t)
            dist = e / np.sum(e)
            return dist

        if self.color == -1:
            (d1, d2) = predictions[:, 2:].shape
            softmaxWithRandomness = softmax(predictions[:, 2:]) * \
                (randomRate * np.random.rand(d1, d2))
            return steps[softmaxWithRandomness.argmax()]

        else:
            (d1, d2) = predictions[:, 1:2].shape
            softmaxWithRandomness = softmax(predictions[:, 1:2]) * \
                (randomRate * np.random.rand(d1, d2))
            return steps[softmaxWithRandomness.argmax()]

    def get_next_states(self, state=None):
        """
            Get all next states
        """
        if self.ended == 0:
            return None
        # next_color = self.color * -1
        n = self.possibleSteps.shape[0]
        next_states = []
        for i in range(n):
            x, y = self.possibleSteps[i]
            # TODO generate next state
            next_states.append(self.flipToEdge(
                position=[x, y], state=np.copy(self.board)))
        return np.array(next_states)

    def get_reward(self, next_states):
        """
            check one of the board, if game has ended
            define black as 1, white as -1
            if black wins return reward 1
            else return reward 0
        """
        r = []
        if self.step <= 50:
            return np.zeros(next_states.shape[0], np.int8)
        else:
            for state in next_states:
                winner, ended = self.judge_terminal(state)
                if ended:
                    r.append(winner)
                else:
                    r.append(0)
            return np.array(r)

    def generate_and_store_game_by_policy(self, policy):
        """generate a game with policy"""
        while self.ended != True:
            actions = self.findAllPossibleSteps(color=self.color)
            next_states = self.get_next_states()
            action = policy.pai(self.board)
            # next_state = self.play(action)
            # next_value = db.find_value(next_state)
            print self.board

    def apply_policy(self, pai, method):
        """apply policy to get the next action"""
        actions = self.possibleSteps
        next_states = self.get_next_states()
        rewards = self.get_reward(next_states)
        # print '=================='
        # print 'apply policy called'
        # print('actions', actions)
        action, optimal_value, move = pai(
            actions, next_states, rewards, method)
        # print('action: ', action)
        return action, optimal_value, move

    def judge_terminal(self, state=None):
        # test next state is the final state
        # if no more step for the next color
        # and no more step for the next next color
        # then the next move is final move
        if type(state) == np.ndarray:
            next_state = state
            # next move
            next_color = self.color * -1
            next_possible_steps = self.findAllPossibleSteps(
                next_state, next_color)
            if next_possible_steps.shape[0] != 0:
                # if next possible steps count > 0
                # game not ended
                return (0, False)
            else:
                # flip color, reexamine
                next_next_color = next_color * -1
                next_next_possible_steps = self.findAllPossibleSteps(
                    next_state, next_next_color)
                if next_next_possible_steps.shape[0] != 0:
                    return (0, False)
                else:
                    winner = self.judgeWinner(next_state)
                    return (winner, True)
        else:
            next_state = self.board
            next_color = self.color
            next_possible_steps = self.findAllPossibleSteps(
                next_state, next_color)
            if next_possible_steps.shape[0] != 0:
                # if next possible steps count > 0
                # game not ended
                return (0, False)
            else:
                # flip color, reexamine
                next_next_color = next_color * -1
                next_next_possible_steps = self.findAllPossibleSteps(
                    next_state, next_next_color)
                if next_next_possible_steps.shape[0] != 0:
                    return (0, False)
                else:
                    winner = self.judgeWinner(next_state)
                    return (winner, True)
