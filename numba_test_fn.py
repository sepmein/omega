import numpy as np
import numba
from board import Board
from numba import jit
omega = Board()


@numba.jit
def blabla(j, p, d, position, searched, board, result, differentColorAppeared, n, color):
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
        if x < 0 or y < 0 or (x > n - 1) or (y > n - 1):
            break
        # 空白的格子
        elif board[x, y] == 0:
            if i == 0:
                break
            elif differentColorAppeared == False:
                break
            else:
                result.append(nextSearchStep)
                break
        # 与当前要下的棋子不同色
        elif board[x, y] != color:
            # 如果找到和当前要下的棋子一样的，那么就把当中所有棋子都变成这种颜色
            differentColorAppeared = True
            continue
        elif board[x, y] == color:
            if differentColorAppeared == True:
                break
            else:
                continue


@numba.jit
def searchPossibleStepsToEdge(board, n, positions, color, directions):
    """
        search in 8 directions
        batch search algorithm
        positions is python list
        TODO: impact heavily on performance
    """
    searched = np.empty((0, 3), np.int8)
    result = []
    for position in positions:
        p = np.array(position, np.int8)
        for j in range(n):
            differentColorAppeared = False
            d = directions[j]
            blabla(j, p, d, position, searched,
                   board, result, differentColorAppeared, n, color)
    return np.array(result)

searchPossibleStepsToEdge(omega.board, omega.n, np.array(
    [[3, 4]]), omega.color, omega.directions)
