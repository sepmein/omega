import numpy as np
import numba
from numba import jit, int8

@jit
def searchPossibleStepsToEdge(state, positions, color):
    """
        search in 8 directions
        batch search algorithm
        positions is python list
        TODO: impact heavily on performance
    """
    directions = np.array(
            [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
    searched = np.empty((0, 3), np.int8)
    result = []
    for position in positions:
        p = position
        for j in range(8):
            differentColorAppeared = False
            d = directions[j]
            for i in range(8):
                nextSearchStep = get_next_search_step(p, i, d)
                # print(nextSearchStep)
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
                if x < 0 or y < 0 or (x > 7) or (y > 7):
                    break
                # 空白的格子
                elif state[x, y] == 0:
                    if i == 0:
                        break
                    elif differentColorAppeared == False:
                        break
                    else:
                        result.append(nextSearchStep)
                        break
                # 与当前要下的棋子不同色
                elif state[x, y] != color:
                    # 如果找到和当前要下的棋子一样的，那么就把当中所有棋子都变成这种颜色
                    differentColorAppeared = True
                    continue
                elif state[x, y] == color:
                    if differentColorAppeared == True:
                        break
                    else:
                        continue
    return np.array(result)

@jit
def get_next_search_step(p, i, d):
    return p + (i + 1) * d
