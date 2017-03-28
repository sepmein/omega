# coding:utf-8
"""test module"""
from board import Board
import random

# use the most simple number for test
b = Board()
print(b.board)
# flip color to next move
# searchResult = b.searchPossibleStepsToEdge([3, 4])
# print(searchResult)

# b.play([3,2])
# b.findAllPossibleSteps()
# b.board
# b.play([2,2])
# b.findAllPossibleSteps()
# b.board
# b.play([2,3])
# b.findAllPossibleSteps()
# b.board

# b.color
# b.findAllPossibleSteps()
# b.play([3,2])
# b.color
# b.findAllPossibleSteps()

# b.flipToEdge([7,7])

# # randomly start a game
# while b.state == 1:
# 	possibleSteps = b.findAllPossibleSteps()
# 	if possibleSteps.shape[0] == 0:
# 		b.play(possibleSteps)
# 	else:
# 		randomIndex = int(random.random() * possibleSteps.shape[0])
# 		b.play(possibleSteps[randomIndex])
b.get_next_states()