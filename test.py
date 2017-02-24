#coding:utf-8
"""test module"""
from board import Board

#use the most simple number for test
b = Board(4)
print(b.board)
#flip color to next move
b.searchToEdge([3,4])
