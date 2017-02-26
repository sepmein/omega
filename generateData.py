# coding:utf-8
from board import Board

"""每个批次10000个"""
for i in range(200):
    print('Batch:', i)
    b = Board(4)
    b.generateAndExportGame(100, 'result' + str(i))
