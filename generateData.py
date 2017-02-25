from board import Board

"""每个批次10000个"""
for i in range(20):
    b = Board(4)
    b.generateAndExportGame(10000, 'result' + str(i)
