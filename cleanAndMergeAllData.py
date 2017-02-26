from board import Board
import numpy as np
b = Board(4)

for i in range(200):
    print('start clean batch:', i)
    d = b.loadData('result' + str(i) + '.npy')
    cleaned = b.formatData(d)
    np.save('formatted' + str(i), cleaned)
