from board import Board
from keras.models import load_model
model = load_model('dnn_c2_v1.h5')
b = Board(4)
bestMove = b.pickBestMoveWithRandomness(model)
print(bestMove)
