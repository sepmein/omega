from board import Board
from keras.models import load_model
import numpy as np
from keras.utils.np_utils import to_categorical

data = []
for i in range(200):
    data.append(np.load('formatted' + str(i) + '.npy'))
DATASET = np.vstack(stack for stack in data)
DATASET[:, 67][DATASET[:, 67] == 0] = 2
training_ratio = 0.9
training_num = int(training_ratio * DATASET.shape[0])
TRAINING_SET = DATASET[:training_num]
TEST_SET = DATASET[training_num:]
X_train = TRAINING_SET[:, :67]
Y_train_int = TRAINING_SET[:, 67:]
Y_train = to_categorical(Y_train_int)

X_test = TEST_SET[:, :67]
Y_test_int = TEST_SET[:, 67:]
Y_test = to_categorical(Y_test_int)
model = load_model('dnn_c2_v1.h5')
b = Board(4)

(x, y) = b.generateGameUsingModelOnce(model)
y_catgorical = np.zeros((y.shape[0], 3))
for i in y:
    if y[i] == 0:
        y_catgorical[i] = [1., 0, 0]
    elif y[i] == 1:
        y_catgorical[i] = [0, 1., 0]
    elif y[i] == -1:
        y_catgorical[i] = [0, 0, 1.]
model.train_on_batch(x, y_catgorical)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics)
(x1, y1) = b.generateGameUsingModelOnce(model)
y1_catgorical = np.zeros((y1.shape[0], 3))
for i in y1:
    if y1[i] == 0:
        y1_catgorical[i] = [1., 0, 0]
    elif y1[i] == 1:
        y1_catgorical[i] = [0, 1., 0]
    elif y1[i] == -1:
        y1_catgorical[i] = [0, 0, 1.]
model.train_on_batch(x, y1_catgorical)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics)
