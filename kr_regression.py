import numpy as np

data = []
for i in range(200):
    data.append(np.load('formatted' + str(i) + '.npy'))
DATASET = np.vstack(stack for stack in data)
training_ratio = 0.7
training_num = int(training_ratio * DATASET.shape[0])
TRAINING_SET = DATASET[:training_num]
TEST_SET = DATASET[training_num:]

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical


model.add(Dense(100, input_dim=67))
model.add(Activation('relu'))
model.add(Dense(output_dim=100))
model.add(Activation('relu'))
# model.add(Dense(output_dim=256))
# model.add(Activation('relu'))
model.add(Dense(output_dim=1))
model.add(Activation('softmax'))

model.compile(loss='mse',
              optimizer='rmsprop', metrics=['accuracy'])

X_train = TRAINING_SET[:, :67]
Y_train = TRAINING_SET[:, 67:]
# Y_train = to_categorical(Y_train_int)

X_test = TEST_SET[:, :67]
Y_test = TEST_SET[:, 67:]
# Y_test = to_categorical(Y_test_int)
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics)

config = model.get_config()
print(config)
print(model.get_weights())
print(model.to_json())
model.save('kr_model.h5')
model.save_weights('kr_model_weights.h5')
