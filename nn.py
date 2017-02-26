import tensorflow as tf
import numpy as np

DATASET = np.load('formatted0.npy')
training_ratio = 0.8
training_num = int(training_ratio * DATASET.shape[0])
TRAINING_SET = DATASET[:training_num]
TEST_SET = DATASET[training_num:]

board_columns = tf.contrib.layers.real_valued_column(
    "board",    dimension=64,    dtype=tf.int8)
player_column = tf.contrib.layers.real_valued_column(
    "player",    dimension=1,    dtype=tf.int8)
position_column = tf.contrib.layers.real_valued_column(
    "position",    dimension=2,    dtype=tf.int8)

feature_columns = [board_columns, player_column, position_column]

estimator = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    ),
    model_dir="./model/"
)

# Input builders


def input_fn_train():  # returns x, y (where y represents label's class index).
    x = tf.constant(TRAINING_SET[:, :67])
    y = tf.constant(TRAINING_SET[:, 67:])
    return (x, y)
estimator.fit(input_fn=input_fn_train)


def input_fn_eval():  # returns x, y (where y represents label's class index).
    x = tf.constant(TEST_SET[:, :67])
    y = tf.constant(TEST_SET[:, 67:])
    return (x, y)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)  # returns predicted labels (i.e. label's class index).
