

from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
y_train = to_categorical(y_train, 10)

model = Sequential()

model.add(Dense(10, input_shape=(784,)))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy')
from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(x_train, y_train, verbose=1, callbacks=[tensorboard])
