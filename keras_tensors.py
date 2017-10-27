from keras.layers import Dense, ELU, Input
from keras.models import Model
import tensorflow as tf

input_size=5

# This returns a tensor
inputs = Input(shape=(input_size,))

hidden1 = Dense(128) (inputs)
hidden1a = ELU() (hidden1)

some_quantity = tf.Variable(tf.random_normal([128,128]), name = 'some_quantity')
average_of_quantity = tf.reduce_mean(some_quantity)
hidden_plus_avg = hidden1a+average_of_quantity

hidden2 = Dense(128) (hidden1a)
hidden2a = ELU() (hidden2)

output = Dense(1, activation='sigmoid') (hidden2a)

# !!! We define model as everything connecting two Tensors !!!
model = Model(inputs=inputs, outputs=output)


print(inputs)
print(output)
print(hidden_plus_avg)
model.summary()
