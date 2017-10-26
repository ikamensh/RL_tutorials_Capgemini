from keras.layers import Dense, Dropout, ELU, Input
from keras.models import Model
from keras.optimizers import adam
from keras.loss import binary_crossentropy

ad = adam(lr=1e-3)

# This returns a tensor
inputs = Input(shape=(input_size,))

hidden1 = Dense(4 * model_size) (inputs)
hidden1a = ELU() (hidden1)

hidden2 = Dense(4 * model_size)(hidden1a)
hidden2a = ELU()(hidden2)

hidden3 = Dense(4 * model_size)(hidden2a)
hidden3a = ELU()(hidden3)
hidden3d = Dropout(0.3)(hidden3a)

hidden4 = Dense(2 * model_size)(hidden3d)
hidden4a = ELU()(hidden4)
hidden4d = Dropout(0.3)(hidden4a)
print(hidden4d)

hidden5 = Dense(model_size)(hidden4d)
last_layer = ELU()(hidden5)
hidden = Dropout(0.3)(last_layer)

output = Dense(1, activation='sigmoid') (hidden)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=output)

#Replace output = LAYER_X to your layer to access model that shares weights with the original, but outputs values of desired layer!
features_model = Model(inputs=inputs, outputs=last_layer)

model.compile(optimizer=ad,
			  loss=binary_crossentropy)
			  
features_model.compile(optimizer=ad,
			  loss=binary_crossentropy)

