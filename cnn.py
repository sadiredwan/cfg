import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten


if __name__ == '__main__':
	
	X_train = pickle.load(open("X.pickle", "rb"))
	y_train = pickle.load(open("y.pickle", "rb"))
	X_train =  X_train.reshape(-1, 128, 128, 1)

	model = Sequential()

	model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 1)))
	model.add(Activation('relu'))
	model.add(Conv2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(Conv2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))

	# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
	model.fit(X_train, y_train, batch_size=20, epochs=3, validation_split=0.1)