import pickle
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, AveragePooling2D, Flatten


def resnet152_model(img_rows, img_cols, color_type=1, num_classes=None):
	input_layer = Input(shape=(img_rows, img_cols, color_type), name='data')
	fc_layer = Flatten()(input_layer)
	fc_layer = Dense(num_classes, activation='softmax', name='fc')(fc_layer)
	weights_path = 'models/resnet152_weights_tf.h5'
	model = Model(input_layer, fc_layer)
	model.load_weights(weights_path, by_name=True)
	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

	return model


if __name__ == '__main__':
	num_classes = 3
	X = pickle.load(open("X_rgb.pickle", "rb"))
	y = pickle.load(open("y_rgb.pickle", "rb"))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
	img_rows, img_cols, channels = X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2]
	epochs = int(input('Epochs: '))
	batch_size = int(input('Batch Size: '))
	model = resnet152_model(img_rows, img_cols, channels, num_classes)
	model.fit(X_train, y_train,
		batch_size=batch_size,
		nb_epoch=epochs,
		shuffle=True,
		verbose=1,
		validation_data=(X_test, y_test),)

	# predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)
	# score = log_loss(y_test, predictions_valid)