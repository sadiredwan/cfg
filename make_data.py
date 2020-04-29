import pickle
import os, cv2
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def normalize(x):
	min_val = np.min(x)
	max_val = np.max(x)
	x = (x-min_val)/(max_val-min_val)
	return x


if __name__ == '__main__':
	PATH = os.getcwd()
	data_path = PATH + '/data'
	data_dir_list = os.listdir(data_path)

	num_classes = 3
	labels_name = {'class1':0,'class2':1,'class3':2}

	img_data_list = []
	labels_list = []

	for dataset in data_dir_list:
		img_list = os.listdir(data_path + '/' + dataset)
		print('Loading the images of dataset-'+'{}\n'.format(dataset))
		label = labels_name[dataset]
		for img in img_list:
			input_img = cv2.imread(data_path + '/' + dataset + '/' + img )
			input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_resize = cv2.resize(input_img, (128, 128))
			img_data_list.append(input_img_resize)
			labels_list.append(label)

	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	# img_data /= 255
	normalize(img_data)
	print(img_data.shape)

	labels = np.array(labels_list)
	print(np.unique(labels, return_counts=True))
	y = np_utils.to_categorical(labels, num_classes)
	X, y = shuffle(img_data, y, random_state=2)

	# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2)

	pickle_out = open("X.pickle", "wb")
	pickle.dump(X, pickle_out)
	pickle_out.close()

	pickle_out = open("y.pickle", "wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()