import pickle
import numpy as np
from sklearn.model_selection import train_test_split


class Conv:
	def __init__(self, num_filters, filter_size):
		self.num_filters = num_filters
		self.filter_size = filter_size
		self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size*filter_size)

	def region(self, image):
		height, width = image.shape
		self.image = image
		for i in range(height-self.filter_size+1):
			for j in range(width-self.filter_size+1):
				image_patch = image[i: (i+self.filter_size), j: (j+self.filter_size)]
				yield image_patch, i, j

	def forward_propagation(self, image):
		height, width = image.shape
		conv_out = np.zeros((height-self.filter_size+1, width-self.filter_size+1, self.num_filters))
		for image_patch, i, j in self.region(image):
			conv_out[i, j] = np.sum(image_patch*self.conv_filter, axis=(1, 2))
		return conv_out

	def back_propagation(self, dL_dout, learning_rate):
		dL_dF = np.zeros(self.conv_filter.shape)
		for image_patch, i, j in self.region(self.image):
			for k in range(self.num_filters):
				dL_dF[k] += image_patch * dL_dout[i, j, k]
		self.conv_filter -= learning_rate * dL_dF
		return dL_dF


class MaxPool:
	def __init__(self, filter_size):
		self.filter_size = filter_size

	def region(self, image):
		new_height = image.shape[0] // self.filter_size
		new_width = image.shape[1] // self.filter_size
		self.image = image
		for i in range(new_height):
			for j in range(new_width):
				image_patch = image[(i*self.filter_size): (i*self.filter_size+self.filter_size),
									(j*self.filter_size): (j*self.filter_size+self.filter_size)]
				yield image_patch, i, j

	def forward_propagation(self, image):
		height, width, num_filters = image.shape
		output = np.zeros((height//self.filter_size, width//self.filter_size, num_filters))
		for image_patch, i, j in self.region(image):
			output[i, j] = np.amax(image_patch, axis=(0, 1))
		return output

	def back_propagation(self, dL_dout):
		dL_dmax = np.zeros(self.image.shape)
		for image_patch, i, j in self.region(self.image):
			height, width, num_filters = image_patch.shape
			max_val = np.amax(image_patch, axis=(0, 1))
			for _i in range(height):
				for _j in range(width):
					for _k in range(num_filters):
						dL_dmax[i*self.filter_size+_i, j*self.filter_size+_j, _k] = dL_dout[i, j, _k]
		return dL_dmax


class Softmax:
	def __init__(self, input_node, softmax_node):
		self.weight = np.random.randn(input_node, softmax_node)/input_node
		self.bias = np.zeros(softmax_node)

	def forward_propagation(self, image):
		self.original_image_shape = image.shape
		image_modified = image.flatten()
		self.modified_input = image_modified
		output_val = np.dot(image_modified, self.weight) + self.bias
		self.out = output_val
		exp_out = np.exp(output_val)
		return exp_out/np.sum(exp_out, axis=0)

	def back_propagation(self, dL_dout, learning_rate):
		for i, grad in enumerate(dL_dout):
			if grad == 0:
				continue
			transform = np.exp(self.out)
			total = np.sum(transform)
			dy_dz = -transform[i] * transform / (total**2)
			dy_dz[i] = transform[i] * (total-transform[i]) / (total**2)
			dz_dw = self.modified_input
			dz_db = 1
			dz_din = self.weight
			dL_dz = grad * dy_dz
			dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
			dL_db = dL_dz * dz_db
			dL_din = dz_din @ dL_dz
		self.weight -= learning_rate * dL_dw
		self.bias -= learning_rate * dL_db
		return dL_din.reshape(self.original_image_shape)



X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb")) #one hot for tensorflow
y = np.argmax(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
conv_layer1 = Conv(8, 3)
maxpool_layer1 = MaxPool(2)
softmax_layer1 = Softmax(126//2 * 126//2 * 8, 3)


def cnn_forward(image, label):
	out_p = conv_layer1.forward_propagation((image/255) - 0.5)
	out_p = maxpool_layer1.forward_propagation(out_p)
	out_p = softmax_layer1.forward_propagation(out_p)
	cross_entropy_loss = -np.log(out_p[label])
	accuracy_eval = 1 if np.argmax(out_p) == label else 0
	return out_p, cross_entropy_loss, accuracy_eval


def cnn_train(image, label, learning_rate=0.005):
	out, loss, acc = cnn_forward(image, label)
	gradient = np.zeros(3)
	gradient[label] = -1/out[label]
	gradient_back = softmax_layer1.back_propagation(gradient, learning_rate)
	gradient_back = maxpool_layer1.back_propagation(gradient_back)
	gradient_back = conv_layer1.back_propagation(gradient_back, learning_rate)
	return loss, acc


epochs = int(input('Epochs: '))
batch_size = int(input('Batch Size: '))
for epoch in range(epochs):
	loss = 0
	num_correct = 0
	j = 1
	for i, (image, label) in enumerate(zip(X_train, y_train)):
		if i%batch_size == 0:
			print('%d steps of %d steps: AvgLoss: %.3f, Accuracy: %d%%' %(i+1, batch_size*j, loss/batch_size, num_correct))
			loss = 0 #sgd
			num_correct = 0
			j += 1
		dL, accuracy = cnn_train(image, label)
		loss += dL
		num_correct += accuracy
#testing
print('Testing: ')
loss = 0
num_correct = 0
for image, label in enumerate(zip(X_test, y_test)):
	_, dL, accuracy = cnn_forward(image, label)
	loss += dL
	num_correct += accuracy
num_tests = len(X_test)
print('Test Loss: ', loss/num_tests)
print('Test Accuracy: ', num_correct/num_tests)