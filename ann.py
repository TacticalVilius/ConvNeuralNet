import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum, auto

ActivationFunction = namedtuple("ActivationFunction", "f d_f")
sigmoid = ActivationFunction(f = (lambda x : 1 / (1 +  np.exp(-x))), d_f = (lambda y : (1 - y) * y))
tanh = ActivationFunction(f = (lambda x : np.tanh(x)), d_f = (lambda y : 1 - y**2))
relu = ActivationFunction(f = (lambda x : np.maximum(0, x)), d_f = (lambda y : (y > 0).astype(int)))

class Regularization(Enum):
	L2 = auto()
	L1 = auto()
	ELASTIC_NET = auto()

class ANN:

	def __init__(self, data_dim, layer_num = 2, activation_func = relu, reg_type = Regularization.L2):
		self.layers = []
		input_dim = data_dim
		layer_size = 100
		classes_num = 3
		dropout_prob = 0.5 # Probablility to drop a neuron
		for i in range(layer_num - 1):
			self.layers.append(Layer(layer_size, input_dim, activation_func.f, activation_func.d_f, dropout_prob))
			input_dim = layer_size
		output_layer = Layer(classes_num, input_dim, lambda x: x, lambda x: 1, 0.0)
		self.layers.append(output_layer)

		self.reg_type = reg_type
		self.reg_const = 0.001
		self.step_size = 0.1

	def evaluate(self, X, labels):
		output = self.forward(X, True)
		predicted_classes = np.argmax(output, axis=1)
		print("Training accuracy: %.2f" % (np.mean(predicted_classes == labels)))

	def learn_from(self, X, labels, debug=False):
		output = self.forward(X, debug)
		self.backward(output, labels, debug)

	def forward(self, X, debug):
		prev_layer_output = X
		if debug:
			num_neurons = 0
			num_dead_neurons = 0
		for i, layer in enumerate(self.layers):
			prev_layer_output = layer.forward(prev_layer_output)
			if debug and i != len(self.layers) - 1:
				num_neurons += prev_layer_output.shape[1]
				num_dead_neurons += np.sum(np.all(prev_layer_output == 0, axis=0))
		if debug:
			print("Dead neuron percentage: " + str(num_dead_neurons / num_neurons) + " (%d / %d)" % (num_dead_neurons, num_neurons))
		return prev_layer_output

	def backward(self, output, labels, debug):
		# Gradient of the loss function based on the final output of the network
		d_next_layer_output = self.loss_func_grad(output, labels, debug)
		for i, layer in reversed(list(enumerate(self.layers))):
			# Gradient of the loss function based on the input to this layer
			d_next_layer_output = layer.backward(d_next_layer_output, self.reg_type, self.reg_const)

		for layer in self.layers:
			layer.param_update(self.step_size)

	'''
	output: array with outputs from output layer. Shape (N, K) where N is # of data points and K is # of classes.
	labels: array with the correct label for each data point. Shape (N,).
	returns array with gradients of the loss function based on the score function (output of the network) for each data point. Shape (N, K).
	'''
	def loss_func_grad(self, output, labels, debug):
		data_points_num = output.shape[0]
		classes_num = output.shape[1]

		mask = np.zeros_like(output, dtype=bool)
		mask[np.arange(data_points_num), labels] = True

		correct_class_scores = output[np.arange(data_points_num), labels]
		correct_class_scores.shape = (data_points_num, 1)
		output = output - correct_class_scores + 1
		output[output < 0] = 0
		
		if debug:
			loss = (np.sum(output[~mask]) / data_points_num) + self.get_reg_loss(self.reg_type)
			print("Loss: " + str(loss))

		output[output > 0] = 1

		wrong_classes_output = output[~mask]
		wrong_classes_output.shape = (data_points_num, classes_num - 1)
		num_wrong_results = np.sum(wrong_classes_output, axis=1)
		output[mask] = -num_wrong_results

		output = output / data_points_num
		return output

	def get_reg_loss(self, reg_type):
		if reg_type == Regularization.L2:
			return self.reg_const * sum(list(map((lambda layer: np.sum(layer.W * layer.W)), self.layers))) / 2
		elif reg_type == Regularization.L1:
			return self.reg_const * sum(list(map((lambda layer: np.sum(np.abs(layer.W))), self.layers)))
		elif reg_type == Regularization.ELASTIC_NET:
			# Using the same constant for the L1 and L2 parts. These could be different if desired.
			return self.get_reg_loss(Regularization.L1) + self.get_reg_loss(Regularization.L2)

class Layer:

	def __init__(self, layer_size, input_dim, activation_func, d_activation_func, dropout_prob):
		self.size = layer_size
		self.activation_func = activation_func
		self.d_activation_func = d_activation_func
		self.dropout_prob = dropout_prob

		self.W = self.get_initial_weights(layer_size, input_dim)
		self.b = self.get_initial_bias(layer_size)
		self.d_W = None
		self.d_b = None
		self.input = None
		self.output = None

	def get_initial_weights(self, layer_size, input_dim):
		# We want approximately same distribution of initial neuron outputs across the network
		if self.activation_func != relu.f:
			# This can be achieved by scaling the weights by sqrt of the neuron's fan-in
			return np.random.randn(input_dim, layer_size) / np.sqrt(input_dim)
		else:
			# Recommended for ReLU neurons
			return np.random.randn(input_dim, layer_size) * np.sqrt(2 / input_dim)

	def get_initial_bias(self, layer_size):
		return np.zeros((1, layer_size))

	def forward(self, input):
		self.input = input
		# Inverted dropout
		self.dropout_mask = (np.random.rand(self.input.shape[0], self.W.shape[1]) >= self.dropout_prob) / (1 - self.dropout_prob)
		self.output = self.activation_func(self.input.dot(self.W) + self.b) * self.dropout_mask
		return self.output

	# d_output: The gradient of the loss function based on the ouput of this layer
	# Returns the gradient of the loss function based on the input to this layer
	def backward(self, d_output, reg_type, reg_const):
		# The gradient of the loss function based on the pre-activation function output of this layer
		d_pre_act_func = d_output * self.dropout_mask * self.d_activation_func(self.output)
		# Thee gradient of the loss function based on the weights used in this layer
		self.d_W = self.input.T.dot(d_pre_act_func) + self.d_regularization(reg_type, reg_const)
		# The gradient of the loss function based on the bias used in this layer
		self.d_b = np.sum(d_pre_act_func, axis=0, keepdims=True)

		# THe gradient of the loss function based on the input to this layer
		return d_pre_act_func.dot(self.W.T)

	def param_update(self, step_size):
		self.W -= step_size * self.d_W
		self.b -= step_size * self.d_b

	def d_regularization(self, reg_type, reg_const):
		if reg_type == Regularization.L2:
			return reg_const * self.W
		elif reg_type == Regularization.L1:
			return reg_const * (self.W > 0) - reg_const * (self.W < 0)
		elif reg_type == Regularization.ELASTIC_NET:
			# Using the same constant for the L1 and L2 parts. These could be different if desired.
			return self.d_regularization(Regularization.L1, reg_const) + self.d_regularization(Regularization.L2, reg_const)

# Code in this function is copied from http://cs231n.github.io/neural-networks-case-study/#gra with just some minor differences
def generate_data(N, D, K):
	X = np.zeros((N*K, D)) # data points
	y = np.zeros(N*K, dtype='uint8') # class labels for the data points
	for j in range(K):
		ix = range(N*j, N*(j+1))
		r = np.linspace(0.0, 1, N)
		t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
		X[ix] = np.c_[r * np.sin(t)**2, r * np.cos(t)]
		y[ix] = j
	return X,y

def sample_data(data, labels, sample_size):
	idx = np.random.choice(data.shape[0], sample_size, replace=False)
	return data[idx, :], labels[idx]

def center_around_point(data, point):
	return data - point

def center_around_origin(data):
	mean = np.mean(data, axis = 0)
	return center_around_point(data, mean), mean

def normalize_dimensions(data):
	return data / np.std(data, axis = 0)

# Remember to center data around origin before calling this function
def principal_component_analysis(data, reduce_to_dim):
	covariance_matrix = data.T.dot(data) / data.shape[0]
	U, S, V = np.linalg.svd(covariance_matrix)
	data_projected_reduced = data.dot(U[:, :reduce_to_dim])
	return data_projected_reduced

# Remember to center data around origin before calling this function
def whiten(data):
	covariance_matrix = data.T.dot(data) / data.shape[0]
	U, S, V = np.linalg.svd(covariance_matrix)
	data_projected = X.dot(U)

	smoothing_const = 1e-5
	eigenvalues = np.sqrt(S + smoothing_const)
	data_whitened = data_projected / eigenvalues
	return data_whitened

N = 100 # num of points per class
D = 2 # data dimensions
K = 3 # num of classes
X,y = generate_data(N, D, K)
'''
plt.subplot(2, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.subplot(2, 1, 2)
'''
X, train_mean = center_around_origin(X)
'''
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
'''

batch_size = 128
ann = ANN(2, layer_num=2, activation_func=relu)
for i in range(10000):
	# Mini-batch Gradient Descent
	data_batch, corr_labels = sample_data(X, y, batch_size)
	ann.learn_from(data_batch, corr_labels, debug=(i%1000==0))

X_eval, y_eval = generate_data(N, D, K)
# Important to use the mean that was computed on the training data, not the validation data
X_eval = center_around_point(X_eval, train_mean)
ann.evaluate(X_eval, y_eval)
