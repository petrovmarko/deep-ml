
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None

	def initialize(self, optimizer):
		# Initialize weights W, biases w0, and optimizers
		K = 1 / np.sqrt(self.input_shape[0])
		self.W = np.random.uniform(-K,K, (self.input_shape[0], self.n_units))
		self.w0 = np.zeros(self.n_units)
		self.W_opt = copy.deepcopy(optimizer)
		self.w0_opt = copy.deepcopy(optimizer)

	def parameters(self):
		# Return total number of parameters
		return self.w0.size + self.W.size

	def forward_pass(self, X, training=True):
		# Compute and return the forward pass
		if training:
			self.layer_input = X
		return X @ self.W + self.w0

	def backward_pass(self, accum_grad):
		# Compute gradients, update weights if trainable, return grad w.r.t. i
		grad_W = self.layer_input.T @ accum_grad
		grad_b = np.sum(accum_grad, axis=0)

		if self.trainable:
			self.W_opt.update(self.W, grad_W)
			self.w0_opt.update(self.w0, grad_b)
		return accum_grad @ self.W.T
	def output_shape(self):
		# Return output shape tuple
		return (self.n_units, )
