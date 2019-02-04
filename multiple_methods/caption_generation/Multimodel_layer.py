from keras import backend as k
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class Multimodel_Layer(Layer):
	def __init__(self, output_dim, input_dim=None, **kwargs):
		self.output_dim = output_dim
		self.input_dim = input_dim
		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(Multimodel_Layer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(name="W",shape=(4096, self.output_dim), initializer = 'uniform', trainable = True)

		self.U = self.add_weight(name="U", shape=(256, self.output_dim), initializer = 'uniform', trainable = True)

		self.b = self.add_weight(name="b", shape=(self.output_dim,), initializer = 'uniform', trainable = True)

	def call(self, inputs, mask=None):

		if len(inputs) <=1:
			print("got none input to attention layer....")

		h = inputs[0]
		v = inputs[1]

		####################Multimodel Layer#############################
		U_h = k.dot(h,self.U)
		W_v = k.dot(v,self.W)
		
		self.f = k.tanh(W_v + U_h + self.b)
		#print "f : ",k.int_shape(self.f)
		
		return self.f
		
	def compute_output_shape(self, input_shape):
		output_dim = k.int_shape(self.f)
		return (None, output_dim[1], output_dim[2])

