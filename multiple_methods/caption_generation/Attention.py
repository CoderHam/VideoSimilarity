from keras import backend as k
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class Attention_Layer(Layer):
	def __init__(self, output_dim, input_dim=None, **kwargs):
		self.output_dim = output_dim
		self.input_dim = input_dim
		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(Attention_Layer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(name="W",shape=(4096, self.output_dim), initializer = 'uniform', trainable = True)

		self.U = self.add_weight(name="U",shape=(256, self.output_dim), initializer = 'uniform', trainable = True)

		self.b = self.add_weight(name="b", shape=(self.output_dim,), initializer = 'uniform', trainable = True)

		self.w = self.add_weight(name="w", shape=(self.output_dim, 4096), initializer = 'uniform', trainable = True)

	def call(self, inputs, mask=None):

		print("\n\tAttention_Layer....")
		if len(inputs) <=1:
			print("got none input to attention layer....")

		h = inputs[0]
		v = inputs[1]
				
		print("h : ",k.int_shape(h))
		print("v : ",k.int_shape(v))
		print("W : ",k.int_shape(self.W))
		print("U : ",k.int_shape(self.U))
		
		####################Attention Layer I#############################
		U_h = k.dot(h,self.U)
		W_v = k.dot(v,self.W)

		U_h = k.reshape(U_h,(-1, k.int_shape(U_h)[1], 1, k.int_shape(U_h)[2]))
		W_v = k.reshape(W_v,(-1, 1, k.int_shape(W_v)[1], k.int_shape(W_v)[2]))

		print("U_h : ",k.int_shape(U_h))
		print("W_v : ",k.int_shape(W_v))


		f = k.tanh(W_v + U_h + self.b) 
		
		print("f : ",k.int_shape(f))
		
		####################Attention Layer II############################
		q = k.dot(f,self.w)
		print("q : ",k.int_shape(q))

		
		####################Sequential Softmax Layer######################
		
		beta = k.exp(q)/k.sum(k.exp(q), axis=0)
		print("beta : ",k.int_shape(beta)) 
		
		####################Weighted Averaging Layer######################
		
		v = k.reshape(v, (-1, 1, k.int_shape(v)[1], k.int_shape(v)[2]))
		u = beta*v	
		u = tf.transpose(u, [2,3,1,0])
		u = k.sum(u, axis=0)
		self.u = tf.transpose(u, [2,1,0])

		print("u : ",k.int_shape(self.u))

		return self.u

		
	def compute_output_shape(self, input_shape):
		output_dim = k.int_shape(self.u)
		return (None, output_dim[1], output_dim[2])

