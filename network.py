import numpy as np
import cPickle
import os
import sys
import collections
from theano import *
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet import softmax
from theano.misc.pkl_utils import dump, load
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from math import floor

from PIL import Image

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def ReLU(z):
    newZ = z.clip(0, float('inf'))
    return newZ
    
def mse(y, a, n):
    """ Theano mean-squared error math. Used as cost function in network. """
    return T.sum(T.sqr(y-a))/n

def cross_entropy(y, a, n):
    return T.sum(T.nnet.binary_crossentropy(a,y))/n

def log_likelihood(y, a, n):
    return -T.sum(T.log(a[y.nonzero()]))/n
    
def l2_norm(c, weights, n, lam):
    l2_sum = 0
    for weight in weights:
        if weight is not None:
            l2_sum = l2_sum + 0.5*T.sum(T.sqr(weight))*lam
    return c + l2_sum
    
class Network():
    """ 
    A convolutional neural network class that implements the scholastic
    gradient descent learning algorithm 
    """
    def __init__(self, layers, mini_batch_size, cost_func=mse, freeze_layers=False):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.cost_func = cost_func
        self.out_funcs = []
        if not isinstance(freeze_layers,collections.Sequence):
            freeze_layers = [False for layer in layers]
        
        # Theano feedforward function that connects the output of each layer
        #     to the input of the next
         
        v = T.vector()
        out = v.dimshuffle('x', 'x', 'x', 0)
        for layer in layers:
            out = layer.set_inpt(out, training=False)
            self.out_funcs.append(theano.function(inputs = [v], outputs = out))
        self.feedforward = self.out_funcs[-1]
        
        inpt = T.matrix()
        out = inpt
        for layer in layers:
            out = layer.set_inpt(out, training=True)
            
        
                   
        # List of each individual parameter for all the layers
        self.params = []
        self.full_params = []
        self.conv_layers = [] 
        
        for k in xrange(len(layers)):
            if layers[k].is_convolutional:
                self.conv_layers += [k]
                
        for freeze_layer, layer in zip(freeze_layers,layers):
            self.full_params += layer.params
            if freeze_layer:
                # Freeze the convolutional layer parameters by not updating them.
                continue
            self.params += layer.params

        # Theano cost function (x=target output, y=output of network)
        target = T.matrix()
        cost = self.cost_func(target, out, mini_batch_size)
        # Apply L2 Normalization
        l2_lambda = T.fscalar()
        weights = [layer.w for layer in layers]
        self.fmaps = []
        for weight,layer in zip(weights,layers):
            if isinstance(layer,ConvLayer):
                self.fmaps.append(weight)
        cost = l2_norm(cost, weights, mini_batch_size, l2_lambda)
        
        grads = [T.grad(cost, param) for param in self.params]

        # weights = weights - nabla_w * eta
        eta = T.fscalar()
        i = T.iscalar()
        self.test_data_x = shared(np.zeros((mini_batch_size,784),config.floatX))
        self.test_data_y = shared(np.zeros((mini_batch_size,10),config.floatX))
        self.update_params = theano.function(inputs = [i,eta,l2_lambda],
                        outputs = [cost],
                        givens = [
                            (inpt,self.test_data_x[i*mini_batch_size:(i+1)*mini_batch_size]),
                            (target,self.test_data_y[i*mini_batch_size:(i+1)*mini_batch_size])],                            
                        #updates = self.weights - nabla_w*eta
                        updates = 
                            [(param, param - eta*grad) 
                               for (param, grad) in zip(self.params,grads)])
    
    def evaluate(self, test_data):
        total_correct = 0
        for x,y in zip(test_data[0],test_data[1]):
            guess = np.argmax(self.feedforward(x))
            actual_num = np.argmax(y)
            total_correct += (guess == actual_num)
        return int(total_correct)
                            
    def SGD(self, training_data, epochs, mini_batch_size, eta, l2_lambda = 0.00,
            test_data=None):  
        rng = numpy.random.RandomState()
        n = len(training_data[0])
        
        for iteration in xrange(epochs):
            if test_data is not None:
                num_correct = self.evaluate(test_data)
                print "Evaluation: " + str(num_correct) + " / " + str(len(test_data[0]))
            order = range(len(training_data[0]))
            rng.shuffle(order)
            mini_batch_x = [training_data[0][i] for i in order]
            mini_batch_y = [training_data[1][i] for i in order]
            
            self.test_data_x.set_value(mini_batch_x)
            self.test_data_y.set_value(mini_batch_y)
            
            # Update the parameters for each image in the mini_batch
            for i in xrange(int(floor(len(mini_batch_x)/mini_batch_size))):
                cost = self.update_params(i,eta,l2_lambda)
                if i==0:
                    print cost
                if not np.isfinite(cost):
                    print 'Cost function became non-finite. Aborting...'
                    sys.exit()
        if test_data is not None:
            num_correct = self.evaluate(test_data)
            print "Evaluation: " + str(num_correct) + " / " + str(len(test_data[0]))
    
    def save_params(self, filename):
        cwd = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cwd,filename+'.zip'),'wb') as f:
            params = [param.get_value() for param in self.full_params]
            dump(params, f)
    
    def load_params(self, filename):
        cwd = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cwd,filename+'.zip'),'rb') as f:
            params = load(f)
            #print params,self.params[0].get_value()
            for i,param in enumerate(params):
                self.full_params[i].set_value(param)
                
    def save_feature_maps(self,filename):
        cwd = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cwd,filename+'.zip'),'wb') as f:
            for k in self.conv_layers:
                params += [param.get_value() for param in self.params[k]]
                dump(params, f)
        
    def load_feature_maps(self, filename, src_layer, src_fmap_nums, dst_layer, dst_fmap_nums):
        cwd = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cwd,filename+'.zip'),'rb') as f:
            params = load(f)
            curr_fmaps = self.layers[dst_layer].w.get_value()
            curr_biases = self.layers[dst_layer].b.get_value()
            src_w_param_num = self.full_params.index(self.layers[src_layer].w)
            src_b_param_num = self.full_params.index(self.layers[src_layer].b)
            for src_fmap_num, dst_fmap_num in zip(src_fmap_nums,dst_fmap_nums):
                curr_fmaps[dst_fmap_num] = params[src_w_param_num][src_fmap_num]
                curr_biases[dst_fmap_num] = params[src_b_param_num][src_fmap_num]
            self.layers[dst_layer].w.set_value(curr_fmaps)
            self.layers[dst_layer].b.set_value(curr_biases)
            
        
rng = numpy.random.RandomState()
    
class ConvLayer(object):
    """Used to create a convolutional layer."""

    def __init__(self, filter_shape, image_shape, border_mode='valid', unique_weights_per_input=False,
                            dropout_rate=0.0, activation_fn=sigmoid):
        """
        `filter_shape` : a tuple of length 4, whose entries are
                0: the number of filters,
                1: the number of input feature maps,
                2: the filter height, and
                3: the filter width.
        `image_shape` : a tuple of length 4, whose entries are 
                0: the mini-batch size,
                1: the number of input feature maps,
                2: the image height, and
                3: the image width.
        """        
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation_fn=activation_fn
        self.dropout_rate = dropout_rate
        self.unique_weights_per_input = unique_weights_per_input
        self.border_mode = border_mode
        self.nrg = RandomStreams()
        self.is_convolutional = True
        
        # initialize weights and biases
        n_outputs = image_shape[0]*image_shape[2]*image_shape[3]
        self.b = shared(rng.normal(0.5,0.5,self.filter_shape[0]).astype(config.floatX))
        if self.unique_weights_per_input:
            self.w = shared(rng.normal(0,1.0/(n_outputs*(1-self.dropout_rate)),filter_shape).astype(config.floatX))
            self.params = [self.w, self.b]
        else:
            self.w = shared(rng.normal(0,1.0/(n_outputs*(1-self.dropout_rate)),(filter_shape[0], 1, filter_shape[2], filter_shape[3])).astype(config.floatX), broadcastable=(False, True, False, False))
            self.inp_fmap_coeff_w = shared(rng.normal(0,1.0,(filter_shape[0],filter_shape[1],1,1)).astype(config.floatX), broadcastable=(False, False, True, True))
            self.params = [self.w, self.inp_fmap_coeff_w, self.b]
        

    def set_inpt(self, inpt, training):
        # image, feature maps, imagex, imagey
        if training:
            new_shape = self.image_shape
        else:
            new_shape = (1,)+self.image_shape[1:]
        self.inpt = inpt.reshape(new_shape)
        
        # Extend w for all input maps
        if self.unique_weights_per_input:
            full_w = self.w
        else:
            full_w = self.w*self.inp_fmap_coeff_w
            
        #biases = self.b
        if training:
            # Bernoulli process needed for dropout
            bern = self.nrg.binomial(size = self.image_shape,p=1.0-self.dropout_rate).astype(config.floatX)
            #bern = bern.reshape(T.shape(self.inpt))
            self.inpt=self.inpt*bern
            #biases = self.b*bern
        else:
            inpt = inpt*(1-self.dropout_rate)
        conv_out = conv2d(self.inpt, full_w, image_shape=new_shape, filter_shape=self.filter_shape, border_mode=self.border_mode)
        self.output = conv_out + self.b.dimshuffle(('x', 0, 'x', 'x'))
        return self.activation_fn(self.output)

class PoolLayer(object):
    """Used to create a pooling layer."""
    
    def __init__(self, image_shape, pool_size, pool_mode = 'max'):
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.is_convolutional = False
        
        self.params = []
        self.w = None
        
    def set_inpt(self, inpt, training):
        if training:
            new_shape = self.image_shape
        else:
            new_shape = (1,)+self.image_shape[1:]
        self.inpt = inpt.reshape(new_shape)
        #images2neibs currently not fully implemented
        #neibs = T.nnet.neighbours.images2neibs(self.inpt, self.pool_size)
        #return self.pool_func(neibs)
        
        return max_pool_2d(self.inpt,self.pool_size, ignore_border = False, 
                                st=None, padding=(0,0), mode=self.pool_mode)
        
        
class FullyConnectedLayer(object):
    """Used to create a fully connected layer of neurons."""
    
    def __init__(self, n_inputs, n_outputs, dropout_rate = 0.0, activation_fn=sigmoid):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.is_convolutional = False
        
        # Initializing weights and biases to samples from normal Gaussian
        self.w = shared(rng.normal(0,1.0/(self.n_outputs*(1-self.dropout_rate)),(self.n_outputs,self.n_inputs)).astype(config.floatX), borrow=True)
        self.b = shared(rng.normal(0,1.0,(self.n_outputs,)).astype(config.floatX), borrow=True)
        self.params = [self.w, self.b]
        
        self.nrg = RandomStreams()
    
    def set_inpt(self, inpt, training):
        self.inpt = T.flatten(inpt, 2)
        if training:
            bern = self.nrg.binomial(size = T.shape(self.inpt),p=1.0-self.dropout_rate, ndim=2).astype(config.floatX)
            #bern = bern.reshape(T.shape(inpt))
            inpt = inpt*bern
        else:
            inpt = inpt*(1-self.dropout_rate)
        
        self.output = T.dot(self.w, self.inpt.T).T + self.b.dimshuffle('x',0)
        return self.activation_fn(self.output)