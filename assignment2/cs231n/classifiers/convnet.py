import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MultiLayerConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:
  
  [conv - [bn?] - relu - conv - [bn?] - relu - 2x2 max pool] * conv_depth - 
  [affine - [bn?] - relu] * fc_depth - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), conv_depth=3, fc_depth=2, num_filters=None,
               filter_size=3, hidden_dims=None, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - conv_depth: Number of [conv relu conv relu pool] blocks
    - fc_depth: Number of fc layers
    - num_filters: List storing number of filters to use in each convolutional layer
                   If set to None, default is [32, 32, 64, 64, 128, 128 ...]
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dims: List storing number of units to use in the fully-connected hidden layer
                   If set to None, default is [200, 400, 600 ...]
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    # [32, 32, 64, 64, 128, 128 ...]
    if num_filters is None:
      num_filters = 32 * 2**(np.arange(conv_depth).repeat(2))
    
    # [200, 400, 600 ...]
    if hidden_dims is None:
      hidden_dims = (np.arange(fc_depth) + 1) * 200
    
    ############################################################################
    # TODO: Initialize weights and biases for the multi-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    C, H, W = input_dim
    WW = filter_size
    HH = filter_size
    idx = 1
    F0 = C
    # init conv - relu - pool layers
    for i in range(conv_depth * 2):
      F1 = num_filters[i]
      self.params['W' + str(idx)] = weight_scale * np.random.randn(F1, F0, HH, WW)
      self.params['b' + str(idx)] = np.zeros(F1)
      F0 = F1
      idx += 1
    
    # init fc layers
    D = (H * W * F0) / 4**conv_depth
    for i in range(fc_depth):
      self.params['W' + str(idx)] = weight_scale * np.random.randn(D, hidden_dims[i])
      self.params['b' + str(idx)] = np.zeros(hidden_dims[i])
      D = hidden_dims[i]
      idx += 1
    
    # init one affine layer
    self.params['W' + str(idx)] = weight_scale * np.random.randn(D, num_class)
    self.params['b' + str(idx)] = np.zeros(num_class)
   
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    scores, cache3 = affine_forward(out2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    dout2, dW3, db3 = affine_backward(dscores, cache3)
    dW3 += self.reg * W3
    dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
    dW2 += self.reg * W2
    dx, dW1, db1 = conv_relu_pool_backward(dout1, cache1)
    dW1 += self.reg * W1
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads