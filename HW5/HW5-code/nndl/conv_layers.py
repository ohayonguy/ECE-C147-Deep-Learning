import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  padded_x = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
  out = np.zeros(shape=(x.shape[0], w.shape[0], int(1 + (x.shape[2] + 2 * pad - w.shape[2]) / stride),
                        int(1 + (x.shape[3] + 2 * pad - w.shape[3]) / stride)))
  for example in range(x.shape[0]):
    for f in range(out.shape[1]):
      for i in range(out.shape[2]):
        for j in range(out.shape[3]):
          out[example, f, i, j] = b[f] + np.sum(w[f] * padded_x[example, :, i * stride:i * stride + w.shape[2],
                                                   j * stride:j * stride + w.shape[3]])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dw = np.zeros_like(w)
  dx = np.zeros_like(x)
  db = np.zeros_like(b)
  dxpad = np.zeros_like(xpad)

  for f in range(dout.shape[1]):  # F
    for example in range(dout.shape[0]):  # N
      for h_tag in range(dout.shape[2]):  # H'
        for w_tag in range(dout.shape[3]):  # W'
          offset_h = stride * h_tag
          offset_w = stride * w_tag
          dw[f] += dout[example, f, h_tag, w_tag] * xpad[example, :, offset_h:offset_h+w.shape[2],
                                                    offset_w:offset_w+w.shape[3]]
          dxpad[example, :, offset_h:offset_h+w.shape[2],
                        offset_w:offset_w+w.shape[3]] += dout[example, f, h_tag, w_tag] * w[f]
  db = np.sum(dout, axis=(0, 2, 3))
  dx = dxpad[:, :, pad:-pad, pad:-pad]  # The padded parameters are not relevant.
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  out_height = int((x.shape[2] - pool_height) / stride) + 1
  out_width = int((x.shape[3] - pool_width) / stride) + 1
  out = np.zeros(shape=(x.shape[0], x.shape[1], out_height, out_width))

  for example in range(out.shape[0]):
    for c in range(out.shape[1]):
      for h in range(out.shape[2]):
        for w in range(out.shape[3]):
          out[example, c, h, w] = \
            np.amax(x[example, c, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  dx = np.zeros_like(x)
  for f in range(dout.shape[1]):  # F
    for example in range(dout.shape[0]):  # N
      for h_tag in range(dout.shape[2]):  # H'
        for w_tag in range(dout.shape[3]):  # W'
          pool_window = x[example, f, h_tag * stride:h_tag * stride + pool_height, w_tag * stride:w_tag * stride + pool_width]
          h_max_index, w_max_index = np.unravel_index(np.argmax(pool_window, axis=None), pool_window.shape)
          dx[example, f, h_max_index + h_tag*stride, w_max_index + w_tag*stride] = dout[example, f, h_tag, w_tag]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  out = np.array(batchnorm_forward(x.reshape(x.swapaxes(0,1).reshape(x.shape[1], -1).T, gamma, beta, bn_param))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta