import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count += 1
        dW[:,j] += X[i]
    dW[:,y[i]] += -X[i] * count

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
    
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  Scores = X.dot(W)
  correct_class_scores = Scores[np.arange(num_train), y]
  margins = Scores.copy()
  # reshape scores vector of correct class to make it broadcast correctly
  margins -= correct_class_scores.reshape(correct_class_scores.shape[0], 1)
  margins += 1
  margins[np.arange(num_train), y] = 0
  margins[np.where(margins < 0)] = 0
  # losses = margins.sum(axis=1)
  # loss = losses.sum() / num_train
  # directly summing over all margins is much faster (0.005 vs 0.02)
  loss = margins.sum() / num_train
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # COnstruct a matrix Ind of shape (C, N)
  # Ind[j, i] = -count(margin_ij > 0) if j = y_i
  #           = 1(margin_ij > 0) if j != y_i
  # then dW[j].T = 1 / N * sum_over_i(Ind[i, j] * X[i])
  #              = 1/ N * Ind[j] * X
  # hense dW.T = Ind * X
  # where '*' means dot product
  count = (margins > 0).sum(axis=1).astype(np.float)
  IndT = (margins > 0).astype(np.float)
  IndT[np.arange(num_train), y] = - count
  dW = IndT.T.dot(X).T / num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
