from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train_eg_no = X.shape[0]
    num_classes = W.shape[1]
    scores = np.dot(X,W)
    for i in range(train_eg_no):  
      current_scores = scores[i]
      #normalization trick
      current_scores -= np.max(current_scores)
      correct_class = current_scores[y[i]]

      #loss
      sum_i = np.sum(np.exp(current_scores))
      li = - np.log(np.exp(correct_class) / sum_i)
      loss += li

      #gradient
      for j in xrange(num_classes):
            if j == y[i]:
                dW[:, y[i]] += (np.exp(current_scores[j]) / sum_i - 1) * X[i, :]
            else:
                dW[:, j] += (np.exp(current_scores[j]) / sum_i) * X[i, :]


    loss = loss/train_eg_no
    dW = dW / train_eg_no

    #regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train_eg_no = X.shape[0]
    num_classes = W.shape[1]
    scores = np.dot(X,W)
    #normalization trick
    norm = np.tile(np.max(scores, axis =1), (10,1)).T
    scores -= norm
    correct_class = scores[np.arange(scores.shape[0]),y]

    #loss
    sum_i = np.sum(np.exp(scores), axis=1)
    li = - np.log(np.exp(correct_class) / sum_i)
    loss = np.sum(li)

    #gradient
    sum_2 = np.sum(np.exp(scores), axis=1, keepdims=True)
    ind = np.zeros(scores.shape)
    ind[np.arange(scores.shape[0]), y] = 1
    dW = X.T.dot((np.exp(scores) / sum_2 - ind))

    loss = loss/train_eg_no
    dW = dW / train_eg_no

    #regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
