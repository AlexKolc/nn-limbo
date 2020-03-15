import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    
    preds = predictions.copy()
    if len(preds.shape) == 1:
        preds -= np.max(preds)
        exps = np.exp(preds)
        probs = exps / np.sum(exps)
    else:
        preds -= np.max(preds, axis=1).reshape(-1, 1)
        exps = np.exp(preds)
        probs = exps / np.sum(exps, axis=1).reshape(-1, 1)
    return probs



def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    # raise Exception("Not implemented!")
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    
    if len(probs.shape) == 1: 
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        logs = -np.log(probs[range(batch_size), target_index.T])   
        loss = np.sum(logs) / batch_size
        
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    # raise Exception("Not implemented!")
    probs = softmax(predictions) 
    loss = cross_entropy_loss(probs, target_index)
    d_preds = probs.copy()
    if len(d_preds.shape) == 1:
        d_preds[target_index] -= 1
    else:
        batch_size = probs.shape[0]
        d_preds[range(batch_size), target_index.T] -= 1
                
    return loss, d_preds / target_index.shape[0]


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        # raise Exception("Not implemented!")
        self.X = X
        return np.maximum(0, X)
            
    def backward(self, d_out):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        d_result = d_out * np.greater_equal(self.X, 0)        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        # raise Exception("Not implemented!")
        self.X = X
        return np.dot(X, self.W.value) + self.B.value
        
    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        # raise Exception("Not implemented!")        
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0)[:, None].T
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 2 * self.padding + height - self.filter_size + 1 
        out_width = 2 * self.padding + width - self.filter_size + 1
        
        X_with_padding = np.zeros((batch_size, 2 * self.padding + height, 2 * self.padding + width, channels))
        X_with_padding[:,self.padding:height + self.padding, self.padding:self.padding + width, :] = X

        self.X_forward = X, X_with_padding 
         
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                window = X_with_padding[:, y : y + self.filter_size, x : x + self.filter_size, :, np.newaxis]
                out[:, y, x, :] = np.sum(window * self.W.value, axis = (1, 2, 3)) + self.B.value
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        X, X_with_padding = self.X_forward
        X_grad = np.zeros(X_with_padding.shape)
        
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                window = X_with_padding[:, y : y + self.filter_size, x : x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * window, axis=0)
                X_grad[:, y : y + self.filter_size, x : x + self.filter_size, :] += np.sum(self.W.value * grad, axis=4) 

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        
        return X_grad[:, self.padding:height + self.padding, self.padding:width + self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        batch_size, height, width, channels = X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros((batch_size, out_height, out_width, channels))

        self.X = X

        for y in range(out_height):
            for x in range(out_width):
                window = X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                out[:, y, x, :] = np.max(window, axis=(1,2))
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros(self.X.shape)

        for y in range(out_height):
            for x in range(out_width):
                window = self.X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                dx = (d_out[:, y, x, :])[:, np.newaxis, np.newaxis, :]

                max_el = (window == np.max(window, axis=(1,2))[:, np.newaxis, np.newaxis, :])

                out[:, y:y+self.pool_size, x:x+self.pool_size, :] += dx*max_el
        return out

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
