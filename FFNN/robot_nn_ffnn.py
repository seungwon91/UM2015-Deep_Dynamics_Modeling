
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

# Hidden layer for multi-layer neural network
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W_hidden=None, b_hidden=None,
                 activation=T.nnet.relu):
        """
        Hidden unit activation is given by: ReLu(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.activation = activation

        if W_hidden is None:
            W_values = np.asarray(rng.uniform(low=-0.01/n_in, high=0.01/n_in, size=(n_in, n_out)), dtype=theano.config.floatX)
            W_hidden = theano.shared(value=W_values, name='W_hidden', borrow=True)

        if b_hidden is None:
            b_values = np.asarray(rng.uniform(low=-0.01, high=0.01, size=(n_out,)), dtype=theano.config.floatX)
            b_hidden = theano.shared(value=b_values, name='b_hidden', borrow=True)

        self.W = W_hidden
        self.b = b_hidden

        # parameters of the model
        self.params = [self.W, self.b]

    def __call__(self, input, true_result=None):
        lin_output = T.dot(input, self.W) + self.b.dimshuffle('x',0)

        if self.activation=="leaky-relu":
            output = T.nnet.relu(lin_output, alpha=0.005)
        else:
            output = self.activation(lin_output)

        if true_result is None:
            return output
        else:
            if true_result.ndim != output.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', output.type)
                )
            diff = abs(output - true_result)
            return [diff.sum(), (diff**2).sum(), diff.max()]


# output layer of multi-layer neural network for regression
class LinearLayer(object):
    def __init__(self, rng, n_in, n_out, W_lin=None, b_lin=None):
        """ Initialize the parameters of the linear layer

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the output space
        """
        # initialize with 0 the weights W and b as a matrix of shape (n_in, n_out)
        if W_lin is None:
            W_values = np.asarray(rng.uniform(low=-0.05/n_in, high=0.05/n_in, size=(n_in, n_out)), dtype=theano.config.floatX)
            W_lin = theano.shared(value=W_values, name='W_lin', borrow=True)

        if b_lin is None:
            b_values = np.asarray(rng.uniform(low=-0.01, high=0.01, size=(n_out,)), dtype=theano.config.floatX)
            b_lin = theano.shared(value=b_values, name='b_lin', borrow=True)

        self.W = W_lin
        self.b = b_lin

        # parameters of the model
        self.params = [self.W, self.b]

    def __call__(self, input, true_result=None):
        output = T.dot(input, self.W) + self.b.dimshuffle('x',0)
        if true_result is None:
            return output
        else:
            if true_result.ndim != output.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', output.type)
                )
            diff = abs(output - true_result)
            return [diff.sum(), (diff**2).sum(), diff.max()]


#Multi-Layer Regression Network
class FFNN3(object):
    def __init__(self, rng, n_in, n_hidden, n_out, batch_size, hist_window, pred_window, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        if para_list is None:
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.nnet.relu
            )
        else:
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_hidden,
                W_hidden=para_list[0],
                b_hidden=para_list[1],
                activation=T.nnet.relu
            )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        if para_list is None:
            self.outputLayer = LinearLayer(
                n_in=n_hidden,
                n_out=n_out,
                rng=rng,
            )
        else:
            self.outputLayer = LinearLayer(
                n_in=n_hidden,
                n_out=n_out,
                rng=rng,
                W_lin=para_list[2],
                b_lin=para_list[3]
            )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params

        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window


    def __call__(self, input, true_result=None):
        for cnt in range(self.pred_window):
            # shifting input
            if cnt < 1:
                sliced_input = input[:, 0:(2*self.hist_window+2)]
            else:
                sliced_input = T.concatenate([sliced_input[:, 2:(2*self.hist_window)], result[:, (2*cnt-2):(2*cnt)], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of model output
            hidden_output = self.hiddenLayer(sliced_input)
        
            output = self.outputLayer(hidden_output)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result)
            return [diff.sum(), (diff**2).sum(), diff.max()]


#Multi-Layer Regression Network
class FFNN3_relu(object):
    def __init__(self, rng, n_in, n_hidden, n_out, batch_size, hist_window, pred_window, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        if para_list is None:
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.nnet.relu
            )
        else:
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_hidden,
                W_hidden=para_list[0],
                b_hidden=para_list[1],
                activation=T.nnet.relu
            )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        if para_list is None:
            self.outputLayer = HiddenLayer(
                n_in=n_hidden,
                n_out=n_out,
                rng=rng,
                activation=T.nnet.relu
            )
        else:
            self.outputLayer = HiddenLayer(
                n_in=n_hidden,
                n_out=n_out,
                rng=rng,
                W_lin=para_list[2],
                b_lin=para_list[3],
                activation=T.nnet.relu
            )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params

        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window


    def __call__(self, input, true_result=None):
        for cnt in range(self.pred_window):
            # shifting input
            if cnt < 1:
                sliced_input = input[:, 0:(2*self.hist_window+2)]
            else:
                sliced_input = T.concatenate([sliced_input[:, 2:(2*self.hist_window)], result[:, (2*cnt-2):(2*cnt)], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of model output
            hidden_output = self.hiddenLayer(sliced_input)
        
            output = self.outputLayer(hidden_output)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result)
            return [diff.sum(), (diff**2).sum(), diff.max()]


#Multi-Layer input/state distinguished Regression Network
class FFNN3_FFNN3(object):
    def __init__(self, rng, n_in, n_hidden_list, n_out, batch_size, hist_window, pred_window, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # FFNN3 for observation history
        if para_list is None:
            self.first_ffnn3 = FFNN3_relu(
                rng=rng,
                n_in=n_in-2,
                n_hidden=n_hidden_list[0],
                n_out=n_hidden_list[1],
                batch_size=batch_size,
                hist_window=hist_window,
                pred_window=pred_window
            )
        else:
            self.first_ffnn3 = FFNN3_relu(
                rng=rng,
                n_in=n_in-2,
                n_hidden=n_hidden_list[0],
                n_out=n_hidden_list[1],
                batch_size=batch_size,
                hist_window=hist_window,
                pred_window=pred_window,
                para_list=para_list[0:4]
            )

        # FFNN3 for state(from observation history) and command input
        if para_list is None:
            self.second_ffnn3 = FFNN3(
                rng=rng,
                n_in=n_hidden_list[1]+2,
                n_hidden=n_hidden_list[2],
                n_out=n_out,
                batch_size=batch_size,
                hist_window=hist_window,
                pred_window=pred_window
            )
        else:
            self.second_ffnn3 = FFNN3(
                rng=rng,
                n_in=n_hidden_list[1]+2,
                n_hidden=n_hidden_list[2],
                n_out=n_out,
                batch_size=batch_size,
                hist_window=hist_window,
                pred_window=pred_window,
                para_list=para_list[4:8]
            )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (self.first_ffnn3.L1 + self.second_ffnn3.L1)

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.first_ffnn3.L2_sqr + self.second_ffnn3.L2_sqr)

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_ffnn3.params + self.second_ffnn3.params

        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window


    def __call__(self, input, true_result=None):
        for cnt in range(self.pred_window):
            # shifting input
            if cnt < 1:
                sliced_input = input[:, 0:(2*self.hist_window+2)]
            else:
                sliced_input = T.concatenate([sliced_input[:, 2:(2*self.hist_window)], result[:, (2*cnt-2):(2*cnt)], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of state-ffnn3 output
            first_ffnn_output = self.first_ffnn3(sliced_input[:,0:(2*self.hist_window)])
            internal_state = T.concatenate([first_ffnn_output, sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)]], axis=1)
            
            output = self.second_ffnn3(internal_state)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result)
            return [diff.sum(), (diff**2).sum(), diff.max()]

