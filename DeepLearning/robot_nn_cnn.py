
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
            W_values = np.asarray(rng.uniform(low=-0.0001/n_in, high=0.0001/n_in, size=(n_in, n_out)), dtype=theano.config.floatX)
            W_hidden = theano.shared(value=W_values, name='W_hidden', borrow=True)

        if b_hidden is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b_hidden = theano.shared(value=b_values, name='b_hidden', borrow=True)

        self.W = W_hidden
        self.b = b_hidden

        # parameters of the model
        self.params = [self.W, self.b]

    def __call__(self, input, true_result=None):
        lin_output = T.dot(input, self.W) + self.b
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
            W_values = np.asarray(rng.uniform(low=-0.0005/n_in, high=0.0005/n_in, size=(n_in, n_out)), dtype=theano.config.floatX)
            W_lin = theano.shared(value=W_values, name='W_lin', borrow=True)

        if b_lin is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b_lin = theano.shared(value=b_values, name='b_lin', borrow=True)

        self.W = W_lin
        self.b = b_lin

        # parameters of the model
        self.params = [self.W, self.b]

    def __call__(self, input, true_result=None):
        output = T.dot(input, self.W) + self.b
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


# convolution layer
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(1, 2), W_conv=None, b_conv=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))


        # initialize weights with random weights
        # the bias is a 1D tensor -- one bias per output feature map
        if W_conv is None:
            W_bound = np.sqrt(1. / (fan_in + fan_out))
            W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)
            W_conv = theano.shared(value=W_values, name='W_conv', borrow=True)

        if b_conv is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b_conv = theano.shared(value=b_values, name='b_conv', borrow=True)

        self.W = W_conv
        self.b = b_conv
        self.filter_shape = filter_shape
        self.input_shape = image_shape
        self.poolsize = poolsize

        # store parameters of this layer
        self.params = [self.W, self.b]


    def __call__(self, input, true_result=None):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.input_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        # use ReLu as output activation function
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

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


#3 Layer convolutional Network(input-conv-conv)
class ConvLayer3(object):
    def __init__(self, rng, n_in, batch_size, hist_window, n_kerns = [32, 32], kern_size = [[1, 4],[1, 3]], para_list=None, flatten_out=True):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        #layer1_n_in = int(hist_window-kern_size[0][1]+1)
        layer1_n_in = int(hist_window-kern_size[0][0]+1)

        # The first convolutional pooling layer
        # input data (1, 2, hist_window) -> (1, 2, hist_window-3)
        # no pooling / output to (n_kerns[0], 2, (hist_window-3))

        if para_list is None:
            self.Layer0 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_in, hist_window, 2),
                filter_shape=(n_kerns[0], n_in, kern_size[0][0], kern_size[0][1]),
                poolsize=(1, 1)
            )
        else:
            self.Layer0 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_in, hist_window, 2),
                filter_shape=(n_kerns[0], n_in, kern_size[0][0], kern_size[0][1]),
                poolsize=(1, 1),
                W_conv = para_list[0],
                b_conv = para_list[1]
            )

        # The linear regression layer
        # input data (n_kerns[0], 2, (hist_window-3)) -> (n_kerns[0], 2, hist_window-5)
        # no pooling / output dimension (n_kerns[1], 2, hist_window-5)

        if para_list is None:
            self.Layer1 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_kerns[0], layer1_n_in, 3-kern_size[0][1]),
                filter_shape=(n_kerns[1], n_kerns[0], kern_size[1][0], kern_size[1][1]),
                poolsize=(1, 1)
            )
        else:
            self.Layer1 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_kerns[0], layer1_n_in, 3-kern_size[0][1]),
                filter_shape=(n_kerns[1], n_kerns[0], kern_size[1][0], kern_size[1][1]),
                poolsize=(1, 1),
                W_conv = para_list[2],
                b_conv = para_list[3]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.Layer0.params + self.Layer1.params
        self.batch_size = batch_size
        self.num_ch_in = n_in
        self.hist_window = hist_window
        self.flatten_out = flatten_out


    def __call__(self, input, true_result=None):
        layer0_input = input[:,0:2*self.hist_window*self.num_ch_in].reshape((self.batch_size, self.num_ch_in, self.hist_window, 2))
        layer0_output = self.Layer0(layer0_input)
        output = self.Layer1(layer0_output, true_result)
        if (true_result is None) and self.flatten_out:
            return output.flatten(2)
        else:
            return output


#4 Layer convolutional Network(input-conv-conv)
class ConvLayer4(object):
    def __init__(self, rng, n_in, batch_size, hist_window, n_kerns = [64, 32, 16], kern_size = [[1, 4],[1, 3],[2, 2]], para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        #layer1_n_in = int(hist_window-kern_size[0][1]+1)
        #layer2_n_in = int(layer1_n_in-kern_size[1][1]+1)
        layer1_n_in = int(hist_window-kern_size[0][0]+1)
        layer2_n_in = int(layer1_n_in-kern_size[1][0]+1)


        # The first convolutional pooling layer
        # input data (1, 2, hist_window) -> (1, 2, hist_window-3)
        # no pooling / output to (n_kerns[0], 2, (hist_window-3))


        if para_list is None:
            self.Layer0 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_in, hist_window, 2),
                filter_shape=(n_kerns[0], n_in, kern_size[0][0], kern_size[0][1]),
                poolsize=(1, 1)
            )
        else:
            self.Layer0 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_in, hist_window, 2),
                filter_shape=(n_kerns[0], n_in, kern_size[0][0], kern_size[0][1]),
                poolsize=(1, 1),
                W_conv = para_list[0],
                b_conv = para_list[1]
            )

        # The second convolutional pooling layer
        # input data (n_kerns[0], 2, (hist_window-3)) -> (n_kerns[1], 2, hist_window-5)
        # no pooling / output dimension (n_kerns[1], 2, hist_window-5)

        if para_list is None:
            self.Layer1 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_kerns[0], layer1_n_in, 3-kern_size[0][1]),
                filter_shape=(n_kerns[1], n_kerns[0], kern_size[1][0], kern_size[1][1]),
                poolsize=(1, 1)
            )
        else:
            self.Layer1 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_kerns[0], layer1_n_in, 3-kern_size[0][1]),
                filter_shape=(n_kerns[1], n_kerns[0], kern_size[1][0], kern_size[1][1]),
                poolsize=(1, 1),
                W_conv = para_list[2],
                b_conv = para_list[3]
            )

        # The third convolutional pooling layer
        # input data (n_kerns[1], 2, (hist_window-5)) -> (n_kerns[2], 1, hist_window-6)
        # no pooling / output dimension (n_kerns[2], 1, hist_window-6)

        if para_list is None:
            self.Layer2 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_kerns[1], layer2_n_in, 4-kern_size[0][1]-kern_size[1][1]),
                filter_shape=(n_kerns[2], n_kerns[1], kern_size[2][0], kern_size[2][1]),
                poolsize=(1, 1)
            )
        else:
            self.Layer2 = LeNetConvPoolLayer(
                rng,
                image_shape=(batch_size, n_kerns[1], layer2_n_in, 4-kern_size[0][1]-kern_size[1][1]),
                filter_shape=(n_kerns[2], n_kerns[1], kern_size[2][0], kern_size[2][1]),
                poolsize=(1, 1),
                W_conv = para_list[4],
                b_conv = para_list[5]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.Layer0.params + self.Layer1.params + self.Layer2.params
        self.batch_size = batch_size
        self.num_ch_in = n_in
        self.hist_window = hist_window



    def __call__(self, input, true_result=None):
        layer0_input = input[:,0:2*self.hist_window*self.num_ch_in].reshape((self.batch_size, self.num_ch_in, self.hist_window, 2))
        layer0_output = self.Layer0(layer0_input)
        layer1_output = self.Layer1(layer0_output)
        output = self.Layer2(layer1_output, true_result)

        if true_result is None:
            return output.flatten(2)
        else:
            return output


# 3 Layer of Feed-Forward Network for Regression
class FeedForwardLayer3(object):
    def __init__(self, rng, n_in, n_hidden, n_out, para_list=None):
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


    def __call__(self, input, true_result=None):
        hidden_output = self.hiddenLayer(input)
        output = self.outputLayer(hidden_output, true_result)
        return output


# 4 Layer of Feed-Forward Network for Regression
class FeedForwardLayer4(object):
    def __init__(self, rng, n_in, n_hidden, n_out, para_list=None):
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
            self.hiddenLayer1 = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_hidden[0],
                activation=T.nnet.relu
            )
        else:
            self.hiddenLayer1 = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_hidden[0],
                W_hidden=para_list[0],
                b_hidden=para_list[1],
                activation=T.nnet.relu
            )

        if para_list is None:
            self.hiddenLayer2 = HiddenLayer(
                rng=rng,
                n_in=n_hidden[0],
                n_out=n_hidden[1],
                activation=T.nnet.relu
            )
        else:
            self.hiddenLayer2 = HiddenLayer(
                rng=rng,
                n_in=n_hidden[0],
                n_out=n_hidden[1],
                W_hidden=para_list[2],
                b_hidden=para_list[3],
                activation=T.nnet.relu
            )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        if para_list is None:
            self.outputLayer = LinearLayer(
                n_in=n_hidden[1],
                n_out=n_out,
                rng=rng,
            )
        else:
            self.outputLayer = LinearLayer(
                n_in=n_hidden[1],
                n_out=n_out,
                rng=rng,
                W_lin=para_list[4],
                b_lin=para_list[5]
            )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer1.W).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum()
            + abs(self.hiddenLayer2.W).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.outputLayer.params


    def __call__(self, input, true_result=None):
        hidden_output1 = self.hiddenLayer1(input)
        hidden_output2 = self.hiddenLayer2(hidden_output1)
        output = self.outputLayer(hidden_output2, true_result)
        return output


# convolutional Networks with action transformation(cnn-action-fc)
class CNN3_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, batch_size, hist_window, pred_window, n_kerns, kern_size, n_hidden, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # The first convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size
            )
        else:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size,
                para_list = para_list[0:4]
            )

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        feature_dim = n_kerns[1]* (4-kern_size[0][1]-kern_size[1][1]) * int(hist_window-(kern_size[0][0]-1)-(kern_size[1][0]-1))

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, feature_dim)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(feature_dim,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[4]
            self.ba = para_list[5]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[6:10]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_cnn.params + self.act_trans_params + self.output_net.params
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
                #sliced_input = T.concatenate([sliced_input[:, 1:self.hist_window], result[:, 2*cnt-2:2*cnt-1], sliced_input[:, self.hist_window+1:2*self.hist_window], result[:, 2*cnt-1:2*cnt], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of model output
            conv3_output = self.first_cnn(sliced_input)

                # action transformation
            trans_action = T.dot(sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], self.Wa) + self.ba
            act_transform = conv3_output * trans_action
        
            output = self.output_net(act_transform)
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


# convolutional Networks with action transformation(cnn-action-fc)
class CNN3_gen_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, n_hidden, batch_size, hist_window, pred_window, n_kerns, kern_size, n_act_feature, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # The first convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size
            )
        else:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size,
                para_list = para_list[0:4]
            )

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        feature_dim = n_kerns[1]* (4-kern_size[0][1]-kern_size[1][1]) * int(hist_window-(kern_size[0][0]-1)-(kern_size[1][0]-1))

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(feature_dim, n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature, feature_dim)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(feature_dim,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[4]
            self.Wenc = para_list[5]
            self.Wdec = para_list[6]
            self.ba = para_list[7]

        self.act_trans_params = [self.Wa, self.Wenc, self.Wdec, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[8:12]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_cnn.params + self.act_trans_params + self.output_net.params
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
                #sliced_input = T.concatenate([sliced_input[:, 1:self.hist_window], result[:, 2*cnt-2:2*cnt-1], sliced_input[:, self.hist_window+1:2*self.hist_window], result[:, 2*cnt-1:2*cnt], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of model output
            conv3_output = self.first_cnn(sliced_input)

                # action transformation
            trans_action = T.dot(sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], self.Wa)
            trans_conv3_out = T.dot(conv3_output, self.Wenc)
            act_transform = T.dot( (trans_conv3_out * trans_action), self.Wdec ) + self.ba
        
            output = self.output_net(act_transform)
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


# convolutional Networks with action transformation(cnn-action-fc)
class CNN3_action_FFNN4(object):
    def __init__(self, rng, n_in, n_out, batch_size, hist_window, pred_window, n_kerns, kern_size, n_hidden, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # The first convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size
            )
        else:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size,
                para_list = para_list[0:4]
            )

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        feature_dim = n_kerns[1]* (4-kern_size[0][1]-kern_size[1][1]) * int(hist_window-(kern_size[0][0]-1)-(kern_size[1][0]-1))

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, feature_dim)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(feature_dim,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[4]
            self.ba = para_list[5]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer4(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer4(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[6:12]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_cnn.params + self.act_trans_params + self.output_net.params
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
            conv3_output = self.first_cnn(sliced_input)

                # action transformation
            trans_action = T.dot(sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], self.Wa) + self.ba
            act_transform = conv3_output * trans_action
        
            output = self.output_net(act_transform)
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


# convolutional Networks with action transformation(cnn-action-fc)
class CNN4_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, batch_size, hist_window, pred_window, n_kerns, kern_size, n_hidden, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # The first convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.first_cnn = ConvLayer4(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size
            )
        else:
            self.first_cnn = ConvLayer4(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size,
                para_list = para_list[0:6]
            )

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        feature_dim = n_kerns[2]* (5-kern_size[0][1]-kern_size[1][1]-kern_size[2][1]) * int(hist_window-(kern_size[0][0]-1)-(kern_size[1][0]-1)-(kern_size[2][0]-1))

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, feature_dim)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(feature_dim,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[6]
            self.ba = para_list[7]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[8:12]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_cnn.params + self.act_trans_params + self.output_net.params
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
                #sliced_input = T.concatenate([sliced_input[:, 1:self.hist_window], result[:, 2*cnt-2:2*cnt-1], sliced_input[:, self.hist_window+1:2*self.hist_window], result[:, 2*cnt-1:2*cnt], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of model output
            conv4_output = self.first_cnn(sliced_input)

                # action transformation
            trans_action = T.dot(sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], self.Wa) + self.ba
            act_transform = conv4_output * trans_action
        
            output = self.output_net(act_transform)
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


class twoCNN3_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, n_hidden, batch_size, hist_window, pred_window, n_kerns, kern_size, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # The first convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=n_in,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns[0:2],
                kern_size = kern_size[0:2]
            )
        else:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=n_in,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns[0:2],
                kern_size = kern_size[0:2],
                para_list = para_list[0:4]
            )

        # The second convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.second_cnn = ConvLayer3(
                rng=rng,
                n_in=n_in,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns[2:4],
                kern_size = kern_size[2:4]
            )
        else:
            self.second_cnn = ConvLayer3(
                rng=rng,
                n_in=n_in,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns[2:4],
                kern_size = kern_size[2:4],
                para_list = para_list[4:8]
            )

        # dimension of feature from two CNN
        feature_dim = n_kerns[1]* (4-kern_size[0][1]-kern_size[1][1]) * int(hist_window-(kern_size[0][0]-1)-(kern_size[1][0]-1)) + n_kerns[3]* (4-kern_size[2][1]-kern_size[3][1]) * int(hist_window-(kern_size[2][0]-1)-(kern_size[3][0]-1))

        # action transformation
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(3, feature_dim)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(feature_dim,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[8]
            self.ba = para_list[9]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=feature_dim,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[10:14]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_cnn.params + self.second_cnn.params + self.act_trans_params + self.output_net.params
        self.params_firstway = self.first_cnn.params + self.act_trans_params + self.output_net.params
        self.params_secondway = self.second_cnn.params + self.act_trans_params + self.output_net.params

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
            first_conv3_output = self.first_cnn(sliced_input)
            second_conv3_output = self.second_cnn(sliced_input)

            intermediate_feature = T.concatenate([first_conv3_output, second_conv3_output], axis=1)

                # action transformation
            #trans_action = T.dot(sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], self.Wa) + self.ba
            act_vec = T.concatenate([sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], abs(sliced_input[:, 2*self.hist_window:2*self.hist_window+1])*sliced_input[:, 2*self.hist_window+1:2*self.hist_window+2]], axis=1)
            trans_action = T.dot(act_vec, self.Wa) + self.ba
            act_transform = intermediate_feature * trans_action
        
            output = self.output_net(act_transform)
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


# convolutional Networks with action transformation(cnn-action-linear)
class CNN3_action_linear(object):
    def __init__(self, rng, n_in, n_out, batch_size, hist_window, pred_window, n_kerns, kern_size, para_list=None):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # The first convolutional pooling network
        # returns flattened vector(batch_size, num_features)
        if para_list is None:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size
            )
        else:
            self.first_cnn = ConvLayer3(
                rng=rng,
                n_in=1,
                batch_size = batch_size,
                hist_window = hist_window,
                n_kerns= n_kerns,
                kern_size = kern_size,
                para_list = para_list[0:4]
            )

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        feature_dim = n_kerns[1]* (4-kern_size[0][1]-kern_size[1][1]) * int(hist_window-(kern_size[0][0]-1)-(kern_size[1][0]-1))

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, feature_dim)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(feature_dim,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[4]
            self.ba = para_list[5]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.Wout = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(feature_dim, n_out)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.bout = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_out,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wout = para_list[6]
            self.bout = para_list[7]
        self.output_params = [self.Wout, self.bout]

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.first_cnn.params + self.act_trans_params + self.output_params
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
                #sliced_input = T.concatenate([sliced_input[:, 1:self.hist_window], result[:, 2*cnt-2:2*cnt-1], sliced_input[:, self.hist_window+1:2*self.hist_window], result[:, 2*cnt-1:2*cnt], input[:, 2*(self.hist_window+cnt):2*(self.hist_window+cnt+1)]], axis=1)

            # computation of model output
            conv3_output = self.first_cnn(sliced_input)

                # action transformation
            trans_action = T.dot(sliced_input[:,(2*self.hist_window):(2*self.hist_window+2)], self.Wa) + self.ba
            act_transform = conv3_output * trans_action

            output = T.dot(act_transform, self.Wout) + self.bout
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
