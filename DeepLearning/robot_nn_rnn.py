
import numpy as np

import theano
import theano.tensor as T

def weight_init(N, rand_state):
    R = np.asarray(rand_state.randn(N, N), dtype=theano.config.floatX)
    A = np.identity(N, dtype=theano.config.floatX) + (1.0/N)*np.dot(np.transpose(R), R)
    e = max(np.linalg.eig(A)[0])
    return A/e

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


# class of Gated Recurrent Unit
class GRU(object):
    def __init__(self, rng, n_in, n_hidden, para_list=None, activation=T.tanh):
        if para_list is None:
            Wreset_value = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(n_in+n_hidden, n_hidden)), dtype=theano.config.floatX)
            breset_value = np.zeros((n_hidden,), dtype=theano.config.floatX)

            Wupdate_value = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(n_in+n_hidden, n_hidden)), dtype=theano.config.floatX)
            bupdate_value = np.zeros((n_hidden,), dtype=theano.config.floatX)

            Wh_value = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(n_in+n_hidden, n_hidden)), dtype=theano.config.floatX)
            bh_value = np.zeros((n_hidden,), dtype=theano.config.floatX)

            self.W_reset = theano.shared(value=Wreset_value, name='W_reset', borrow=True)
            self.b_reset = theano.shared(value=breset_value, name='b_reset', borrow=True)
            self.W_update = theano.shared(value=Wupdate_value, name='W_update', borrow=True)
            self.b_update = theano.shared(value=bupdate_value, name='b_update', borrow=True)
            self.W_h = theano.shared(value=Wh_value, name='W_h', borrow=True)
            self.b_h = theano.shared(value=bh_value, name='b_h', borrow=True)

        else:
            self.W_reset = para_list[0]
            self.b_reset = para_list[1]
            self.W_update = para_list[2]
            self.b_update = para_list[3]
            self.W_h = para_list[4]
            self.b_h = para_list[5]

        self.activation = activation
        self.params = [self.W_reset, self.b_reset, self.W_update, self.b_update, self.W_h, self.b_h]

    def __call__(self, x, h):
        #h_x = T.concatenate([h, x], axis=1)
        #reset_gate = T.nnet.sigmoid(T.dot(h_x, self.W_reset) + self.b_reset)
        #update_gate = T.nnet.sigmoid(T.dot(h_x, self.W_update) + self.b_update)

        reset_gate = T.nnet.sigmoid(T.dot(T.concatenate([h, x], axis=1), self.W_reset) + self.b_reset)
        update_gate = T.nnet.sigmoid(T.dot(T.concatenate([h, x], axis=1), self.W_update) + self.b_update)

        reset_h = reset_gate * h
        reset_h_x = T.concatenate([reset_h, x], axis=1)
        #intermediate_hidden = T.tanh(T.dot(reset_h_x, self.W_h) + self.b_h)
        intermediate_hidden = self.activation(T.dot(reset_h_x, self.W_h) + self.b_h)

        new_h = (1.0-update_gate) * h + update_gate * intermediate_hidden
        return new_h


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
        hidden_output2 = self.hiddenLayer2(input)
        output = self.outputLayer(hidden_output2, true_result)
        return output


# 1 layer Recurrent Networks w/ action transition - 3 layers FFNN
class RNN_w_action_FFNN3(object):
    def __init__(self, rng, n_state, n_hidden, n_in, n_out, batch_size, hist_window, pred_window, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_out, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            '''
            self.Wtrans = theano.shared(weight_init(n_state, rng), borrow=True)
            self.Wobs = theano.shared(
                np.asarray(
                    rng.randn(n_out, n_state)*(1.0/n_state),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans = para_list[0]
            self.Wobs = para_list[1]

        self.rnn_params = [self.Wtrans, self.Wobs]

        # Parameters related to action transformation
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(rng.uniform(low=-0.005, high=0.005, size=(n_in, n_state)),
                dtype=theano.config.floatX), name='W_action', borrow=True
            )
            self.ba = theano.shared(
                np.asarray(rng.uniform(low=-0.005, high=0.005, size=(n_state,)),
                dtype=theano.config.floatX), name='b_action', borrow=True
            )
        else:
            self.Wa = para_list[2]
            self.ba = para_list[3]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change hidden state to wheel speeds
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[4:8]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_in = n_in
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans).sum() + abs(self.Wobs).sum()
        self.rnn_reg_L2 = (self.Wtrans ** 2).sum() + (self.Wobs ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, cmd_input, init_model_output = data_x[:, 0:self.n_out*(self.hist_window-1)], data_x[:, self.n_out*self.hist_window:self.n_out*self.hist_window+self.n_in*self.pred_window], data_x[:, self.n_out*(self.hist_window-1):self.n_out*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(obs_speed[:, self.n_out*cnt:self.n_out*(cnt+1)], self.Wobs), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            cmd_input_multiplier = T.nnet.hard_sigmoid(T.dot(cmd_input[:, self.n_in*cnt:self.n_in*(cnt+1)], self.Wa) + self.ba)
            if cnt < 1:
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state, self.Wtrans)*cmd_input_multiplier + T.dot(init_model_output, self.Wobs), self.leaky_relu_alpha )
            elif not teach_force or true_result is None :
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state, self.Wtrans)*cmd_input_multiplier + T.dot(result[:, self.n_out*(cnt-1):self.n_out*cnt], self.Wobs), self.leaky_relu_alpha )
            else:
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state, self.Wtrans)*cmd_input_multiplier + T.dot(true_result[:, self.n_out*(cnt-1):self.n_out*cnt], self.Wobs), self.leaky_relu_alpha )

            # output transformation
            output = self.output_net(internal_state)

            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, self.n_out*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# 2 layers Recurrent Networks w/ action transition - 3 layers FFNN
class RNN2_w_action_FFNN3(object):
    def __init__(self, rng, n_state, n_hidden, n_in, n_out, batch_size, hist_window, pred_window, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2]

        # Parameters related to action transformation
        if para_list is None:
            self.Wa1 = theano.shared(
                np.asarray(rng.uniform(low=-0.005, high=0.005, size=(n_in, n_state[0])),
                dtype=theano.config.floatX), name='W_action1', borrow=True
            )
            self.ba1 = theano.shared(
                np.asarray(rng.uniform(low=-0.005, high=0.005, size=(n_state[0],)),
                dtype=theano.config.floatX), name='b_action1', borrow=True
            )
            self.Wa2 = theano.shared(
                np.asarray(rng.uniform(low=-0.005, high=0.005, size=(n_in, n_state[1])),
                dtype=theano.config.floatX), name='W_action2', borrow=True
            )
            self.ba2 = theano.shared(
                np.asarray(rng.uniform(low=-0.005, high=0.005, size=(n_state[1],)),
                dtype=theano.config.floatX), name='b_action2', borrow=True
            )
        else:
            self.Wa1 = para_list[4]
            self.ba1 = para_list[5]
            self.Wa2 = para_list[6]
            self.ba2 = para_list[7]

        self.act_trans_params = [self.Wa1, self.ba1, self.Wa2, self.ba2]

        # network to change hidden state to wheel speeds
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[8:12]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_in = n_in
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, cmd_input, init_model_output = data_x[:, 0:self.n_out*(self.hist_window-1)], data_x[:, self.n_out*self.hist_window:self.n_out*self.hist_window+self.n_in*self.pred_window], data_x[:, self.n_out*(self.hist_window-1):self.n_out*self.hist_window]

        internal_state1, internal_state2 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, self.n_out*cnt:self.n_out*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            cmd_input_multiplier1, cmd_input_multiplier2 = T.nnet.hard_sigmoid(T.dot(cmd_input[:, self.n_in*cnt:self.n_in*(cnt+1)], self.Wa1) + self.ba1), T.nnet.hard_sigmoid(T.dot(cmd_input[:, self.n_in*cnt:self.n_in*(cnt+1)], self.Wa2) + self.ba2)
            if cnt < 1:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state1, self.Wtrans1)*cmd_input_multiplier1 + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha )
            elif not teach_force or true_result is None :
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state1, self.Wtrans1)*cmd_input_multiplier1 + T.dot(result[:, self.n_out*(cnt-1):self.n_out*cnt], self.Wobs1), self.leaky_relu_alpha )
            else:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state1, self.Wtrans1)*cmd_input_multiplier1 + T.dot(true_result[:, self.n_out*(cnt-1):self.n_out*cnt], self.Wobs1), self.leaky_relu_alpha )

            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation( T.dot(internal_state2, self.Wtrans2)*cmd_input_multiplier2 + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha )

            # output transformation
            output = self.output_net(internal_state2)

            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, self.n_out*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]






























# recurrent Networks with action transformation(rnn-action-fc)
class RNN_action_FFNN3(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            '''
            self.Wtrans = theano.shared(weight_init(n_state, rng), borrow=True)
            self.Wobs = theano.shared(
                np.asarray(
                    rng.randn(2, n_state)*(1.0/n_state),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans = para_list[0]
            self.Wobs = para_list[1]

        self.rnn_params = [self.Wtrans, self.Wobs]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[2]
            self.ba = para_list[3]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[4:8]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans).sum() + abs(self.Wobs).sum()
        self.rnn_reg_L2 = (self.Wtrans ** 2).sum() + (self.Wobs ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            if cnt < 1:
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(init_model_output, self.Wobs), self.leaky_relu_alpha)
            elif not teach_force or true_result is None :
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs), self.leaky_relu_alpha)
            else:
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs), self.leaky_relu_alpha)

            # action transformation
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_transform = internal_state * trans_action

            output = self.output_net(act_transform)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(2 rnn-action-fc)
class RNN2_action_FFNN3(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''
            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
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
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[6:10]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            if cnt < 1:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)
            elif not teach_force or true_result is None :
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)
            else:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)

            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)

            # action transformation
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_transform = internal_state2 * trans_action

            output = self.output_net(act_transform)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(3 rnn-action-fc)
class RNN3_action_FFNN3(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[2], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[1], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans3 = theano.shared(weight_init(n_state[2], rng), borrow=True)
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.randn(n_state[1], n_state[2])*(1.0/n_state[2]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]
            self.Wtrans3 = para_list[4]
            self.Wobs3 = para_list[5]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2, self.Wtrans3, self.Wobs3]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[2],)),
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
                n_in=n_state[2],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[2],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[8:12]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum() + abs(self.Wtrans3).sum() + abs(self.Wobs3).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum() + (self.Wtrans3 ** 2).sum() + (self.Wobs3 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2, internal_state3 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[2]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            if cnt < 1:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)
            elif not teach_force or true_result is None :
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)
            else:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)

            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)

            # action transformation
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_transform = internal_state3 * trans_action

            output = self.output_net(act_transform)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(rnn-action-fc)
class RNN_gen_action_FFNN3(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans = theano.shared(weight_init(n_state, rng), borrow=True)
            self.Wobs = theano.shared(
                np.asarray(
                    rng.randn(2, n_state)*(1.0/n_state),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans = para_list[0]
            self.Wobs = para_list[1]

        self.rnn_params = [self.Wtrans, self.Wobs]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state, n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature, n_state)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[2]
            self.Wenc = para_list[3]
            self.Wdec = para_list[4]
            self.ba = para_list[5]

        self.act_trans_params = [self.Wa, self.Wenc, self.Wdec, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[6:10]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans).sum() + abs(self.Wobs).sum()
        self.rnn_reg_L2 = (self.Wtrans ** 2).sum() + (self.Wobs ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            if cnt < 1:
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(init_model_output, self.Wobs), self.leaky_relu_alpha)
            elif not teach_force or true_result is None :
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs), self.leaky_relu_alpha)
            else:
                internal_state = self.leaky_integration * internal_state + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state, self.Wtrans) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs), self.leaky_relu_alpha)

            # action transformation
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_internal_state = T.dot(internal_state, self.Wenc)
            act_transform = T.dot( (trans_internal_state * trans_action), self.Wdec ) + self.ba

            output = self.output_net(act_transform)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(2 rnn-action-fc)
class RNN2_gen_action_FFNN3(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature, n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
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
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[8:12]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            if cnt < 1:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)
            elif not teach_force or true_result is None :
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)
            else:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)

            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)

            # action transformation
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_internal_state = T.dot(internal_state2, self.Wenc)
            act_transform = T.dot( (trans_internal_state * trans_action), self.Wdec ) + self.ba

            output = self.output_net(act_transform)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(3 rnn-action-fc)
class RNN3_gen_action_FFNN3(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[2], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[1], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans3 = theano.shared(weight_init(n_state[2], rng), borrow=True)
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.randn(n_state[1], n_state[2])*(1.0/n_state[2]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]
            self.Wtrans3 = para_list[4]
            self.Wobs3 = para_list[5]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2, self.Wtrans3, self.Wobs3]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[2], n_act_feature)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature, n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[2],)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa = para_list[6]
            self.Wenc = para_list[7]
            self.Wdec = para_list[8]
            self.ba = para_list[9]

        self.act_trans_params = [self.Wa, self.Wenc, self.Wdec, self.ba]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[2],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[2],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[10:14]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum() + abs(self.Wtrans3).sum() + abs(self.Wobs3).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum() + (self.Wtrans3 ** 2).sum() + (self.Wobs3 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2, internal_state3 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[2]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            if cnt < 1:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)
            elif not teach_force or true_result is None :
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)
            else:
                internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)

            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)

            # action transformation
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_internal_state = T.dot(internal_state3, self.Wenc)
            act_transform = T.dot( (trans_internal_state * trans_action), self.Wdec ) + self.ba

            output = self.output_net(act_transform)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]








# recurrent Networks with action transformation(2 rnn-action-fc)
class RNN2_gen_action_FFNN3_tmp(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        if para_list is None:
            self.Wa1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[0], n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[0],)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wa2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa1 = para_list[4]
            self.Wenc1 = para_list[5]
            self.Wdec1 = para_list[6]
            self.ba1 = para_list[7]
            self.Wa2 = para_list[8]
            self.Wenc2 = para_list[9]
            self.Wdec2 = para_list[10]
            self.ba2 = para_list[11]

        self.act_trans_params = [self.Wa1, self.Wenc1, self.Wdec1, self.ba1, self.Wa2, self.Wenc2, self.Wdec2, self.ba2]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[12:16]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd_past, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:4*self.hist_window-2], data_x[:, 4*self.hist_window-2:4*self.hist_window+2*self.pred_window-2], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            trans_action1 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa1)
            trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            trans_action1 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa1)
            if cnt < 1:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            elif not teach_force or true_result is None:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            else:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)

            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

            output = self.output_net(internal_state2)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(3 rnn-action-fc)
class RNN3_gen_action_FFNN3_tmp(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_hidden, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[2], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[1], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans3 = theano.shared(weight_init(n_state[2], rng), borrow=True)
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.randn(n_state[1], n_state[2])*(1.0/n_state[2]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]
            self.Wtrans3 = para_list[4]
            self.Wobs3 = para_list[5]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2, self.Wtrans3, self.Wobs3]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[0], n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[0],)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wa2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wa3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[2], n_act_feature[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[2], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[2],)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa1 = para_list[6]
            self.Wenc1 = para_list[7]
            self.Wdec1 = para_list[8]
            self.ba1 = para_list[9]
            self.Wa2 = para_list[10]
            self.Wenc2 = para_list[11]
            self.Wdec2 = para_list[12]
            self.ba2 = para_list[13]
            self.Wa3 = para_list[14]
            self.Wenc3 = para_list[15]
            self.Wdec3 = para_list[16]
            self.ba3 = para_list[17]

        self.act_trans_params = [self.Wa1, self.Wenc1, self.Wdec1, self.ba1, self.Wa2, self.Wenc2, self.Wdec2, self.ba2, self.Wa3, self.Wenc3, self.Wdec3, self.ba3]

        # network to change features to wheels' speed
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[2],
                n_hidden=n_hidden,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[2],
                n_hidden=n_hidden,
                n_out=n_out,
                para_list = para_list[18:22]
            )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum() + abs(self.Wtrans3).sum() + abs(self.Wobs3).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum() + (self.Wtrans3 ** 2).sum() + (self.Wobs3 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd_past, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:4*self.hist_window-2], data_x[:, 4*self.hist_window-2:4*self.hist_window+2*self.pred_window-2], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2, internal_state3 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[2]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            trans_action1 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa1)
            trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

            trans_action3 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa3)
            trans_internal_state3 = T.dot( (self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)), self.Wenc3)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * (T.dot( (trans_internal_state3 * trans_action3), self.Wdec3 ) + self.ba3)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            trans_action1 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa1)
            if cnt < 1:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            elif not teach_force or true_result is None:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            else:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)

            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

            trans_action3 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa3)
            trans_internal_state3 = T.dot( (self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)), self.Wenc3)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * (T.dot( (trans_internal_state3 * trans_action3), self.Wdec3 ) + self.ba3)

            output = self.output_net(internal_state3)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]






# recurrent Networks with action transformation(2 rnn-action-linear)
class RNN2_gen_action_linear(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)
        if para_list is None:
            self.Wa1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[0], n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[0],)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wa2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa1 = para_list[4]
            self.Wenc1 = para_list[5]
            self.Wdec1 = para_list[6]
            self.ba1 = para_list[7]
            self.Wa2 = para_list[8]
            self.Wenc2 = para_list[9]
            self.Wdec2 = para_list[10]
            self.ba2 = para_list[11]

        self.act_trans_params = [self.Wa1, self.Wenc1, self.Wdec1, self.ba1, self.Wa2, self.Wenc2, self.Wdec2, self.ba2]

        # network to change features to wheels' speed
        if para_list is None:
            self.Wout = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_out)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.bout = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_out,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wout = para_list[12]
            self.bout = para_list[13]
        self.output_params = [self.Wout, self.bout]

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd_past, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:4*self.hist_window-2], data_x[:, 4*self.hist_window-2:4*self.hist_window+2*self.pred_window-2], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            trans_action1 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa1)
            trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            trans_action1 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa1)
            if cnt < 1:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            elif not teach_force or true_result is None:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            else:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)

            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

            output = T.dot( internal_state2, self.Wout ) + self.bout
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# recurrent Networks with action transformation(3 rnn-action-linear)
class RNN3_gen_action_linear(object):
    def __init__(self, rng, n_state, n_out, batch_size, hist_window, pred_window, n_act_feature, para_list=None, rnn_activation=T.nnet.relu, leaky_relu_alpha=0, leaky_integration=0):

        if para_list is None:
            '''
            self.Wtrans1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(2, n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[0], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )

            self.Wtrans3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.05, high=0.05, size=(n_state[2], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.03, high=0.03, size=(n_state[1], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            '''

            self.Wtrans1 = theano.shared(weight_init(n_state[0], rng), borrow=True)
            self.Wobs1 = theano.shared(
                np.asarray(
                    rng.randn(2, n_state[0])*(1.0/n_state[0]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans2 = theano.shared(weight_init(n_state[1], rng), borrow=True)
            self.Wobs2 = theano.shared(
                np.asarray(
                    rng.randn(n_state[0], n_state[1])*(1.0/n_state[1]),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wtrans3 = theano.shared(weight_init(n_state[2], rng), borrow=True)
            self.Wobs3 = theano.shared(
                np.asarray(
                    rng.randn(n_state[1], n_state[2])*(1.0/n_state[2]),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wtrans1 = para_list[0]
            self.Wobs1 = para_list[1]
            self.Wtrans2 = para_list[2]
            self.Wobs2 = para_list[3]
            self.Wtrans3 = para_list[4]
            self.Wobs3 = para_list[5]

        self.rnn_params = [self.Wtrans1, self.Wobs1, self.Wtrans2, self.Wobs2, self.Wtrans3, self.Wobs3]

        # feature-action transformation
        # use element-wise multiplication with affine transformed action vector
        # Wf .* (Wa * action_vector + ba)

        if para_list is None:
            self.Wa1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[0], n_act_feature[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[0], n_state[0])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba1 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[0],)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wa2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_act_feature[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[1], n_state[1])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba2 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wa3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(2, n_act_feature[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wenc3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[2], n_act_feature[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.Wdec3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_act_feature[2], n_state[2])),
                    dtype=theano.config.floatX), borrow=True
            )
            self.ba3 = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_state[2],)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wa1 = para_list[6]
            self.Wenc1 = para_list[7]
            self.Wdec1 = para_list[8]
            self.ba1 = para_list[9]
            self.Wa2 = para_list[10]
            self.Wenc2 = para_list[11]
            self.Wdec2 = para_list[12]
            self.ba2 = para_list[13]
            self.Wa3 = para_list[14]
            self.Wenc3 = para_list[15]
            self.Wdec3 = para_list[16]
            self.ba3 = para_list[17]

        self.act_trans_params = [self.Wa1, self.Wenc1, self.Wdec1, self.ba1, self.Wa2, self.Wenc2, self.Wdec2, self.ba2, self.Wa3, self.Wenc3, self.Wdec3, self.ba3]

        # network to change features to wheels' speed
        if para_list is None:
            self.Wout = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.01, high=0.01, size=(n_state[2], n_out)),
                    dtype=theano.config.floatX), borrow=True
            )
            self.bout = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=(n_out,)),
                    dtype=theano.config.floatX), borrow=True
            )
        else:
            self.Wout = para_list[18]
            self.bout = para_list[19]
        self.output_params = [self.Wout, self.bout]

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.rnn_params + self.act_trans_params + self.output_params
        self.batch_size = batch_size
        self.n_state = n_state
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.rnn_activation = rnn_activation
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_integration = leaky_integration

        self.rnn_reg_L1 = abs(self.Wtrans1).sum() + abs(self.Wobs1).sum() + abs(self.Wtrans2).sum() + abs(self.Wobs2).sum() + abs(self.Wtrans3).sum() + abs(self.Wobs3).sum()
        self.rnn_reg_L2 = (self.Wtrans1 ** 2).sum() + (self.Wobs1 ** 2).sum() + (self.Wtrans2 ** 2).sum() + (self.Wobs2 ** 2).sum() + (self.Wtrans3 ** 2).sum() + (self.Wobs3 ** 2).sum()

    def __call__(self, data_x, true_result=None, rnn_reg_out=False, teach_force=False, error_weight=None):
        obs_speed, joystick_cmd_past, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:4*self.hist_window-2], data_x[:, 4*self.hist_window-2:4*self.hist_window+2*self.pred_window-2], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1, internal_state2, internal_state3 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX), T.zeros((self.batch_size, self.n_state[2]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            trans_action1 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa1)
            trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
            #trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(obs_speed[:, 2*cnt:2*(cnt+1)], self.Wobs1))), self.Wenc1)
            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            #trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2))), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

            trans_action3 = T.dot(joystick_cmd_past[:, 2*cnt:2*(cnt+1)], self.Wa3)
            trans_internal_state3 = T.dot( (self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)), self.Wenc3)
            #trans_internal_state3 = T.dot( (self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3))), self.Wenc3)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * (T.dot( (trans_internal_state3 * trans_action3), self.Wdec3 ) + self.ba3)

        for cnt in range(self.pred_window):
            # updating latent feature vector
            trans_action1 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa1)
            if cnt < 1:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
                #trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(init_model_output, self.Wobs1))), self.Wenc1)
            elif not teach_force or true_result is None:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
                #trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(result[:, 2*(cnt-1):2*cnt], self.Wobs1))), self.Wenc1)
            else:
                trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1), self.leaky_relu_alpha)), self.Wenc1)
                #trans_internal_state1 = T.dot( (self.rnn_activation(T.dot(internal_state1, self.Wtrans1) + T.dot(true_result[:, 2*(cnt-1):2*cnt], self.Wobs1))), self.Wenc1)

            internal_state1 = self.leaky_integration * internal_state1 + (1-self.leaky_integration) * (T.dot( (trans_internal_state1 * trans_action1), self.Wdec1 ) + self.ba1)

            trans_action2 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa2)
            trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2), self.leaky_relu_alpha)), self.Wenc2)
            #trans_internal_state2 = T.dot( (self.rnn_activation(T.dot(internal_state2, self.Wtrans2) + T.dot(internal_state1, self.Wobs2))), self.Wenc2)
            internal_state2 = self.leaky_integration * internal_state2 + (1-self.leaky_integration) * (T.dot( (trans_internal_state2 * trans_action2), self.Wdec2 ) + self.ba2)

            trans_action3 = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa3)
            trans_internal_state3 = T.dot( (self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3), self.leaky_relu_alpha)), self.Wenc3)
            #trans_internal_state3 = T.dot( (self.rnn_activation(T.dot(internal_state3, self.Wtrans3) + T.dot(internal_state2, self.Wobs3))), self.Wenc3)
            internal_state3 = self.leaky_integration * internal_state3 + (1-self.leaky_integration) * (T.dot( (trans_internal_state3 * trans_action3), self.Wdec3 ) + self.ba3)

            output = T.dot( internal_state3, self.Wout ) + self.bout
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if error_weight is None:
            error_weight = T.ones((self.batch_size, 2*self.pred_window), dtype=theano.config.floatX)

        if true_result is None:
            return result
        else:
            if true_result.ndim != result.ndim:
                raise TypeError(
                    'y should have the same shape as model output',
                    ('y', true_result.type, 'output', result.type)
                )
            diff = abs(result - true_result) * error_weight
            if rnn_reg_out:
                return [diff.sum(), (diff**2).sum(), diff.max(), self.rnn_reg_L1, self.rnn_reg_L2]
            else:
                return [diff.sum(), (diff**2).sum(), diff.max()]


# class of GRU->action_transformation->FFNN3 model
class GRU_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, n_state, n_hidden_ffnn3, batch_size, hist_window, pred_window, para_list=None, activation=T.tanh):
        # Gated Recurrent Layer for memory of the model
        if para_list is None:
            self.memory_struct = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state,
                activation=activation
            )
        else:
            self.memory_struct = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state,
                activation=activation,
                para_list=para_list[0:6]
            )

        # Parameters related to action transformation
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(2, n_state)),
                dtype=theano.config.floatX), name='W_action', borrow=True
            )
            self.ba = theano.shared(
                np.asarray(rng.uniform(low=-0.5, high=0.5, size=(n_state,)),
                dtype=theano.config.floatX), name='b_action', borrow=True
            )
        else:
            self.Wa = para_list[6]
            self.ba = para_list[7]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change hidden state to wheel speeds
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden_ffnn3,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden_ffnn3,
                n_out=n_out,
                para_list = para_list[8:12]
            )

        self.params = self.memory_struct.params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.n_state = n_state

    # long-term simulation on mini-batch data (NOT for Training)
    def predict_batch(self, data_x, data_y=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.memory_struct(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state = self.memory_struct(init_model_output, internal_state)
            else:
                internal_state = self.memory_struct(result[:, 2*(cnt-1):2*cnt], internal_state)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_h_trans = trans_action * internal_state

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if data_y is None:
            return result
        else:
            diff = abs(result - data_y)
            error = [diff.sum(), (diff**2).sum(), diff.max()]
            return error

    # long-term simulation on mini-batch data (for Training)
    def predict_batch_for_train(self, data_x, data_y):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.memory_struct(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state = self.memory_struct(init_model_output, internal_state)
            else:
                internal_state = self.memory_struct(result[:, 2*(cnt-1):2*cnt], internal_state)
            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_h_trans = trans_action * internal_state

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        diff = abs(result - data_y)
        error = [diff.sum(), (diff**2).sum(), diff.max()]
        return error

# class of GRU->generalized action_transformation->FFNN3 model
class GRU_gen_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, n_state, n_hidden_ffnn3, n_act, batch_size, hist_window, pred_window, para_list=None, activation=T.tanh):
        # Gated Recurrent Layer for memory of the model
        if para_list is None:
            self.memory_struct = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state,
                activation=activation
            )
        else:
            self.memory_struct = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state,
                activation=activation,
                para_list=para_list[0:6]
            )

        # Parameters related to action transformation
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(2, n_act)),
                dtype=theano.config.floatX), borrow=True
            )
            self.Wenc = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(n_state, n_act)),
                dtype=theano.config.floatX), borrow=True
            )
            self.Wdec = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(n_act, n_state)),
                dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(rng.uniform(low=-0.5, high=0.5, size=(n_state,)),
                dtype=theano.config.floatX), name='b_action', borrow=True
            )
        else:
            self.Wa = para_list[6]
            self.Wenc = para_list[7]
            self.Wdec = para_list[8]
            self.ba = para_list[9]

        self.act_trans_params = [self.Wa, self.Wenc, self.Wdec, self.ba]

        # network to change hidden state to wheel speeds
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden_ffnn3,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state,
                n_hidden=n_hidden_ffnn3,
                n_out=n_out,
                para_list = para_list[10:14]
            )

        self.params = self.memory_struct.params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.n_state = n_state

    # long-term simulation on mini-batch data (NOT for Training)
    def predict_batch(self, data_x, data_y=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.memory_struct(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state = self.memory_struct(init_model_output, internal_state)
            else:
                internal_state = self.memory_struct(result[:, 2*(cnt-1):2*cnt], internal_state)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_state = T.dot(internal_state, self.Wenc)
            act_h_trans = T.dot( (trans_action * trans_state), self.Wdec ) + self.ba

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if data_y is None:
            return result
        else:
            diff = abs(result - data_y)
            error = [diff.sum(), (diff**2).sum(), diff.max()]
            return error

    # long-term simulation on mini-batch data (for Training)
    def predict_batch_for_train(self, data_x, data_y):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state = T.zeros((self.batch_size, self.n_state), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state = self.memory_struct(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state = self.memory_struct(init_model_output, internal_state)
            else:
                internal_state = self.memory_struct(result[:, 2*(cnt-1):2*cnt], internal_state)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_state = T.dot(internal_state, self.Wenc)
            act_h_trans = T.dot( (trans_action * trans_state), self.Wdec ) + self.ba

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        diff = abs(result - data_y)
        error = [diff.sum(), (diff**2).sum(), diff.max()]
        return error



# class of 2 layers of GRU->action_transformation->FFNN3 model
class GRU2_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, n_state, n_hidden_ffnn3, batch_size, hist_window, pred_window, para_list=None, activation=T.tanh):
        # 1st layer of Gated Recurrent Unit for memory of the model
        if para_list is None:
            self.memory_struct1 = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state[0],
                activation=activation
            )
        else:
            self.memory_struct1 = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state[0],
                activation=activation,
                para_list=para_list[0:6]
            )

        # 2nd layer of Gated Recurrent Unit for memory of the model
        if para_list is None:
            self.memory_struct2 = GRU(
                rng=rng,
                n_in=n_state[0],
                n_hidden=n_state[1],
                activation=activation
            )
        else:
            self.memory_struct2 = GRU(
                rng=rng,
                n_in=n_state[0],
                n_hidden=n_state[1],
                activation=activation,
                para_list=para_list[6:12]
            )

        # Parameters related to action transformation
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(2, n_state[1])),
                dtype=theano.config.floatX), name='W_action', borrow=True
            )
            self.ba = theano.shared(
                np.asarray(rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
                dtype=theano.config.floatX), name='b_action', borrow=True
            )
        else:
            self.Wa = para_list[12]
            self.ba = para_list[13]

        self.act_trans_params = [self.Wa, self.ba]

        # network to change hidden state to wheel speeds
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden_ffnn3,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden_ffnn3,
                n_out=n_out,
                para_list = para_list[14:18]
            )

        self.params = self.memory_struct1.params + self.memory_struct2.params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.n_state = n_state

    # long-term simulation on mini-batch data (NOT for Training)
    def predict_batch(self, data_x, data_y=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX)
        internal_state2 = T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.memory_struct1(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state1 = self.memory_struct1(init_model_output, internal_state1)
            else:
                internal_state1 = self.memory_struct1(result[:, 2*(cnt-1):2*cnt], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_h_trans = trans_action * internal_state2

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if data_y is None:
            return result
        else:
            diff = abs(result - data_y)
            error = [diff.sum(), (diff**2).sum(), diff.max()]
            return error

    # long-term simulation on mini-batch data (for Training)
    def predict_batch_for_train(self, data_x, data_y):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX)
        internal_state2 = T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.memory_struct1(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state1 = self.memory_struct1(init_model_output, internal_state1)
            else:
                internal_state1 = self.memory_struct1(result[:, 2*(cnt-1):2*cnt], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa) + self.ba
            act_h_trans = trans_action * internal_state2

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        diff = abs(result - data_y)
        error = [diff.sum(), (diff**2).sum(), diff.max()]
        return error


# class of 2 layers of GRU->generalized action_transformation->FFNN3 model
class GRU2_gen_action_FFNN3(object):
    def __init__(self, rng, n_in, n_out, n_state, n_hidden_ffnn3, n_act, batch_size, hist_window, pred_window, para_list=None, activation=T.tanh):
        # 1st layer of Gated Recurrent Unit for memory of the model
        if para_list is None:
            self.memory_struct1 = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state[0],
                activation=activation
            )
        else:
            self.memory_struct1 = GRU(
                rng=rng,
                n_in=n_in,
                n_hidden=n_state[0],
                activation=activation,
                para_list=para_list[0:6]
            )

        # 2nd layer of Gated Recurrent Unit for memory of the model
        if para_list is None:
            self.memory_struct2 = GRU(
                rng=rng,
                n_in=n_state[0],
                n_hidden=n_state[1],
                activation=activation
            )
        else:
            self.memory_struct2 = GRU(
                rng=rng,
                n_in=n_state[0],
                n_hidden=n_state[1],
                activation=activation,
                para_list=para_list[6:12]
            )

        # Parameters related to action transformation
        if para_list is None:
            self.Wa = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(2, n_act)),
                dtype=theano.config.floatX), borrow=True
            )
            self.Wenc = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(n_state[1], n_act)),
                dtype=theano.config.floatX), borrow=True
            )
            self.Wdec = theano.shared(
                np.asarray(rng.uniform(low=-0.01, high=0.01, size=(n_act, n_state[1])),
                dtype=theano.config.floatX), borrow=True
            )
            self.ba = theano.shared(
                np.asarray(rng.uniform(low=-0.5, high=0.5, size=(n_state[1],)),
                dtype=theano.config.floatX), name='b_action', borrow=True
            )
        else:
            self.Wa = para_list[12]
            self.Wenc = para_list[13]
            self.Wdec = para_list[14]
            self.ba = para_list[15]

        self.act_trans_params = [self.Wa, self.Wenc, self.Wdec, self.ba]

        # network to change hidden state to wheel speeds
        if para_list is None:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden_ffnn3,
                n_out=n_out
            )
        else:
            self.output_net = FeedForwardLayer3(
                rng=rng,
                n_in=n_state[1],
                n_hidden=n_hidden_ffnn3,
                n_out=n_out,
                para_list = para_list[16:20]
            )

        self.params = self.memory_struct1.params + self.memory_struct2.params + self.act_trans_params + self.output_net.params
        self.batch_size = batch_size
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.n_state = n_state

    # long-term simulation on mini-batch data (NOT for Training)
    def predict_batch(self, data_x, data_y=None):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX)
        internal_state2 = T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.memory_struct1(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state1 = self.memory_struct1(init_model_output, internal_state1)
            else:
                internal_state1 = self.memory_struct1(result[:, 2*(cnt-1):2*cnt], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_state = T.dot(internal_state2, self.Wenc)
            act_h_trans = T.dot( (trans_action * trans_state), self.Wdec ) + self.ba

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        if data_y is None:
            return result
        else:
            diff = abs(result - data_y)
            error = [diff.sum(), (diff**2).sum(), diff.max()]
            return error

    # long-term simulation on mini-batch data (for Training)
    def predict_batch_for_train(self, data_x, data_y):
        obs_speed, joystick_cmd, init_model_output = data_x[:, 0:2*(self.hist_window-1)], data_x[:, 2*self.hist_window:2*(self.hist_window+self.pred_window)], data_x[:, 2*(self.hist_window-1):2*self.hist_window]

        internal_state1 = T.zeros((self.batch_size, self.n_state[0]), dtype=theano.config.floatX)
        internal_state2 = T.zeros((self.batch_size, self.n_state[1]), dtype=theano.config.floatX)
        for cnt in range(self.hist_window-1):
            internal_state1 = self.memory_struct1(obs_speed[:, 2*cnt:2*(cnt+1)], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

        for cnt in range(self.pred_window):
            if cnt < 1:
                internal_state1 = self.memory_struct1(init_model_output, internal_state1)
            else:
                internal_state1 = self.memory_struct1(result[:, 2*(cnt-1):2*cnt], internal_state1)
            internal_state2 = self.memory_struct2(internal_state1, internal_state2)

            trans_action = T.dot(joystick_cmd[:, 2*cnt:2*(cnt+1)], self.Wa)
            trans_state = T.dot(internal_state2, self.Wenc)
            act_h_trans = T.dot( (trans_action * trans_state), self.Wdec ) + self.ba

            output = self.output_net(act_h_trans)
            if cnt < 1:
                result = output
            else:
                result = T.concatenate([result, output], axis=1)

        diff = abs(result - data_y)
        error = [diff.sum(), (diff**2).sum(), diff.max()]
        return error
