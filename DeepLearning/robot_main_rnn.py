# Python code to make multilayer neural network for robot modeling

import os
import sys
import timeit
import six.moves.cPickle as pickle
from scipy.io import savemat, loadmat

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import theano
import theano.tensor as T

#from robot_data_cnn import load_data, process_data
#from robot_data import load_data_vulcan_rnn
from robot_data import load_data
from robot_nn_rnn import RNN_action_FFNN3, RNN2_action_FFNN3, RNN3_action_FFNN3, RNN_gen_action_FFNN3, RNN2_gen_action_FFNN3, RNN3_gen_action_FFNN3, RNN2_gen_action_FFNN3_tmp, RNN3_gen_action_FFNN3_tmp, RNN2_gen_action_linear, RNN3_gen_action_linear, RNN_w_action_FFNN3, RNN2_w_action_FFNN3

##################################################################################
#############################     above Library       ############################
#############################   below Miscellaneous   ############################
##################################################################################

# function to plot predicted speed from model
def plot_result(result_dict, orig_data):
    test_x, test_y = orig_data
    plot_x = np.arange(test_x.shape[0])

    if 'LeftWheel_seq' in result_dict:
        if result_dict['LeftWheel_seq'].shape[0] < plot_x.shape[0]:
            result_plot1 = result_dict['LeftWheel_seq'][0,:]
            result_plot2 = result_dict['RightWheel_seq'][0,:]
            result_plot3 = result_dict['LeftWheel_rec'][0,:]
            result_plot4 = result_dict['RightWheel_rec'][0,:]
        else:
            result_plot1 = result_dict['LeftWheel_seq']
            result_plot2 = result_dict['RightWheel_seq']
            result_plot3 = result_dict['LeftWheel_rec']
            result_plot4 = result_dict['RightWheel_rec']

        fig1 = plt.figure()
        plt.subplot(211)
        plt.plot(plot_x, result_plot1, 'r', label='Predicted Speed')
        plt.plot(plot_x, test_y[:,0], 'b', label='Encoded Speed')
        plt.title('Sequential Prediction(Left Wheel)')
        plt.legend()
        plt.subplot(212)
        plt.plot(plot_x, result_plot2, 'r', label='Predicted Speed')
        plt.plot(plot_x, test_y[:,1], 'b', label='Encoded Speed')
        plt.title('Sequential Prediction(Right Wheel)')
        plt.legend()

        fig2 = plt.figure()
        plt.subplot(211)
        plt.plot(plot_x, result_plot3, 'r', label='Predicted Speed')
        plt.plot(plot_x, test_y[:,0], 'b', label='Encoded Speed')
        plt.title('Recursive Prediction(Left Wheel)')
        plt.legend()
        plt.subplot(212)
        plt.plot(plot_x, result_plot4, 'r', label='Predicted Speed')
        plt.plot(plot_x, test_y[:,1], 'b', label='Encoded Speed')
        plt.title('Recursive Prediction(Right Wheel)')
        plt.legend()

        plt.show()

    elif 'LeftWheel' in result_dict:
        if result_dict['LeftWheel'].shape[0] < plot_x.shape[0]:
            result_plot1 = result_dict['LeftWheel'][0,:]
            result_plot2 = result_dict['RightWheel'][0,:]
        else:
            result_plot1 = result_dict['LeftWheel']
            result_plot2 = result_dict['RightWheel']

        fig1 = plt.figure()
        plt.subplot(211)
        plt.plot(plot_x, result_plot1, 'r', label='Predicted Speed')
        plt.plot(plot_x, test_y[:,0], 'b', label='Encoded Speed')
        plt.title('Prediction(Left Wheel)')
        plt.legend()
        plt.subplot(212)
        plt.plot(plot_x, result_plot2, 'r', label='Predicted Speed')
        plt.plot(plot_x, test_y[:,1], 'b', label='Encoded Speed')
        plt.title('Prediction(Right Wheel)')
        plt.legend()

        plt.show()

    else:
        print "Structure of Data Dictionary is changed!"
        print "Structure of Data Dictionary is changed!"
        print "Structure of Data Dictionary is changed!"


# function to print some of saved model parameters
def print_saved_model_parameter(parameter_list, list_index):
    para_print_op = theano.printing.Print('saved parameter')
    para_print = para_print_op(parameter_list[list_index])
    para_print_f = theano.function([], para_print)
    para_print_f()
    print parameter_list[list_index].shape.eval()


# function to print some of model parameters
def print_model_parameter(Model, list_index):
    para_print_op = theano.printing.Print('model parameter')
    para_print = para_print_op(Model.params[list_index])
    para_print_f = theano.function([], para_print)
    para_print_f()
    print Model.params[list_index].shape.eval()


# function to check similarity between kernel/parameters
def inner_product_btw_parameter(para_list, list_index):
    para_size = para_list[list_index].shape.eval()
    print "\t\tSize of kernel/parameter is", para_size

    result_mtx, result_list = np.zeros((para_size[0], para_size[0])), []
    if len(para_size) == 4:
        tmp = para_list[list_index].reshape((para_size[0], para_size[1]*para_size[2]*para_size[3])).eval()
        for cnt1 in range(para_size[0]):
            for cnt2 in range(cnt1+1, para_size[0]):
                vec1, vec2 = tmp[cnt1,:], tmp[cnt2,:]
                vec1_len = np.sqrt(np.inner(vec1, vec1))
                vec2_len = np.sqrt(np.inner(vec2, vec2))
                vec1_vec2_inner = abs(np.inner(vec1, vec2))
                result_mtx[cnt1,cnt2] = vec1_vec2_inner/(vec1_len * vec2_len)
                result_list.append(vec1_vec2_inner/(vec1_len * vec2_len))
    elif len(para_size) == 2:
        tmp = para_list[list_index].reshape((para_size[0], para_size[1])).eval()
        for cnt1 in range(para_size[0]):
            for cnt2 in range(cnt1+1, para_size[0]):
                vec1, vec2 = tmp[cnt1,:], tmp[cnt2,:]
                vec1_len = np.sqrt(np.inner(vec1, vec1))
                vec2_len = np.sqrt(np.inner(vec2, vec2))
                vec1_vec2_inner = abs(np.inner(vec1, vec2))
                result_mtx[cnt1,cnt2] = vec1_vec2_inner/(vec1_len * vec2_len)
                result_list.append(vec1_vec2_inner/(vec1_len * vec2_len))

    tmp2 = para_list[list_index+1].reshape((para_size[0],)).eval()
    max_tmp = np.argmax(result_mtx)
    max_ind = [max_tmp//para_size[0], max_tmp%para_size[0]]
    print "\t\t", np.max(result_mtx)
    print "\t\t", tmp[max_ind[0],:], tmp2[max_ind[0]]
    print "\t\t", tmp[max_ind[1],:], tmp2[max_ind[1]]
    result_list.sort(reverse=True)
    print result_list


# print all hyper-parameters of training procedure
def print_train_detail(robot_type, model_type, hist_window, pred_window, batch_size, lr, lr_decay, n_ep, use_RMSPROP, n_hidden, dim_act, data_file_name, result_folder_name, model_architecture, use_TF=None, use_pretrained=None, err_weight=None, n_state=None, n_kerns=None, kern_size=None, relu_alpha=None, leaky_int_coef=None, rnn_reg_weight=None):
    print "\n\nRobot/Model/Architecture : ", robot_type, model_type, model_architecture
    print "history/prediction window : ", hist_window, pred_window
    print "Batch Size : ", batch_size, ", Max # of Epochs : ", n_ep
    print "Learning rate : ", lr, ", Learning rate decay : ", lr_decay

    if use_RMSPROP:
        print "RMSPROP for SGD"
    else:
        print "Pure SGD"

    if not (err_weight is None):
        print "Non-trivial Weights to each time-steps for error"
    else:
        print "    trivial Weights for error"

    if model_type == 'RNN':
        if use_TF:
            print "Teacher Forcing for Training"
        else:
            print "Use Own Outputs for Training"
        print "Dimension of RNN states : ", n_state
    else:
        if use_pretrained:
            print "Initialize with Pre-trained Model"
        else:
            print "Initialize with Random Weights"
        print "# of CNN channels : ", n_kerns
        print "Dimension of CNN kernels : ", kern_size

    print "# of hidden units of FFNN : ", n_hidden
    print "Dimension of Generalized Action Transform : ", dim_act

    if model_type == 'RNN':
        print "Leaky-ReLu coefficient : ", relu_alpha
        print "Leaky Integration coefficient : ", leaky_int_coef
        print "Weight for Regularization : ", rnn_reg_weight

    print "Data File Name : ", data_file_name
    print "Result Folder Name : ", result_folder_name
    print "\n\n"


##################################################################################
#############################   above Miscellaneous   ############################
#############################     below Training      ############################
##################################################################################

### function to train the model
def train_rnn_deep_dynamics(n_out, n_state, n_hidden, hist_window, pred_window, robot_type, model_type, model_struct, result_folder, dataset, rng, useSmoothed=True, learning_rate=0.000001, learning_rate_decay=0.01, rnn_regularization=[10**-10, 10**-10], n_epochs=5000, n_save=20, batch_size=200, patience_list=[50000, 3], start_epoch=None, paraset=None, evalset=None, use_RMSPROP=False, use_TeacherForcing=False, decay_rate_RMSPROP=0.9, relu_alpha=0, error_weight=None, leaky_integration=0, dim_act=128):
    # print all hyper-parameters related to training procedure
    print_train_detail(robot_type=robot_type, model_type=model_type, hist_window=hist_window, pred_window=pred_window, batch_size=batch_size, lr=learning_rate, lr_decay=learning_rate_decay, n_ep=n_epochs, use_RMSPROP=use_RMSPROP, use_TF=use_TeacherForcing, err_weight=error_weight, n_state=n_state, n_hidden=n_hidden, dim_act=dim_act, relu_alpha=relu_alpha, leaky_int_coef=leaky_integration, rnn_reg_weight=rnn_regularization, data_file_name=dataset, result_folder_name=result_folder, model_architecture=model_struct)

    # load data
    #datasets = load_data_vulcan_rnn(dataset, hist_window, pred_window)
    datasets = load_data(dataset, hist_window, pred_window, robot_type=robot_type, model_type=model_type)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    final_test_set_x, final_test_set_y = datasets[3]

    # compute number of minibatches for training, validation and testing
    num_train = train_set_x.get_value(borrow=True).shape[0]
    num_valid = valid_set_x.get_value(borrow=True).shape[0]
    num_test = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches = num_train // batch_size
    n_valid_batches = num_valid // batch_size
    n_test_batches = num_test // batch_size

    valid_data_num = batch_size * n_valid_batches
    test_data_num = batch_size * n_test_batches
    x_dim, y_dim = train_set_x.get_value(borrow=True).shape[1], train_set_y.get_value(borrow=True).shape[1]

    if error_weight is None:
        error_weight = T.ones((batch_size, 2*pred_window), dtype=theano.config.floatX)


    # make folder to save resulted model parameter/evaluation files
    curr_path = os.getcwd()
    if result_folder in os.listdir(curr_path):
        print "subfolder exists"
    else:
        print "Not Exist, so make that"
        os.mkdir(result_folder)

    if not (paraset is None):
        paraname = './' + result_folder + '/' + paraset
        with open(paraname, 'rb') as f3:
            tmp_para_list = pickle.load(f3)
            if len(tmp_para_list) == 2:
                best_para_list, robot_para_list = tmp_para_list[0], tmp_para_list[1]
            elif len(tmp_para_list) == 4:
                best_para_list, best_para_grad_list, robot_para_list, robot_para_grad_list = tmp_para_list[0], tmp_para_list[1], tmp_para_list[2], tmp_para_list[3]
            else:
                robot_para_list = tmp_para_list[0:len(tmp_para_list)//2]
                robot_para_grad_list = tmp_para_list[len(tmp_para_list)//2:len(tmp_para_list)]
            print "Parameter File is loaded"
    else:
        robot_para_list = paraset
        robot_para_grad_list = paraset

    hist_valid_error, hist_test_error = [], []
    best_validation_loss, best_iter = np.inf, 0
    if not (evalset is None) and not (start_epoch is None):
        evalname = './' + result_folder + '/' + evalset
        result_tmp = loadmat(evalname)
        for cnt in range(start_epoch):
            hist_valid_error.append([result_tmp['history_validation_error'][cnt][0], result_tmp['history_validation_error'][cnt][1], result_tmp['history_validation_error'][cnt][2]])
            hist_test_error.append([result_tmp['history_test_error'][cnt][0], result_tmp['history_test_error'][cnt][1], result_tmp['history_test_error'][cnt][2]])
            if hist_valid_error[-1][1] < best_validation_loss:
                best_validation_loss = hist_valid_error[-1][1]
        #best_validation_loss = (best_validation_loss**2)*valid_data_num
        best_validation_loss = (best_validation_loss**2)*2*valid_data_num*pred_window
        print "Result File is loaded"
        print "\tLowest validation loss : ", np.sqrt(best_validation_loss/valid_data_num)
        print "\n"

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    t_epoch = T.fscalar()  # epoch to change learning rate
    x = T.matrix('x')   # input data in matrix form(num_data-by-num_features)
    y = T.matrix('y')  # output data in matrix form(num_data-by-num_outputs)

    if model_struct == 'rnn_action_ffnn3':
        # construct the RNN, act_trans, FFNN class
        print "\tRNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    elif model_struct == 'rnn2_action_ffnn3':
        # construct the 2 layer RNN, act_trans, FFNN class
        print "\t2 layer RNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN2_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    elif model_struct == 'rnn3_action_ffnn3':
        # construct the 3 layer RNN, act_trans, FFNN class
        print "\t3 layer RNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN3_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    elif model_struct == 'rnn_gen_action_ffnn3':
        # construct the RNN, act_trans, FFNN class
        print "\tRNN -> generalized action transformation -> 3 layer FFNN"
        robot_model = RNN_gen_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn2_gen_action_ffnn3':
        # construct the 2 layer RNN, act_trans, FFNN class
        print "\t2 layer RNN -> generalized action transformation -> 3 layer FFNN"
        robot_model = RNN2_gen_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn3_gen_action_ffnn3':
        # construct the 3 layer RNN, act_trans, FFNN class
        print "\t3 layer RNN -> generalized action transformation -> 3 layer FFNN"
        robot_model = RNN3_gen_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn2_gen_action_ffnn3_tmp':
        # construct the 2 layer RNN w/ gen_act_trans -> FFNN class
        print "\t2 layer RNN w/ gen. action transform. -> 3 layer FFNN"
        robot_model = RNN2_gen_action_FFNN3_tmp(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn3_gen_action_ffnn3_tmp':
        # construct the 3 layer RNN w/ gen_act_trans, FFNN class
        print "\t3 layer RNN w/ gen. action transform.  -> 3 layer FFNN"
        robot_model = RNN3_gen_action_FFNN3_tmp(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn2_gen_action_linear':
        # construct the 2 layer RNN, act_trans, FFNN class
        print "\t2 layer RNN w/ gen. action transform. -> linear transform"
        robot_model = RNN2_gen_action_linear(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn3_gen_action_linear':
        # construct the 3 layer RNN, act_trans, linear class
        print "\t3 layer RNN w/ gen. action transform. -> linear transform"
        robot_model = RNN3_gen_action_linear(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration, n_act_feature=dim_act)
    elif model_struct == 'rnn_w_act_ffnn3':
        print "\tRNN with action transform -> 3 layers of FFNN"
        robot_model = RNN_w_action_FFNN3(rng=rng, n_state=n_state, n_hidden=n_hidden, n_in=2, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    elif model_struct == 'rnn2_w_act_ffnn3':
        print "\t2 layer RNN with action transform -> 3 layer FFNN"
        robot_model = RNN2_w_action_FFNN3(rng=rng, n_state=n_state, n_hidden=n_hidden, n_in=2, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    else:
        print "Structure of Model is Wrong!!"
        print "Structure of Model is Wrong!!"
        print "Structure of Model is Wrong!!"
        return -1


    # theano function to compute error of model on minibatch test data
    test_model_error = theano.function(
        inputs=[index],
        outputs=robot_model(test_set_x[index * batch_size:(index + 1) * batch_size], test_set_y[index * batch_size:(index + 1) * batch_size])
    )

    # theano function to compute error of model on minibatch validation data
    validate_model_error = theano.function(
        inputs=[index],
        outputs=robot_model(valid_set_x[index * batch_size:(index + 1) * batch_size], valid_set_y[index * batch_size:(index + 1) * batch_size])
    )

    if not use_TeacherForcing:
        print "\tUse Own Outputs to compute error"
    else:
        print "\tUse Teacher-Forcing to compute error"

    # cost for training procedure
    cost = robot_model(x, y, teach_force=use_TeacherForcing, error_weight=error_weight)[1]
    # cost = robot_model(x, y, teach_force=use_TeacherForcing, error_weight=error_weight)[1] + 5.0 * robot_model(x, y, teach_force=use_TeacherForcing, error_weight=error_weight)[2]
    # cost = T.log(np.float32(4.0) + robot_model(x, y, teach_force=use_TeacherForcing, error_weight=error_weight)[2]) * robot_model(x, y, teach_force=use_TeacherForcing, error_weight=error_weight)[1]
    # cost = robot_model(x, y)[1] + 5.0 * robot_model(x, y)[2] + rnn_regularization[0] * robot_model(x, y, True)[3] + rnn_regularization[1] * robot_model(x, y, True)[4]

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in robot_model.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    if not use_RMSPROP:
        print "\tUse Stochastic Gradient Descent"
        #updates = [
        #    (param, param - learning_rate * gparam)
        #    for param, gparam in zip(robot_model.params, gparams)
        #]
        updates = [
            (param, param - learning_rate * (1/(1+learning_rate_decay*t_epoch)) * gparam)
            for param, gparam in zip(robot_model.params, gparams)
        ]
    else:
        print "\tUse RMSPROP for SGD"
        accs, acc_news, grad_scales = [], [], []
        if robot_para_grad_list is None:
            for param, gparam in zip(robot_model.params, gparams):
                acc = theano.shared(param.get_value(borrow=True)*0.0, borrow=True)
                acc_new = decay_rate_RMSPROP * acc + (1-decay_rate_RMSPROP)*(gparam**2)
                grad_scale = T.sqrt(acc_new+10**-8)

                accs.append(acc)
                acc_news.append(acc_new)
                grad_scales.append(grad_scale)
        else:
            for gparam, init_acc in zip(gparams, robot_para_grad_list):
                acc = theano.shared(init_acc.get_value(borrow=True), borrow=True)
                acc_new = decay_rate_RMSPROP * acc + (1-decay_rate_RMSPROP)*(gparam**2)
                grad_scale = T.sqrt(acc_new+10**-8)

                accs.append(acc)
                acc_news.append(acc_new)
                grad_scales.append(grad_scale)

        #updates = [(param, param - learning_rate * gparam/grad_sc) for param, gparam, grad_sc in zip(robot_model.params, gparams, grad_scales)] + [(acc, acc_new) for acc, acc_new in zip(accs, acc_news)]
        updates = [(param, param - learning_rate * (1/(1+learning_rate_decay*t_epoch)) * gparam/grad_sc) for param, gparam, grad_sc in zip(robot_model.params, gparams, grad_scales)] + [(acc, acc_new) for acc, acc_new in zip(accs, acc_news)]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, t_epoch],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            train_model.maker.fgraph.toposort()]):
        print('Used the cpu')
    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              train_model.maker.fgraph.toposort()]):
        print('Used the gpu')
    else:
        print('ERROR, not able to tell if theano used the cpu or the gpu')
        print(train_model.maker.fgraph.toposort())


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = patience_list[0]  # look as this many examples regardless
    patience_increase = patience_list[1]  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                   # considered significant

    validation_frequency = n_train_batches
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    start_time = timeit.default_timer()
    start_time_part = timeit.default_timer()

    if start_epoch is None:
        epoch = 0
    else:
        epoch = start_epoch
    recent_update = epoch

    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            #minibatch_avg_cost = train_model(minibatch_index)
            minibatch_avg_cost = train_model(minibatch_index, epoch)

            # iteration number
            num_iter = (epoch - 1) * n_train_batches + minibatch_index

            if (num_iter + 1) % validation_frequency == 0:
                # compute error loss on validation set
                validation_losses_tmp = [validate_model_error(i) for i in range(n_valid_batches)]
                validation_losses = np.zeros(3)

                validation_losses[0:2] = np.sum(validation_losses_tmp, axis=0)[0:2]
                validation_losses[2] = np.amax(validation_losses_tmp, axis=0)[2]
                this_validation_loss = validation_losses[1]

                if np.any(np.isnan(validation_losses)):
                    print '\tIs NaN in validation_losses_tmp\t'
                    exit()

                #hist_valid_error.append([validation_losses[0]/valid_data_num, np.sqrt(validation_losses[1]/valid_data_num), validation_losses[2]])
                hist_valid_error.append([validation_losses[0]/(2*valid_data_num*pred_window), np.sqrt(validation_losses[1]/(2*valid_data_num*pred_window)), validation_losses[2]])
                print 'epoch %i, validation error %f/%f/%f' % (epoch, validation_losses[0]/valid_data_num, np.sqrt(validation_losses[1]/valid_data_num), validation_losses[2])

                del validation_losses_tmp

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, num_iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = num_iter

                    # test it on the test set
                    test_losses_tmp = [test_model_error(i) for i in range(n_test_batches)]
                    test_losses = np.zeros(3)

                    test_losses[0:2] = np.sum(test_losses_tmp, axis=0)[0:2]
                    test_losses[2] = np.amax(test_losses_tmp, axis=0)[2]

                    #hist_test_error.append([test_losses[0]/test_data_num, np.sqrt(test_losses[1]/test_data_num), test_losses[2]])
                    hist_test_error.append([test_losses[0]/(2*test_data_num*pred_window), np.sqrt(test_losses[1]/(2*test_data_num*pred_window)), test_losses[2]])
                    print '\tepoch %i, test error of best model %f/%f/%f' % (epoch, test_losses[0]/test_data_num, np.sqrt(test_losses[1]/test_data_num), test_losses[2])

                    del test_losses_tmp

                    # save the best model
                    savepicklename = './' + result_folder + '/best_rnn_model(hw'+ str(hist_window) +'_pw' +str(pred_window) +'_b' + str(batch_size)+').pkl'

                    best_para_list = list(robot_model.params)

                    if use_RMSPROP:
                        best_para_grad_list = list(accs)
                        with open(savepicklename, 'wb') as f2:
                            pickle.dump([best_para_list, best_para_grad_list, robot_model.params, list(accs)], f2)
                    else:
                        with open(savepicklename, 'wb') as f2:
                            pickle.dump([best_para_list, robot_model.params], f2)
                else:
                    hist_test_error.append(hist_test_error[-1])

                # save intermediate history of validation/test error
                result2={}
                result2['history_validation_error'] = hist_valid_error
                result2['history_test_error'] = hist_test_error
                result2['num_observation_step_in_input'] = hist_window
                result2['num_prediction_step_in_input'] = pred_window
                result2['learning_rate'] = [learning_rate, learning_rate_decay]
                result2['batch_size'] = batch_size
                result2['epoch'] = epoch
                result2['update_threshold'] = improvement_threshold
                result2['state_dimension'] = n_state
                result2['num_hidden_decoder'] = n_hidden
                #result2['regularization_weight'] = rnn_regularization

                savefilename2 = './' + result_folder + '/Result_of_rnn_Model(hw'+ str(hist_window) +'_pw' + str(pred_window) +'_b' + str(batch_size) + ').mat'
                savemat(savefilename2, result2)

            if patience <= num_iter:
                done_looping = True
                break

        # save intermediate model/prediction result
        if (epoch > 0) and (epoch%n_save == 0):
            savepicklename2 = './' + result_folder + '/best_rnn_model(hw'+ str(hist_window) +'_pw' +str(pred_window) +'_b'+ str(batch_size)+ '_epoch' + str(epoch) +').pkl'

            if use_RMSPROP:
                with open(savepicklename2, 'wb') as f3:
                    pickle.dump([best_para_list, best_para_grad_list, robot_model.params, list(accs)], f3)
            else:
                with open(savepicklename2, 'wb') as f3:
                    pickle.dump([best_para_list, robot_model.params], f3)

            end_time_part = timeit.default_timer()
            print '\t\tTime for sub-procedure', end_time_part - start_time_part
            start_time_part = timeit.default_timer()
            recent_update = epoch

    end_time = timeit.default_timer()
    try:
        print 'Optimization complete. Best validation score of %f at iteration %i, with test performance %f/%f/%f' % (best_validation_loss, best_iter + 1, test_losses[0], test_losses[1], test_losses[2])
        print 'Time for training', end_time - start_time
    finally:
        return recent_update


##################################################################################
################################# above Training #################################
#################################   below Test   #################################
##################################################################################

# function to predict output of best model
def pred_rnn_deep_dynamics(dataset, paraset, n_state, n_hidden, robot_type, model_type, model_struct, rng, n_out=2, hist_window=8, pred_window=25, batch_size=200, pred_index=[-1, -1, -1], useSmoothed=True, predMode='recursive', relu_alpha=0):
    #datasets = load_data_vulcan_rnn(dataset, hist_window, pred_window)
    datasets = load_data(dataset, hist_window, pred_window, robot_type=robot_type, model_type=model_type)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    final_test_set_x, final_test_set_y = datasets[3]

    # range of prediction(start_index:end_index-1 or start_index:start_index+interval-1
    #                     which means it does not include last index)
    if pred_index[0] < 0:
        start_index = 0
    else:
        start_index = pred_index[0]
    if (pred_index[1] < 0 and pred_index[2] < 0) or (pred_index[1] > test_set_x.get_value(borrow=True).shape[0]):
        end_index = test_set_x.get_value(borrow=True).shape[0]
    elif pred_index[1] < 0 and pred_index[2] > 0:
        end_index = start_index + pred_index[2]
    else:
        end_index = pred_index[1] + 1


    with open(paraset, 'rb') as f2:
        tmp_para_list = pickle.load(f2)
        if len(tmp_para_list) == 2:
            robot_para_list, last_robot_para_list = tmp_para_list[0], tmp_para_list[1]
        else:
            robot_para_list = tmp_para_list
        print "Parameter File is loaded"

    index = T.lscalar()

    if model_struct == 'rnn_action_ffnn3':
        # construct the RNN, act_trans, FFNN class
        print "\tRNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha)
    elif model_struct == 'rnn2_action_ffnn3':
        # construct the 2 layer RNN, act_trans, FFNN class
        print "\t2 layer RNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN2_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha)
    elif model_struct == 'rnn3_action_ffnn3':
        # construct the 3 layer RNN, act_trans, FFNN class
        print "\t3 layer RNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN3_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=robot_para_list, leaky_relu_alpha=relu_alpha)
    else:
        print "Structure of Model is Wrong!!"
        print "Structure of Model is Wrong!!"
        print "Structure of Model is Wrong!!"
        return -1


    prediction_on_data = np.zeros((end_index-start_index,2))
    test_output_rec_shared = theano.shared(np.asarray(np.zeros(2*hist_window+2*pred_window),
dtype=theano.config.floatX), borrow=True)
    test_y = np.zeros((end_index-start_index,2))

    # theano function for prediction of model on minibatch test data
    test_model_output = theano.function(
        inputs=[index],
        outputs=robot_model(test_set_x[index * batch_size:(index + 1) * batch_size])[:,0:2]
    )

    # theano function for prediction of model on test data sequentially
    test_model_output_seq = theano.function(
        inputs=[index],
        outputs=robot_model(T.tile(test_set_x[index:(index + 1)], (batch_size, 1)))[0, 0:2]
    )

    # theano function for prediction of model on test data recursively
    test_model_output_rec = theano.function(
        inputs=[],
        outputs=robot_model(T.tile(test_output_rec_shared, (batch_size, 1)))[0, :].reshape((pred_window, 2))
    )

    if predMode == 'sequential':
        print "\tPredict wheels' speed Not recursively"
        savefilename = 'Result_of_conv_Model(hw'+ str(hist_window) +'_pw' + str(pred_window) +'_b' + str(batch_size) + 'seq).mat'

        for cnt in range(end_index-start_index):
            prediction_on_data[cnt,:] = test_model_output_seq(cnt)
        test_y = test_set_y.get_value(borrow=True)[start_index:end_index,0:2]

    elif predMode == 'recursive':
        print "\tPredict wheels' speed recursively"
        savefilename = 'Result_of_conv_Model(hw'+ str(hist_window) +'_pw' + str(pred_window) +'_b' + str(batch_size) + 'rec).mat'


        n_test_rec = (end_index - start_index) // pred_window
        for cnt in range(n_test_rec):
            test_x_tmp = test_set_x.get_value()[start_index+cnt*pred_window,:]
            if cnt > 0:
                test_x_tmp[0:2*hist_window] = prediction_on_data[cnt*pred_window-hist_window:cnt*pred_window,:].flatten()

            test_output_rec_shared.set_value(test_x_tmp)
            prediction_on_data[cnt*pred_window:(cnt+1)*pred_window,:] = test_model_output_rec()
        test_y[0:n_test_rec*pred_window, :] = test_set_y.get_value(borrow=True)[start_index:start_index+n_test_rec*pred_window,0:2]

        if end_index > start_index + n_test_rec*pred_window:
            cnt = end_index - start_index - pred_window
            test_x_tmp = test_set_x.get_value()[cnt+start_index,:]
            test_x_tmp[0:2*hist_window] = prediction_on_data[cnt+start_index-hist_window:cnt+start_index,:].flatten()

            test_output_rec_shared.set_value(test_x_tmp)
            prediction_on_data[cnt:cnt+pred_window,:] = test_model_output_rec()
            test_y[cnt:cnt+pred_window, :] = test_set_y.get_value(borrow=True)[start_index+cnt,:].reshape((pred_window,2))

    #diff = abs(prediction_on_data - test_set_y.get_value()[start_index:end_index,0:2])
    diff = abs(prediction_on_data - test_y)
    prediction_error = [diff.sum(), (diff**2).sum(), diff.max()]

    result={}
    result['LeftWheel'] = prediction_on_data[:,0]
    result['RightWheel'] = prediction_on_data[:,1]
    result['L1Error'] = prediction_error[0]
    result['L2Error'] = prediction_error[1]
    result['LinfError'] = prediction_error[2]
    savemat(savefilename, result)
    print 'Result File is successfully saved! Errors are %f, %f, %f' % (prediction_error[0], prediction_error[1], prediction_error[2])

    plot_result(result, (test_y, test_y))


############################
### function to predict output/error of best model for 5 seconds
############################
def test_rnn_deep_dynamics_5s(dataset, paraset, result_folder, n_state, n_hidden, robot_type, model_type, model_struct, rng, n_out=2, hist_window=8, pred_window=25, batch_size=200, useSmoothed=True, saveFigure=False, relu_alpha=0, leaky_integration=0.0, dim_act=0):

    batch_size = 1

    if (robot_type == 'Vulcan'):
        plot_cmd_divider = 100
    elif (robot_type == 'Fetch') or (robot_type == 'Fetch2') or (robot_type == 'MagicBot'):
        plot_cmd_divider = 1

    datasets = load_data(dataset, hist_window, pred_window, robot_type=robot_type, model_type=model_type)

    pred_window = 125

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    final_test_set_x, final_test_set_y = datasets[3]

    if not (paraset is None):
        paraname = './' + result_folder + '/' + paraset
        with open(paraname, 'rb') as f3:
            tmp_para_list = pickle.load(f3)
            if len(tmp_para_list) == 2:
                best_para_list, robot_para_list = tmp_para_list[0], tmp_para_list[1]
            elif len(tmp_para_list) == 4:
                best_para_list, best_para_grad_list, robot_para_list, robot_para_grad_list = tmp_para_list[0], tmp_para_list[1], tmp_para_list[2], tmp_para_list[3]
            else:
                robot_para_list = tmp_para_list[0:len(tmp_para_list)//2]
                robot_para_grad_list = tmp_para_list[len(tmp_para_list)//2:len(tmp_para_list)]
            print "Parameter File is loaded"
    else:
        robot_para_list = paraset
        robot_para_grad_list = paraset

    index = T.lscalar()

    if model_struct == 'rnn_action_ffnn3':
        # construct the RNN, act_trans, FFNN class
        print "\tRNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=best_para_list, leaky_relu_alpha=relu_alpha)
    elif model_struct == 'rnn2_action_ffnn3':
        # construct the 2 layer RNN, act_trans, FFNN class
        print "\t2 layer RNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN2_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=best_para_list, leaky_relu_alpha=relu_alpha)
    elif model_struct == 'rnn3_action_ffnn3':
        # construct the 3 layer RNN, act_trans, FFNN class
        print "\t3 layer RNN -> action transformation -> 3 layer FFNN"
        robot_model = RNN3_action_FFNN3(rng=rng, n_state=n_state, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, n_hidden=n_hidden, para_list=best_para_list, leaky_relu_alpha=relu_alpha)
    elif model_struct == 'rnn_w_act_ffnn3':
        print "\tRNN with action transform -> 3 layers of FFNN"
        robot_model = RNN_w_action_FFNN3(rng=rng, n_state=n_state, n_hidden=n_hidden, n_in=2, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, para_list=best_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    elif model_struct == 'rnn2_w_act_ffnn3':
        print "\t2 layer RNN with action transform -> 3 layer FFNN"
        robot_model = RNN2_w_action_FFNN3(rng=rng, n_state=n_state, n_hidden=n_hidden, n_in=2, n_out=n_out, batch_size = batch_size, hist_window=hist_window, pred_window=pred_window, para_list=best_para_list, leaky_relu_alpha=relu_alpha, leaky_integration=leaky_integration)
    else:
        print "Structure of Model is Wrong!!"
        print "Structure of Model is Wrong!!"
        print "Structure of Model is Wrong!!"
        return -1


    prediction_on_data = np.zeros(final_test_set_y.get_value(borrow=True).shape)
    #test_data_in_numpy = np.zeros((batch_size,2*hist_window+50))
    test_data_in_numpy = np.zeros((batch_size, 2*hist_window+250),
dtype=theano.config.floatX)
    test_data_in_shared = theano.shared(np.asarray(np.zeros(test_data_in_numpy.shape),
dtype=theano.config.floatX), borrow=True)

    pred_left_speed = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125))
    pred_right_speed = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125))
    forward_cmd = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125))
    left_cmd = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125))

    # theano function for prediction of model on minibatch test data
    test_model_output_for_remains = theano.function(
        inputs=[],
        outputs=robot_model(test_data_in_shared)
    )

    print "\tPredict wheels' speed for 5 seconds"

    n_test = final_test_set_x.get_value(borrow=True).shape[0]
    n_iter = n_test // batch_size
    n_remain = n_test - n_iter * batch_size

    start_time_rec_pred = timeit.default_timer()

    for cnt in range(n_iter):
        test_data_in_numpy[:,:] = final_test_set_x.get_value(borrow=True)[cnt*batch_size:(cnt+1)*batch_size,:]
        test_data_in_shared.set_value(test_data_in_numpy)
        prediction_on_data[cnt*batch_size:(cnt+1)*batch_size,:] = test_model_output_for_remains()

        tmp_cmd = final_test_set_x.get_value(borrow=True)[cnt*batch_size:(cnt+1)*batch_size, 2*hist_window:(2*hist_window+250)].reshape((batch_size, 125, 2))
        tmp_pred = prediction_on_data[cnt*batch_size:(cnt+1)*batch_size, :].reshape((batch_size, 125, 2))

        for in_cnt in range(batch_size):
            pred_left_speed[cnt*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,0]
            pred_right_speed[cnt*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,1]
            forward_cmd[cnt*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,0]
            left_cmd[cnt*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,1]
        if cnt in [n_iter//5, n_iter*2//5, n_iter*3//5, n_iter*4//5]:
            print "\t\t%.2f %% complete!" % (cnt*100.0/n_iter)

    if n_remain > 0:
        test_data_in_numpy[0:n_remain,:] = final_test_set_x.get_value(borrow=True)[n_iter*batch_size:n_test,:]
        test_data_in_shared.set_value(test_data_in_numpy)
        prediction_on_data[n_iter*batch_size:n_test,:] = test_model_output_for_remains()[0:n_remain,:]

        tmp_cmd = final_test_set_x.get_value(borrow=True)[n_iter*batch_size:n_test, 2*hist_window:(2*hist_window+250)].reshape((n_remain, 125, 2))
        tmp_pred = prediction_on_data[n_iter*batch_size:n_test, :].reshape((n_remain, 125, 2))

        for in_cnt in range(n_remain):
            pred_left_speed[n_iter*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,0]
            pred_right_speed[n_iter*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,1]
            forward_cmd[n_iter*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,0]
            left_cmd[n_iter*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,1]

    end_time_rec_pred = timeit.default_timer()
    rec_pred_time = (end_time_rec_pred - start_time_rec_pred)/n_test
    print '\t\t5 second(125 data points) LTS of model(%f)' % rec_pred_time

    diff = abs(prediction_on_data - final_test_set_y.get_value(borrow=True))
    max_index = diff.argmax() // final_test_set_y.get_value(borrow=True).shape[1]
    prediction_error = [diff.sum()/n_test, np.sqrt((diff**2).sum()/n_test), diff.max()]

    print 'Errors for 5 seconds simulation are %f, %f, %f' % (prediction_error[0], prediction_error[1], prediction_error[2])

    save_result = {}
    save_result['model_prediction'] = prediction_on_data
    save_result['joystick_command'] = final_test_set_x.get_value(borrow=True)[:, (2*hist_window):(2*hist_window+250)]
    save_result['observed_speed'] = final_test_set_y.get_value(borrow=True)
    save_result['time_for_simulation'] = end_time_rec_pred - start_time_rec_pred
    save_result['difference_btw_prediction_data'] = diff
    save_result['error'] = prediction_error

    if len(result_folder) > 0:
        save_file_name = './' + result_folder + '/5sec_LTS_result' + paraset[15:-4] + '.mat'
    else:
        save_file_name = '5sec_LTS_result' + paraset[15:-4] + '.mat'
    savemat(save_file_name, save_result)

    # plot (lots of) graphs
    if saveFigure:
        real_y_tmp = final_test_set_y.get_value(borrow=True).reshape((n_test,125,2))
        real_y_left, real_y_right = np.zeros((n_test,125)), np.zeros((n_test,125))
        for cnt in range(n_test):
            real_y_left[cnt,:] = real_y_tmp[cnt,:,0]
            real_y_right[cnt,:] = real_y_tmp[cnt,:,1]

        if len(result_folder) > 0:
            if not('ResultPlot' in os.listdir('./'+result_folder)):
                os.mkdir('./'+result_folder+'/ResultPlot')
            result_folder = './' + result_folder+'/ResultPlot'
        else:
            if not('ResultPlot' in os.listdir(os.getcwd())):
                os.mkdir('ResultPlot')
            result_folder = './ResultPlot'
        pdf_name = result_folder + '/resultplots' + paraset[15:-4] + '.pdf'
        pp = PdfPages(pdf_name)

        for cnt in range(n_test // 25 + 1):
            if cnt < n_test//25:
                plot_cnt = cnt
            else:
                plot_cnt= max_index / 25.0

            if cnt in [n_test//125, n_test*2//125, n_test*3//125, n_test*4//125]:
                print "\t\t%.2f %% complete!" % (cnt*100.0/(n_test//25))
            #plot_x = np.linspace(plot_cnt, plot_cnt+5, num=125, endpoint=True)
            plot_x = np.linspace(0, 5, num=125, endpoint=True)
            plot_y1 = pred_left_speed[int(plot_cnt*25),:]
            plot_y2 = pred_right_speed[int(plot_cnt*25),:]
            plot_real_y1 = real_y_left[int(plot_cnt*25),:]
            plot_real_y2 = real_y_right[int(plot_cnt*25),:]
            plot_for_cmd = forward_cmd[int(plot_cnt*25),:]/plot_cmd_divider
            plot_left_cmd = left_cmd[int(plot_cnt*25),:]/plot_cmd_divider
            title1 = 'Left Wheel 5sec Prediction from ' + str(plot_cnt)
            title2 = 'Right Wheel 5sec Prediction from ' + str(plot_cnt)
            filename = result_folder + '/5secLTS_' + str(plot_cnt) + '.png'

            fig1 = plt.figure()
            plt.subplot(211)
            plt.plot(plot_x, plot_y1, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_real_y1, 'b-.', label='Encoded Speed')
            plt.plot(plot_x, plot_for_cmd, 'k--', label='Forward Command')
            plt.plot(plot_x, plot_left_cmd, 'k-.', label='Left Command')
            plt.title(title1)
            #plt.legend(loc=2)
            plt.subplot(212)
            plt.plot(plot_x, plot_y2, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_real_y2, 'b-.', label='Encoded Speed')
            plt.plot(plot_x, plot_for_cmd, 'k--', label='Forward Command')
            plt.plot(plot_x, plot_left_cmd, 'k-.', label='Left Command')
            plt.title(title2)
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 1.0))

            fig1.savefig(filename, bbox_inches='tight', pad_inches=0)
            pp.savefig()
            plt.close()

        pp.close()
        #plt.show()


##################################################################################
#################################   above Test   #################################
#################################   below Main   #################################
##################################################################################

# export PATH=/usr/local/cuda-8.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
# THEANO_FLAGS='cuda.root=/usr/local/cuda/',device=gpu0,floatX=float32 python robot_main_rnn.py
# THEANO_FLAGS='cuda.root=/usr/local/cuda/',device=gpu1,floatX=float32 python robot_main_rnn2.py
# THEANO_FLAGS='cuda.root=/usr/local/cuda/',device=gpu2,floatX=float32 python robot_main_rnn3.py
# THEANO_FLAGS='cuda.root=/usr/local/cuda/',device=gpu0,floatX=float32,lib.cnmem=0.85 python robot_main_rnn.py

if __name__ == '__main__':
    theano.config.floatX = 'float32'
    theano.config.warn_float64 = 'raise'
    sys.setrecursionlimit(2000)

    batch_size_in = 32
    #lr, n_ep = 0.0000002, 3000
    lr, lr_decay, n_ep, n_save = 0.001, 0.05, 500, 5
    #lr, lr_decay, n_ep, n_save = 0.0002, 0.08, 500, 5

    patience_list_in = [50000000, 3]
    rng = np.random.RandomState(1234)
    #rng = np.random.RandomState(2000)

    str_mode = raw_input('Type mode of this program\t')
    hist_window = input('\t# of history of observation\t')
    pred_window = input('\t# of recursive prediction\t')


    error_weight_tmp = np.ones((batch_size_in, 2*pred_window), dtype=theano.config.floatX)
    error_reduction = 0.99
    for cnt in range(1, pred_window):
        error_weight_tmp[:, 2*cnt:2*(cnt+1)] = error_weight_tmp[:, 2*(cnt-1):2*cnt]*error_reduction

    # update_mode T=RMSPROP, F=SGD // train_mode T=TeacherForcing, F=General Method
    update_mode, train_mode, error_weight_in = True, False, None
    #update_mode, train_mode, error_weight_in = True, False, error_weight_tmp


    ### Vulcan robot/ RNN
    #robot_type, model_type = 'Vulcan', 'RNN'
    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = 256, 128, [0, 0], 0.001, 0.1, 0
    #model_class_type = 'rnn_action_ffnn3'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 32], 128, [0, 0], 0.001, 0.1, 0
    #model_class_type = 'rnn2_action_ffnn3'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 32, 64], 128, [0, 0], 0.01, 0.1, 0
    #model_class_type = 'rnn3_action_ffnn3'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = 256, 128, [0, 0], 0.001, 0.15, 32
    #model_class_type = 'rnn_gen_action_ffnn3'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 32], 128, [0, 0], 0.001, 0.15, 64
    #model_class_type = 'rnn2_gen_action_ffnn3'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 32, 64], 128, [0, 0], 0.01, 0.15, 92
    #model_class_type = 'rnn3_gen_action_ffnn3'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 32], 128, [0, 0], 0.001, 0.15, [32, 64]
    #model_class_type = 'rnn2_gen_action_ffnn3_tmp'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 64, 196], 96, [0, 0], 0.01, 0.1, [48, 64, 128]
    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [32, 64, 196], 128, [0, 0], 0.002, 0.5, [48, 64, 128]
    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [16, 32, 64], 96, [0, 0], 0.01, 0.1, [32, 64, 128]
    #model_class_type = 'rnn3_gen_action_ffnn3_tmp'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [32, 128], 0, [0, 0], 0.002, 0.5, [48, 128]
    #model_class_type = 'rnn3_gen_action_linear'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [32, 64, 128], 0, [0, 0], 0.002, 0.5, [48, 64, 96]
    #model_class_type = 'rnn3_gen_action_linear'

    #n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [64, 96], 64, [0, 0], 0.001, 0.15, 0
    #model_class_type = 'rnn2_w_act_ffnn3'


    ### MagicBot robot/ RNN
    robot_type, model_type = 'MagicBot', 'RNN'
    n_state, n_hidden, rnn_reg_weight, relu_alpha, leaky_int_coef, dim_act = [64, 96], 64, [0, 0], 0.001, 0.15, 0
    model_class_type = 'rnn2_w_act_ffnn3'


    if str_mode == "train":
        print "Train new model"
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        result_folder_name = raw_input('\tname of subfolder for result files\t')

        train_rnn_deep_dynamics(n_out=2, n_state=n_state, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, model_struct=model_class_type, result_folder=result_folder_name, rng=rng, learning_rate=lr, learning_rate_decay=lr_decay, n_epochs=n_ep, n_save=n_save, dataset=data_file_name, batch_size=batch_size_in, patience_list=patience_list_in, rnn_regularization=rnn_reg_weight, use_RMSPROP=update_mode, use_TeacherForcing=train_mode, relu_alpha=relu_alpha, error_weight=error_weight_in, leaky_integration=leaky_int_coef, dim_act=dim_act, robot_type=robot_type, model_type=model_type)

    elif str_mode == "train_conti":
        print "Train new model"
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        result_folder_name = raw_input('\tname of subfolder for result files\t')
        start_epoch = input('\t# of epoch to start\t')

        curr_path = os.getcwd()
        result_path = curr_path + '/' + result_folder_name
        para_file_name = 'best_rnn_model(hw' + str(hist_window) + '_pw' + str(pred_window) + '_b' + str(batch_size_in) + '_epoch' + str(start_epoch) +').pkl'
        eval_file_name = 'Result_of_rnn_Model(hw' + str(hist_window) + '_pw' + str(pred_window) + '_b' + str(batch_size_in) + '_epoch' + str(start_epoch) +').mat'
        if (result_folder_name in os.listdir(curr_path)) and (para_file_name in os.listdir(result_path)) and (eval_file_name in os.listdir(result_path)):
            train_rnn_deep_dynamics(n_out=2, n_state=n_state, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, model_struct=model_class_type, result_folder=result_folder_name, rng=rng, learning_rate=lr, learning_rate_decay=lr_decay, n_epochs=n_ep, n_save=n_save, dataset=data_file_name, batch_size=batch_size_in, patience_list=patience_list_in, start_epoch=start_epoch, paraset=para_file_name, evalset=eval_file_name, rnn_regularization=rnn_reg_weight, use_RMSPROP=update_mode, use_TeacherForcing=train_mode, relu_alpha=relu_alpha, error_weight=error_weight_in, leaky_integration=leaky_int_coef, dim_act=dim_act, robot_type=robot_type, model_type=model_type)
        else:
            print "Model to continue training does NOT exist!"
            print "Model to continue training does NOT exist!"

    elif str_mode == "test":
        print "Test the model on existing data"
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'

        pred_mode = raw_input('\tMode of Prediction(sequential/recursive)\t')
        print '\tstart, end, length of prediction interval(or type -1)'
        a = input('\t\t\t')
        b = input('\t\t\t')
        c = input('\t\t\t')

        pred_rnn_deep_dynamics(n_out=2, pred_index=[a, b, c], n_state=n_state, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, model_struct=model_class_type, rng=rng, batch_size=batch_size_in, dataset=data_file_name, paraset=para_file_name, predMode=pred_mode, robot_type=robot_type, model_type=model_type)

    elif str_mode == "test5sec":
        print "Test the model for 5 seconds simulation"
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        result_folder_name = raw_input('\tname of subfolder for result files\t')
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'

        save_figure_str = raw_input('\tSave Figure?(T/F)\t')
        if save_figure_str == 'T':
            save_figure = True
        elif save_figure_str == 'F':
            save_figure = False

        curr_path = os.getcwd()
        result_path = curr_path + '/' + result_folder_name
        if (result_folder_name in os.listdir(curr_path)) and (para_file_name in os.listdir(result_path)):
            test_rnn_deep_dynamics_5s(n_out=2, n_state=n_state, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, model_struct=model_class_type, rng=rng, batch_size=batch_size_in, dataset=data_file_name, result_folder=result_folder_name, paraset=para_file_name, saveFigure=save_figure, robot_type=robot_type, model_type=model_type, relu_alpha=relu_alpha, leaky_integration=leaky_int_coef, dim_act=dim_act)

    elif str_mode == "plot":
        print "plot the speed of wheels predicted from the model"
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '.pkl'
        result_file_name = raw_input('\tName of result file\t')
        result_file_name = result_file_name + '.mat'

        datasets = load_data(data_file_name, hist_window, pred_window, robot_type=robot_type, model_type=model_type)

        test_set_x, test_set_y = datasets[2]
        result_of_model = loadmat(result_file_name)
        plot_result(result_of_model, (test_set_x.get_value(), test_set_y.get_value()))

    elif str_mode == "data":
        print "make new data file"
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        data_dir = raw_input('\tName of subdirectory for data files\t')

        if robot_type == 'Vulcan':
            test_file_list = []
            test_file_list.append('result_log_data_log_061716.mat')
            test_file_list.append('result_log_data_log_061716_2.mat')
            test_file_list.append('result_log_daily_run_101615.mat')
            exclude_file_list = []
            ## excluded data because of poorness
            exclude_file_list.append('result_log_bbb3_060616.mat')
        elif robot_type == 'Fetch':
            test_file_list = []
            test_file_list.append('result_log_fetch_test_1.mat')
            exclude_file_list = []
        elif robot_type == 'MagicBot':
            test_file_list = []
            test_file_list.append('result_test_log_011017_221849.mat')
            test_file_list.append('result_test_log_011117_220413.mat')
            exclude_file_list = []

        datasets = load_data(data_file_name, hist_window, pred_window, datadir=data_dir, testfilelist=test_file_list, excludefilelist=exclude_file_list, robot_type=robot_type, model_type=model_type)

    elif str_mode == "innerproduct":
        print "check similarity between kernels or parameters"
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'
        para_list_cnt = input('\tIndex for parameter list\t')

        with open(para_file_name, 'rb') as f2:
            robot_para_list = pickle.load(f2)
        inner_product_btw_parameter(robot_para_list, para_list_cnt)

    elif str_mode == "printpara":
        print "print parameter of convolution/fc"
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'
        para_list_cnt = input('\tIndex for parameter list\t')

        with open(para_file_name, 'rb') as f3:
            robot_para_list = pickle.load(f3)
        print robot_para_list[0].shape.eval()
        print robot_para_list[2].shape.eval()
        print robot_para_list[4].shape.eval()
        print_saved_model_parameter(robot_para_list, para_list_cnt)

    elif str_mode == "savepara":
        print "save specific parameter of CNN to mat file"
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'
        para_list_cnt = input('\tIndex for parameter list\t')
        save_file_name = raw_input('\tName of mat file to save parameter\t')
        save_file_name = save_file_name + '_parameter' + str(para_list_cnt) + '.mat'

        with open(para_file_name, 'rb') as f3:
            robot_para_list = pickle.load(f3)

        print robot_para_list[para_list_cnt].shape.eval()
        result = {}
        result["parameter"] = robot_para_list[para_list_cnt].get_value(borrow=True)
        savemat(save_file_name, result)

    elif str_mode == "print":
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'

        datasets = load_data(data_file_name, hist_window, pred_window, robot_type=robot_type, model_type=model_type)
        nan_exist = False
        if np.any(np.isnan(datasets[0][0].get_value(borrow=True))):
            print '\tNaN in train data X\t'
            nan_exist = True
        if np.any(np.isnan(datasets[0][0].get_value(borrow=True))):
            print '\tNaN in train data Y\t'
            nan_exist = True
        if np.any(np.isnan(datasets[1][0].get_value(borrow=True))):
            print '\tNaN in valid data X\t'
            nan_exist = True
        if np.any(np.isnan(datasets[1][0].get_value(borrow=True))):
            print '\tNaN in valid data Y\t'
            nan_exist = True
        if np.any(np.isnan(datasets[2][0].get_value(borrow=True))):
            print '\tNaN in test data X\t'
            nan_exist = True
        if np.any(np.isnan(datasets[2][0].get_value(borrow=True))):
            print '\tNaN in test data Y\t'
            nan_exist = True
        if np.any(np.isnan(datasets[3][0].get_value(borrow=True))):
            print '\tNaN in full test data X\t'
            nan_exist = True
        if np.any(np.isnan(datasets[3][1].get_value(borrow=True))):
            print '\tNaN in full test data Y\t'
            nan_exist = True
        if not nan_exist:
            print '\tNo Nan exists\t'

    else:
        print "no valid mode"
