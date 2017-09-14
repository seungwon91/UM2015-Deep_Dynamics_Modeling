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

from robot_data_ffnn import load_data_ffnn
from robot_nn_ffnn import FFNN3, FFNN3_FFNN3

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

##################################################################################
#############################   above Miscellaneous   ############################
#############################     below Training      ############################
##################################################################################

### function to train the model
def train_mlp(n_out, n_hidden, hist_window, pred_window, result_folder, dataset, rng, robot_type, model_struct, useSmoothed=True, learning_rate=0.00001, learning_rate_decay=0.05, n_epochs=5000, batch_size=200, patience_list=[50000, 3], start_epoch=None, paraset=None, evalset=None, decay_rate_RMSPROP=0.9, use_RMSPROP=False):

    curr_path = os.getcwd()
    if result_folder in os.listdir(curr_path):
        print "subfolder exists"
    else:
        print "Not Exist, so make that"
        os.mkdir(result_folder)

    datasets = load_data_ffnn(dataset, hist_window, pred_window, robot_type=robot_type)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    final_test_set_x, final_test_set_y = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    valid_data_num = batch_size * n_valid_batches
    test_data_num = batch_size * n_test_batches
    num_test = test_set_x.get_value(borrow=True).shape[0]

    '''
    if not (paraset is None):
        paraname = './' + result_folder + '/' + paraset
        with open(paraname, 'rb') as f3:
            tmp_para_list = pickle.load(f3)
            if len(tmp_para_list) == 2:
                best_para_list, robot_para_list = tmp_para_list[0], tmp_para_list[1]
            else:
                robot_para_list = tmp_para_list
            print "Parameter File is loaded"
    else:
        robot_para_list = paraset
    '''
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

    best_validation_loss, best_iter = np.inf, 0
    hist_valid_error, hist_test_error = [], []
    if not (evalset is None) and not (start_epoch is None):
        evalname = './' + result_folder + '/' + evalset
        result_tmp = loadmat(evalname)
        #for cnt in range(len(result_tmp['history_validation_error'])):
        for cnt in range(start_epoch):
            hist_valid_error.append([result_tmp['history_validation_error'][cnt][0], result_tmp['history_validation_error'][cnt][1], result_tmp['history_validation_error'][cnt][2]])
            hist_test_error.append([result_tmp['history_test_error'][cnt][0], result_tmp['history_test_error'][cnt][1], result_tmp['history_test_error'][cnt][2]])
            if hist_valid_error[-1][1] < best_validation_loss:
                best_validation_loss = (hist_valid_error[-1][1]**2)*valid_data_num


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    t_epoch = T.fscalar()
    x = T.matrix('x')   # input data in matrix form(num_data-by-num_features)
    y = T.matrix('y')  # output data in matrix form(num_data-by-num_outputs)

    if model_struct == 'ffnn3':
        print "\t3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list)
    elif model_struct == 'ffnn3_ffnn3':
        print "\ttwo 3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3_FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden_list=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list)
    else:
        print 'model type is not defined'
        return 0


    # theano function to compute error of model on minibatch test data
    test_model_error = theano.function(
        inputs=[index],
        outputs=robot_model(test_set_x[index * batch_size:(index + 1) * batch_size], test_set_y[index * batch_size:(index + 1) * batch_size])
    )

    # theano function to compute output of model on minibatch validation data
    validate_model_output = theano.function(
        inputs=[index],
        outputs=robot_model(valid_set_x[index * batch_size:(index + 1) * batch_size])
    )

    # theano function to compute error of model on minibatch validation data
    validate_model_error = theano.function(
        inputs=[index],
        outputs=robot_model(valid_set_x[index * batch_size:(index + 1) * batch_size], valid_set_y[index * batch_size:(index + 1) * batch_size])
    )


    # cost for training procedure
    # cost = robot_model(x, y)[1]
    cost = robot_model(x, y)[1] + 5.0 * robot_model(x, y)[2]
    # cost = T.log(np.float32(1.0) + robot_model(x, y)[2]) * robot_model(x, y)[1]

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in robot_model.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
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

    test_score = 0.
    start_time = timeit.default_timer()
    start_time_part = timeit.default_timer()

    if start_epoch is None:
        epoch = 0
    else:
        epoch = start_epoch

    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, epoch)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute error loss on validation set
                validation_losses_tmp = [validate_model_error(i) for i in range(n_valid_batches)]
                validation_losses = np.zeros(3)

                validation_losses[0:2] = np.sum(validation_losses_tmp, axis=0)[0:2]
                validation_losses[2] = np.amax(validation_losses_tmp, axis=0)[2]
                this_validation_loss = validation_losses[1]

                hist_valid_error.append([validation_losses[0]/valid_data_num, np.sqrt(validation_losses[1]/valid_data_num), validation_losses[2]])

                if np.any(np.isnan(validation_losses)):
                    print '\tIs NaN in validation_losses_tmp\t', np.any(np.isnan(validation_losses_tmp))
                    exit()

                print 'epoch %i, validation error %f/%f/%f' % (epoch, validation_losses[0]/valid_data_num, np.sqrt(validation_losses[1]/valid_data_num), validation_losses[2])
                #print 'epoch %i, validation error %f/%f/%f' % (epoch, validation_losses[0], np.sqrt(validation_losses[1]), validation_losses[2])

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses_tmp = [test_model_error(i) for i in range(n_test_batches)]
                    test_losses = np.zeros(3)

                    test_losses[0:2] = np.sum(test_losses_tmp, axis=0)[0:2]
                    test_losses[2] = np.amax(test_losses_tmp, axis=0)[2]

                    hist_test_error.append([test_losses[0]/test_data_num, np.sqrt(test_losses[1]/test_data_num), test_losses[2]])

                    print '\tepoch %i, test error of best model %f/%f/%f' % (epoch, test_losses[0]/test_data_num, np.sqrt(test_losses[1]/test_data_num), test_losses[2])
                    #print '\tepoch %i, test error of best model %f/%f/%f' % (epoch, test_losses[0], np.sqrt(test_losses[1]), test_losses[2])

                    savepicklename = './' + result_folder + '/best_mlp_model(hw'+ str(hist_window) +'_pw' +str(pred_window) +'_b' + str(batch_size)+').pkl'

                    best_para_list = list(robot_model.params)

                    # save the best model
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
                result2['input_dimension'] = 2*hist_window+2
                result2['learning_rate'] = learning_rate
                result2['batch_size'] = batch_size
                result2['epoch'] = epoch
                result2['update_threshold'] = improvement_threshold

                savefilename2 = './' + result_folder + '/Result_of_mlp_Model(hw'+ str(hist_window) +'_pw' + str(pred_window) +'_b' + str(batch_size) + ').mat'
                savemat(savefilename2, result2)

            if patience <= iter:
                done_looping = True
                break

        # save intermediate model/prediction result
        if (epoch > 0) and (epoch%50 == 0):
            savepicklename2 = './' + result_folder + '/best_mlp_model(hw'+ str(hist_window) +'_pw' +str(pred_window) +'_b'+ str(batch_size)+ '_epoch' + str(epoch) +').pkl'

            if use_RMSPROP:
                with open(savepicklename2, 'wb') as f3:
                    pickle.dump([best_para_list, best_para_grad_list, robot_model.params, list(accs)], f3)
            else:
                with open(savepicklename2, 'wb') as f3:
                    pickle.dump([best_para_list, robot_model.params], f3)

            end_time_part = timeit.default_timer()
            print '\t\tTime for sub-procedure', end_time_part - start_time_part
            start_time_part = timeit.default_timer()

    end_time = timeit.default_timer()
    print 'Optimization complete. Best validation score of %f at iteration %i, with test performance %f/%f/%f' % (best_validation_loss, best_iter + 1, test_losses[0], test_losses[1], test_losses[2])
    print 'Time for training', end_time - start_time



##################################################################################
################################# above Training #################################
#################################   below Test   #################################
##################################################################################

# function to predict output of best model
def pred_mlp(dataset, paraset, n_hidden, rng, robot_type, model_struct, n_out=2, hist_window=8, pred_window=25, batch_size=200, pred_index=[-1, -1, -1], useSmoothed=True, predMode='recursive'):
    datasets = load_data_ffnn(dataset, hist_window, pred_window, robot_type=robot_type)

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
    if model_struct == 'ffnn3':
        print "\t3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list)
    elif model_struct == 'ffnn3_ffnn3':
        print "\ttwo 3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3_FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden_list=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=pred_window, para_list=robot_para_list)
    else:
        print 'model type is not defined'
        return 0

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
        savefilename = 'Result_of_mlp_Model(hw'+ str(hist_window) +'_pw' + str(pred_window) +'_b' + str(batch_size) + 'seq).mat'

        for cnt in range(end_index-start_index):
            prediction_on_data[cnt,:] = test_model_output_seq(cnt)
        test_y = test_set_y.get_value(borrow=True)[start_index:end_index,0:2]

    elif predMode == 'recursive':
        print "\tPredict wheels' speed recursively"
        savefilename = 'Result_of_mlp_Model(hw'+ str(hist_window) +'_pw' + str(pred_window) +'_b' + str(batch_size) + 'rec).mat'


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
def test_mlp_5s(dataset, paraset, n_hidden, rng, robot_type, model_struct, n_out=2, hist_window=8, pred_window=25, batch_size=200, useSmoothed=True, saveFigure=False, result_folder=''):

    batch_size = 1

    if (robot_type == 'Vulcan'):
        plot_cmd_divider = 100.0
    elif (robot_type == 'Fetch') or (robot_type == 'Fetch2') or (robot_type == 'MagicBot'):
        plot_cmd_divider = 1.0

    datasets = load_data_ffnn(dataset, hist_window, pred_window, robot_type=robot_type)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    final_test_set_x, final_test_set_y = datasets[3]


    if not (paraset is None):
        if len(result_folder) > 0:
            paraname = './' + result_folder + '/' + paraset
        else:
            paraname = paraset

        with open(paraname, 'rb') as f3:
            tmp_para_list = pickle.load(f3)
            if len(tmp_para_list) == 2:
                best_para_list, robot_para_list = tmp_para_list[0], tmp_para_list[1]
            elif len(tmp_para_list) == 4:
                best_para_list, best_para_grad_list, robot_para_list, robot_para_grad_list = tmp_para_list[0], tmp_para_list[1], tmp_para_list[2], tmp_para_list[3]
            else:
                best_para_list = tmp_para_list[0:len(tmp_para_list)//2]
                best_para_grad_list = tmp_para_list[len(tmp_para_list)//2:len(tmp_para_list)]
            print "Parameter File is loaded"
    else:
        best_para_list = paraset
        best_para_grad_list = paraset

    index = T.lscalar()
    if model_struct == 'ffnn3':
        print "\t3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=25, para_list=best_para_list)
    elif model_struct == 'ffnn3_ffnn3':
        print "\ttwo 3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3_FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden_list=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=25, para_list=best_para_list)
    else:
        print 'model type is not defined'
        return 0

    prediction_on_data = np.zeros(final_test_set_y.get_value(borrow=True).shape, dtype=theano.config.floatX)
    test_data_in_numpy = np.zeros((batch_size,2*hist_window+50), dtype=theano.config.floatX)
    test_data_in_shared = theano.shared(np.asarray(np.zeros(test_data_in_numpy.shape),
dtype=theano.config.floatX), borrow=True)

    pred_left_speed = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125), dtype=theano.config.floatX)
    pred_right_speed = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125), dtype=theano.config.floatX)
    forward_cmd = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125), dtype=theano.config.floatX)
    left_cmd = np.zeros((final_test_set_y.get_value(borrow=True).shape[0], 125), dtype=theano.config.floatX)

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
        for in_cnt in range(5):
            if in_cnt < 1:
                test_data_in_numpy[:,:] = final_test_set_x.get_value(borrow=True)[cnt*batch_size:(cnt+1)*batch_size, 0:(2*hist_window+50)]
            else:
                test_data_in_numpy[:,0:2*hist_window] = prediction_on_data[cnt*batch_size:(cnt+1)*batch_size, in_cnt*50-2*hist_window:in_cnt*50]
                test_data_in_numpy[:,2*hist_window:2*hist_window+50] = final_test_set_x.get_value(borrow=True)[cnt*batch_size:(cnt+1)*batch_size, (2*hist_window+in_cnt*50):(2*hist_window+(in_cnt+1)*50)]

            test_data_in_shared.set_value(test_data_in_numpy)
            prediction_on_data[cnt*batch_size:(cnt+1)*batch_size, in_cnt*50:(in_cnt+1)*50] = test_model_output_for_remains()

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
        for in_cnt in range(5):
            if in_cnt < 1:
                test_data_in_numpy[0:n_remain,:] = final_test_set_x.get_value(borrow=True)[n_iter*batch_size:n_test, 0:(2*hist_window+50)]
            else:
                test_data_in_numpy[0:n_remain, 0:2*hist_window] = prediction_on_data[n_iter*batch_size:n_test, in_cnt*50-2*hist_window:in_cnt*50]
                test_data_in_numpy[0:n_remain, 2*hist_window:2*hist_window+50] = final_test_set_x.get_value(borrow=True)[n_iter*batch_size:n_test, (2*hist_window+in_cnt*50):(2*hist_window+(in_cnt+1)*50)]

            test_data_in_shared.set_value(test_data_in_numpy)
            prediction_on_data[n_iter*batch_size:n_test, in_cnt*50:(in_cnt+1)*50] = test_model_output_for_remains()[0:n_remain,:]

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
        save_file_name = './' + result_folder + '/5sec_LTS_result' + paraset[14:-4] + '.mat'
    else:
        save_file_name = '5sec_LTS_result' + paraset[14:-4] + '.mat'
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
        pdf_name = result_folder + '/multiplots.pdf'
        pp = PdfPages(pdf_name)

        for cnt in range(n_test // 25 + 1):
            if cnt < n_test//25:
                plot_cnt = cnt
            else:
                plot_cnt= max_index / 25.0

            if cnt in [n_test//125, 2*n_test//125, 3*n_test//125, 4*n_test//125]:
                print "\t\t%.2f %% complete!" % (cnt*100.0/(n_test//25))

            plot_x = np.linspace(plot_cnt, plot_cnt+5, num=125, endpoint=True)
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


############################
### function to predict output of best model for reference input(step/ramp)
############################
def test_mlp_ref(dataset, paraset, n_hidden, rng, model_struct, n_out=2, hist_window=8, pred_window=25, batch_size=200, useSmoothed=True, saveFigure=False):
    datasets = loadmat(dataset)

    final_test_set_x = np.asarray(datasets['reference_input'], dtype=theano.config.floatX)

    with open(paraset, 'rb') as f2:
        tmp_para_list = pickle.load(f2)
        if len(tmp_para_list) == 2:
            robot_para_list, last_robot_para_list = tmp_para_list[0], tmp_para_list[1]
        else:
            robot_para_list = tmp_para_list
        print "Parameter File is loaded"

    index = T.lscalar()

    if model_struct == 'ffnn3':
        print "\t3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=25, para_list=robot_para_list)
    elif model_struct == 'ffnn3_ffnn3':
        print "\ttwo 3 Layer Fully-Connected Network with ReLu hiddens"
        robot_model = FFNN3_FFNN3(rng=rng, n_in=(2*hist_window+2), n_hidden_list=n_hidden, n_out=n_out, batch_size=batch_size, hist_window=hist_window, pred_window=25, para_list=robot_para_list)
    else:
        print 'model type is not defined'
        return 0

    n_test = final_test_set_x.shape[0]
    n_iter = n_test // batch_size
    n_remain = n_test - n_iter * batch_size

    prediction_on_data = np.zeros((n_test, 250), dtype=theano.config.floatX)
    test_data_in_numpy = np.zeros((batch_size,2*hist_window+50), dtype=theano.config.floatX)
    test_data_in_shared = theano.shared(np.asarray(np.zeros(test_data_in_numpy.shape),
dtype=theano.config.floatX), borrow=True)

    pred_left_speed = np.zeros((n_test, 125))
    pred_right_speed = np.zeros((n_test, 125))
    forward_cmd = np.zeros((n_test, 125))
    left_cmd = np.zeros((n_test, 125))

    # theano function for prediction of model on minibatch test data
    test_model_output_for_remains = theano.function(
        inputs=[],
        outputs=robot_model(test_data_in_shared)
    )

    print "\tPredict wheels' speed for reference input"

    start_time_rec_pred = timeit.default_timer()

    for cnt in range(n_iter):
        for in_cnt in range(5):
            if in_cnt < 1:
                test_data_in_numpy[:,:] = final_test_set_x[cnt*batch_size:(cnt+1)*batch_size, 0:(2*hist_window+50)]
            else:
                test_data_in_numpy[:,0:2*hist_window] = prediction_on_data[cnt*batch_size:(cnt+1)*batch_size, in_cnt*50-2*hist_window:in_cnt*50]
                test_data_in_numpy[:,2*hist_window:2*hist_window+50] = final_test_set_x[cnt*batch_size:(cnt+1)*batch_size, (2*hist_window+in_cnt*50):(2*hist_window+(in_cnt+1)*50)]

            test_data_in_shared.set_value(test_data_in_numpy)
            prediction_on_data[cnt*batch_size:(cnt+1)*batch_size, in_cnt*50:(in_cnt+1)*50] = test_model_output_for_remains()

        tmp_cmd = final_test_set_x[cnt*batch_size:(cnt+1)*batch_size, 2*hist_window:(2*hist_window+250)].reshape((batch_size, 125, 2))
        tmp_pred = prediction_on_data[cnt*batch_size:(cnt+1)*batch_size, :].reshape((batch_size, 125, 2))

        for in_cnt in range(batch_size):
            pred_left_speed[cnt*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,0]
            pred_right_speed[cnt*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,1]
            forward_cmd[cnt*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,0]
            left_cmd[cnt*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,1]

    if n_remain > 0:
        for in_cnt in range(5):
            if in_cnt < 1:
                test_data_in_numpy[0:n_remain,:] = final_test_set_x[n_iter*batch_size:n_test, 0:(2*hist_window+50)]
            else:
                test_data_in_numpy[0:n_remain, 0:2*hist_window] = prediction_on_data[n_iter*batch_size:n_test, in_cnt*50-2*hist_window:in_cnt*50]
                test_data_in_numpy[0:n_remain, 2*hist_window:2*hist_window+50] = final_test_set_x[n_iter*batch_size:n_test, (2*hist_window+in_cnt*50):(2*hist_window+(in_cnt+1)*50)]

            test_data_in_shared.set_value(test_data_in_numpy)
            prediction_on_data[n_iter*batch_size:n_test, in_cnt*50:(in_cnt+1)*50] = test_model_output_for_remains()[0:n_remain,:]

        tmp_cmd = final_test_set_x[n_iter*batch_size:n_test, 2*hist_window:(2*hist_window+250)].reshape((n_remain, 125, 2))
        tmp_pred = prediction_on_data[n_iter*batch_size:n_test, :].reshape((n_remain, 125, 2))

        for in_cnt in range(n_remain):
            pred_left_speed[n_iter*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,0]
            pred_right_speed[n_iter*batch_size+in_cnt,:] = tmp_pred[in_cnt,:,1]
            forward_cmd[n_iter*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,0]
            left_cmd[n_iter*batch_size+in_cnt,:] = tmp_cmd[in_cnt,:,1]

    end_time_rec_pred = timeit.default_timer()
    rec_pred_time = (end_time_rec_pred - start_time_rec_pred)/n_test
    print '\t\tresponse of model to reference input(%f)' % rec_pred_time


    save_result = {}
    save_result['model_prediction'] = prediction_on_data
    save_result['joystick_command'] = final_test_set_x[:, (2*hist_window):(2*hist_window+250)]
    save_result['time_for_simulation'] = end_time_rec_pred - start_time_rec_pred

    save_file_name = 'ref_input_result_MLP' + paraset[15:-4] + '.mat'
    savemat(save_file_name, save_result)

    # plot (lots of) graphs
    if saveFigure:
        plot_path = os.getcwd()
        if not('RefResponsePlot_MLP' in os.listdir(plot_path)):
            os.mkdir('RefResponsePlot_MLP')

        pp = PdfPages('multipage_mlp.pdf')

        for cnt in range(n_test):
            plot_x = np.linspace(0, 5, num=125, endpoint=True)
            plot_y1 = pred_left_speed[cnt,:]
            plot_y2 = pred_right_speed[cnt,:]
            plot_for_cmd = forward_cmd[cnt,:]/100.0
            plot_left_cmd = left_cmd[cnt,:]/100.0
            title1 = 'Left Wheel response to ref input ' + str(cnt)
            title2 = 'Right Wheel response to ref input ' + str(cnt)
            filename = './RefResponsePlot_MLP/' + str(cnt) + '.png'

            ymax1 = max(np.amax(plot_y1)+0.1, np.amax(plot_for_cmd)+0.1, np.amax(plot_left_cmd)+0.1, 0.25)
            ymin1 = min(np.amin(plot_y1)-0.1, np.amin(plot_for_cmd)-0.1, np.amin(plot_left_cmd)-0.1, -0.25)
            ymax2 = max(np.amax(plot_y2)+0.1, np.amax(plot_for_cmd)+0.1, np.amax(plot_left_cmd)+0.1, 0.25)
            ymin2 = min(np.amin(plot_y2)-0.1, np.amin(plot_for_cmd)-0.1, np.amin(plot_left_cmd)-0.1, -0.25)


            fig1 = plt.figure()
            plt.subplot(211)
            plt.plot(plot_x, plot_y1, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_for_cmd, 'k--', label='Forward Command')
            plt.plot(plot_x, plot_left_cmd, 'k-.', label='Left Command')
            plt.ylim(ymin1, ymax1)
            plt.title(title1)
            #plt.legend(loc=2)
            plt.subplot(212)
            plt.plot(plot_x, plot_y2, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_for_cmd, 'k--', label='Forward Command')
            plt.plot(plot_x, plot_left_cmd, 'k-.', label='Left Command')
            plt.ylim(ymin2, ymax2)
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
# THEANO_FLAGS='cuda.root=/usr/local/cuda/',device=gpu0,floatX=float32 python robot_main_ffnn.py
# THEANO_FLAGS='cuda.root=/usr/local/cuda/',device=gpu0,floatX=float32,lib.cnmem=0.5 python robot_main_ffnn.py


if __name__ == '__main__':
    theano.config.floatX = 'float32'
    theano.config.warn_float64 = 'raise'

    # update_mode T=RMSPROP, F=SGD
    update_mode = True

    ### Vulcan robot/ CNN
    #robot_type = 'Vulcan'
    #model_struct, n_hidden = 'ffnn3', 128
    #model_struct, n_hidden = 'ffnn3_ffnn3', [64, 62, 32]

    #robot_type = 'Fetch'
    #model_struct, n_hidden = 'ffnn3', 128

    robot_type = 'MagicBot'
    model_struct, n_hidden = 'ffnn3', 96

    batch_size_in = 64
    #lr, n_ep = 0.00005, 6000
    lr, lr_decay, n_ep = 0.0001, 0.05, 1000
    patience_list_in = [10000000, 3]
    #rng = np.random.RandomState(1234)
    rng = np.random.RandomState(2000)

    str_mode = raw_input('Type mode of this program\t')

    if str_mode == "train":
        print "Train new model"
        hist_window = input('\t# of history of observation\t')
        pred_window = input('\t# of recursive prediction\t')
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        result_folder_name = raw_input('\tname of subfolder for result files\t')

        train_mlp(n_out=2, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, result_folder=result_folder_name, dataset=data_file_name, rng=rng, robot_type=robot_type, model_struct=model_struct, learning_rate=lr, learning_rate_decay=lr_decay, n_epochs=n_ep, batch_size=batch_size_in, patience_list=patience_list_in, use_RMSPROP=update_mode)

    elif str_mode == "train_conti":
        print "Train new model"
        hist_window = input('\t# of history of observation\t')
        pred_window = input('\t# of recursive prediction\t')
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        result_folder_name = raw_input('\tname of subfolder for result files\t')
        start_epoch = input('\t# of epoch to start\t')

        curr_path = os.getcwd()
        result_path = curr_path + '/' + result_folder_name
        para_file_name = 'best_mlp_model(hw' + str(hist_window) + '_pw' + str(pred_window) + '_b' + str(batch_size_in) + '_epoch' + str(start_epoch) +').pkl'
        eval_file_name = 'Result_of_mlp_Model(hw' + str(hist_window) + '_pw' + str(pred_window) + '_b' + str(batch_size_in) + '_epoch' + str(start_epoch) +').mat'

        if (result_folder_name in os.listdir(curr_path)) and (para_file_name in os.listdir(result_path)) and (eval_file_name in os.listdir(result_path)):
            train_mlp(n_out=2, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, result_folder=result_folder_name, dataset=data_file_name, rng=rng, robot_type=robot_type, model_struct=model_struct, learning_rate=lr, learning_rate_decay=lr_decay, n_epochs=n_ep, batch_size=batch_size_in, patience_list=patience_list_in, start_epoch=start_epoch, paraset=para_file_name, evalset=eval_file_name, use_RMSPROP=update_mode)
        else:
            print "Model to continue training does NOT exist!"
            print "Model to continue training does NOT exist!"

    elif str_mode == "test":
        print "Test the model on existing data"
        hist_window = input('\t# of history of observation\t')
        pred_window = input('\t# of recursive prediction\t')
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'

        pred_mode = raw_input('\tMode of Prediction(sequential/recursive)\t')
        print '\tstart, end, length of prediction interval(or type -1)'
        a = input('\t\t\t')
        b = input('\t\t\t')
        c = input('\t\t\t')

        pred_mlp(n_out=2, pred_index=[a, b, c], n_hidden=n_hidden, rng=rng, robot_type=robot_type, model_struct=model_struct, hist_window=hist_window, pred_window=pred_window, batch_size=batch_size_in, dataset=data_file_name, paraset=para_file_name, predMode=pred_mode)

    elif str_mode == "test5sec":
        print "Test the model for 5 seconds simulation"
        hist_window = input('\t# of history of observation\t')
        pred_window = input('\t# of recursive prediction\t')
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

        test_mlp_5s(n_out=2, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, rng=rng, robot_type=robot_type, model_struct=model_struct, batch_size=batch_size_in, dataset=data_file_name, paraset=para_file_name, saveFigure=save_figure, result_folder=result_folder_name)

    elif str_mode == "test_ref":
        print "Test the model for 5 seconds simulation"
        hist_window = input('\t# of history of observation\t')
        pred_window = input('\t# of recursive prediction\t')
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '.mat'
        para_file_name = raw_input('\tName of model parameter file\t')
        para_file_name = para_file_name + '.pkl'

        save_figure_str = raw_input('\tSave Figure?(T/F)\t')
        if save_figure_str == 'T':
            save_figure = True
        elif save_figure_str == 'F':
            save_figure = False

        test_mlp_ref(n_out=2, n_hidden=n_hidden, hist_window=hist_window, pred_window=pred_window, rng=rng, model_struct=model_struct, batch_size=batch_size_in, dataset=data_file_name, paraset=para_file_name, saveFigure=save_figure)

    elif str_mode == "plot":
        print "plot the speed of wheels predicted from the model"
        hist_window = input('\t# of history of observation\t')
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '.pkl'
        result_file_name = raw_input('\tName of result file\t')
        result_file_name = result_file_name + '.mat'

        datasets = load_data_ffnn(data_file_name, hist_window, pred_window, robot_type=robot_type)

        test_set_x, test_set_y = datasets[2]
        result_of_model = loadmat(result_file_name)
        plot_result(result_of_model, (test_set_x.get_value(), test_set_y.get_value()))

    elif str_mode == "data":
        print "make new data file"
        hist_window = input('\t# of history of observation\t')
        pred_window = input('\t# of recursive prediction\t')
        data_file_name = raw_input('\tName of data file\t')
        data_file_name = data_file_name + '_' + str(hist_window) + '_' + str(pred_window) + '.pkl'
        data_dir = raw_input('\tName of subdirectory for data files\t')

        if robot_type == 'Vulcan':
            test_file_list = []
            test_file_list.append('result_log_data_log_061716.mat')
            test_file_list.append('result_log_data_log_061716_2.mat')
            test_file_list.append('result_log_daily_run_101615.mat')
            exclude_file_list = []
            exclude_file_list.append('result_log_bbb3_060616.mat')
        elif robot_type == 'Fetch':
            test_file_list = []
            test_file_list.append('result_log_fetch_test_1.mat')
            exclude_file_list = []
            ## exclude to reduce the amount of data for the initial phase of training
            #exclude_file_list.append('result_log_fetch_Dec21_2.mat')
            #exclude_file_list.append('result_log_fetch_Dec28_1.mat')
            #exclude_file_list.append('result_log_fetch_Dec28_2.mat')
        elif robot_type == 'MagicBot':
            test_file_list = []
            test_file_list.append('result_test_log_011017_221849.mat')
            test_file_list.append('result_test_log_011117_220413.mat')
            exclude_file_list = []

        datasets = load_data_ffnn(data_file_name, hist_window, pred_window, datadir=data_dir, testfilelist=test_file_list, excludefilelist=exclude_file_list, robot_type=robot_type)

    elif str_mode == "printpara":
        print "print parameter of convolution/mlp"
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

    else:
        print "no valid mode"
