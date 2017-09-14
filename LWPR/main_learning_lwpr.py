# export LD_LIBRARY_PATH=/usr/local/lib
# export LD_RUN_PATH=/usr/local/lib

import os, sys, timeit
import six.moves.cPickle as pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat
from lwpr import LWPR
from matplotlib.backends.backend_pdf import PdfPages


# Training function
def train_lwpr(datafile, resultfolder, max_num_train, patience_list, improvement_threshold, init_lwpr_setting, hist_window, start_epoch=0, cmd_scaler=1.0, modelfile='lwpr_model'):

    curr_path = os.getcwd()
    if resultfolder in os.listdir(curr_path):
        print "subfolder exists"
    else:
        print "Not Exist, so make subfolder"
        os.mkdir(resultfolder)

    # Load Data
    dataset = loadmat(datafile)
    train_data_x, train_data_y = dataset['train_data_x'], dataset['train_data_y']
    valid_data_x, valid_data_y = dataset['valid_data_x'], dataset['valid_data_y']

    num_data, num_valid = train_data_x.shape[0], valid_data_x.shape[0]

    speed_hw, cmd_hw = hist_window[0], hist_window[1]
    input_dim = 2*(speed_hw+cmd_hw)

    # normalize command part
    train_data_x[:, 2*speed_hw:] = train_data_x[:, 2*speed_hw:] * cmd_scaler
    valid_data_x[:, 2*speed_hw:] = valid_data_x[:, 2*speed_hw:] * cmd_scaler

    # Set-up Parameters/Model for Training Procedure
    max_num_trials = max_num_train
    improvement_threshold = improvement_threshold

    error_hist, best_model_error, prev_train_time = [], np.inf, 0
    initD, initA, penalty = init_lwpr_setting[0], init_lwpr_setting[1], init_lwpr_setting[2]
    w_gen, w_prune = init_lwpr_setting[3], init_lwpr_setting[4]

    best_model_epoch = 0

    if start_epoch < 1:
        # Initialize Two 1-Dimensional Models
        LWPR_model_left = LWPR(input_dim, 1)
        #LWPR_model_left.init_D = initD * np.eye(input_dim)
        tmp_arr = np.ones(input_dim)
        tmp_arr[input_dim-2*cmd_hw:input_dim] = init_lwpr_setting[5]
        LWPR_model_left.init_D = initD * np.diag(tmp_arr)
        LWPR_model_left.update_D = False # True
        #LWPR_model_left.init_alpha = initA * np.eye(input_dim)
        tmp_arr = np.ones(input_dim)
        tmp_arr[input_dim-2*cmd_hw:input_dim] = init_lwpr_setting[5]
        LWPR_model_left.init_alpha = initA * np.diag(tmp_arr)
        LWPR_model_left.penalty = penalty
        LWPR_model_left.meta = True
        LWPR_model_left.meta_rate = 20
        LWPR_model_left.w_gen = w_gen
        LWPR_model_left.w_prune = w_prune

        LWPR_model_right = LWPR(input_dim, 1)
        #LWPR_model_right.init_D = initD * np.eye(input_dim)
        tmp_arr = np.ones(input_dim)
        tmp_arr[input_dim-2*cmd_hw:input_dim] = init_lwpr_setting[5]
        LWPR_model_right.init_D = initD * np.diag(tmp_arr)
        LWPR_model_right.update_D = False # True
        #LWPR_model_right.init_alpha = initA * np.eye(input_dim)
        tmp_arr = np.ones(input_dim)
        tmp_arr[input_dim-2*cmd_hw:input_dim] = init_lwpr_setting[5]
        LWPR_model_right.init_alpha = initA * np.diag(tmp_arr)
        LWPR_model_right.penalty = penalty
        LWPR_model_right.meta = True
        LWPR_model_right.meta_rate = 20
        LWPR_model_right.w_gen = w_gen
        LWPR_model_right.w_prune = w_prune

        patience = patience_list[0]
    else:
        modelfile_name = './' + resultfolder + '/' + modelfile + '_left_epoch' + str(start_epoch-1) + '.bin'
        LWPR_model_left = LWPR(modelfile_name)
        print '\tRead LWPR model for left wheel(%d)' % (LWPR_model_left.num_rfs[0])

        modelfile_name = './' + resultfolder + '/' + modelfile + '_right_epoch' + str(start_epoch-1) + '.bin'
        LWPR_model_right = LWPR(modelfile_name)
        print '\tRead LWPR model for right wheel(%d)' % (LWPR_model_right.num_rfs[0])

        result_file_name = './' + resultfolder + '/Result_of_training_epoch' + str(start_epoch-1) + '.mat'
        result_file = loadmat(result_file_name)
        prev_train_time = result_file['train_time']
        patience = result_file['patience']
        best_model_error = result_file['best_model_error']
        for cnt in range(start_epoch):
            error_hist.append([result_file['history_validation_error'][cnt][0], result_file['history_validation_error'][cnt][1], result_file['history_validation_error'][cnt][2]])


    # Training Part
    model_prediction = np.zeros(valid_data_y.shape)
    tmp_x, tmp_y = np.zeros((input_dim, 1)), np.zeros((1,1))
    print 'start training'
    start_train_time = timeit.default_timer()

    for train_cnt in range(start_epoch, max_num_trials):
        if patience < train_cnt:
            break

        rand_ind = np.random.permutation(num_data)

        for data_cnt in range(num_data):
            tmp_x[:,0] = train_data_x[rand_ind[data_cnt], 0:input_dim]
            tmp_y[0,0] = train_data_y[rand_ind[data_cnt], 0]
            _ = LWPR_model_left.update(tmp_x, tmp_y)

            tmp_y[0,0] = train_data_y[rand_ind[data_cnt], 1]
            _ = LWPR_model_right.update(tmp_x, tmp_y)

            if data_cnt % 5000 == 0:
                print '\ttrain epoch %d, data index %d, #rfs=%d/%d' % (train_cnt, data_cnt, LWPR_model_left.num_rfs, LWPR_model_right.num_rfs)

        for data_cnt in range(num_valid):
            tmp_x[:,0] = valid_data_x[data_cnt, 0:input_dim]
            model_prediction[data_cnt, 0], _ = LWPR_model_left.predict_conf(tmp_x)
            model_prediction[data_cnt, 1], _ = LWPR_model_right.predict_conf(tmp_x)

        diff = abs(valid_data_y - model_prediction)

        new_error = np.asarray([np.sum(diff)/float(num_valid), np.sqrt(np.sum(diff**2)/float(num_valid)), np.max(diff)])
        error_hist.append([new_error[0], new_error[1], new_error[2]])

        # save result of one training epoch
        modelfile_name = './' + resultfolder + '/' + modelfile + '_left_epoch' + str(train_cnt) + '.bin'
        LWPR_model_left.write_binary(modelfile_name)

        modelfile_name = './' + resultfolder + '/' + modelfile + '_right_epoch' + str(train_cnt) + '.bin'
        LWPR_model_right.write_binary(modelfile_name)

        if new_error[1] < best_model_error * improvement_threshold:
            best_model_epoch = train_cnt
            best_model_error = new_error[1]
            patience = max(patience, min(train_cnt+10, int(train_cnt * patience_list[1])) )

            modelfile_name = './' + resultfolder + '/' + modelfile + '_best_left_epoch' + str(train_cnt) + '.bin'
            LWPR_model_left.write_binary(modelfile_name)

            modelfile_name = './' + resultfolder + '/' + modelfile + '_best_right_epoch' + str(train_cnt) + '.bin'
            LWPR_model_right.write_binary(modelfile_name)

        result_file_name = './' + resultfolder + '/Result_of_training_epoch' + str(train_cnt) + '.mat'
        result = {}
        result['train_time'] = timeit.default_timer() - start_train_time + prev_train_time
        result['best_model_error'] = best_model_error
        result['history_validation_error'] = error_hist
        result['patience'] = patience
        result['improvement_threshold'] = improvement_threshold
        result['init_D'] = initD
        result['init_alpha'] = initA
        result['penalty'] = penalty
        result['w_generate_criterion'] = w_gen
        result['w_prune_criterion'] = w_prune
        result['number_speed_in_input'] = 2*speed_hw
        result['number_cmd_in_input'] = 2*cmd_hw
        savemat(result_file_name, result)

        print '\n\tSave Intermediate Result Successfully'
        print '\t%d-th learning : #Data=%d/%d, #rfs=%d/%d, error=%f\n' %(train_cnt, LWPR_model_left.n_data, LWPR_model_right.n_data, LWPR_model_left.num_rfs, LWPR_model_right.num_rfs, error_hist[train_cnt][1])

    print 'end training'
    return best_model_epoch


############################
### function to predict output of best model for 5 seconds simulation
############################
def test_lwpr_5sec(datafile, resultfolder, model_epoch, hist_window, robot_type, cmd_scaler=1.0, modelfile='lwpr_model', saveFigure=False):

    if (robot_type == 'Vulcan'):
        plot_cmd_divider = 100.0
    elif (robot_type == 'Fetch') or (robot_type == 'Fetch2') or (robot_type == 'MagicBot'):
        plot_cmd_divider = 1.0

    # load data file
    dataset = loadmat(datafile)
    test_data_x, test_data_y = dataset['test_data_x'], dataset['test_data_y']
    num_test, test_dim = test_data_x.shape[0], test_data_y.shape[1]//2
    speed_hw, cmd_hw = hist_window[0], hist_window[1]
    input_dim = 2*(speed_hw+cmd_hw)

    # normalize command part
    test_data_x[:, 2*speed_hw:] = test_data_x[:, 2*speed_hw:] * cmd_scaler

    # load model files
    modelfile_name = './' + resultfolder + '/' + modelfile + '_best_left_epoch' + str(model_epoch) + '.bin'
    best_left_model = LWPR(modelfile_name)
    print 'Read Left model (%d)' % (best_left_model.num_rfs[0])

    modelfile_name = './' + resultfolder + '/' + modelfile + '_best_right_epoch' + str(model_epoch) + '.bin'
    best_right_model = LWPR(modelfile_name)
    print 'Read Right model (%d)' % (best_right_model.num_rfs[0])

    result_file_name = './' + resultfolder + '/Result_of_training_epoch' + str(model_epoch) + '.mat'
    result_mat = loadmat(result_file_name)
    train_time = result_mat['train_time']
    hist_valid_error = result_mat['history_validation_error']

    # start making 5 seconds simulation
    pred_on_test = np.zeros((num_test, 2*test_dim))
    tmp_x = np.zeros((input_dim, 1))

    print 'start prediction on test data'
    start_test_time = timeit.default_timer()
    for data_cnt in range(num_test):
        for pred_cnt in range(test_dim):
            if pred_cnt < speed_hw:
                num_value_from_data = 2*(speed_hw-pred_cnt)
                tmp_x[0:num_value_from_data, 0] = test_data_x[data_cnt, 2*pred_cnt:2*speed_hw]
                tmp_x[num_value_from_data:2*speed_hw, 0] = pred_on_test[data_cnt, 0:2*pred_cnt]
            else:
                tmp_x[0:2*speed_hw, 0] = pred_on_test[data_cnt, 2*(pred_cnt-speed_hw):2*pred_cnt]

            tmp_x[2*speed_hw:input_dim, 0] = test_data_x[data_cnt, 2*(speed_hw+pred_cnt):2*(speed_hw+pred_cnt+cmd_hw)]

            pred_on_test[data_cnt, 2*pred_cnt], _ = best_left_model.predict_conf(tmp_x)
            pred_on_test[data_cnt, 2*pred_cnt+1], _ = best_right_model.predict_conf(tmp_x)

        if data_cnt%5000 == 0:
            print '\t\t', data_cnt
    end_test_time = timeit.default_timer()

    diff = abs(test_data_y[0:num_test, 0:2*test_dim] - pred_on_test)
    max_index = diff.argmax() // (test_dim*2)
    error = np.asarray([np.sum(diff)/float(num_test), np.sqrt(np.sum(diff**2)/float(num_test)), np.max(diff)])

    print 'Error on Test Data! %f/%f/%f' %(error[0], error[1], error[2])

    save_file_name = './' + resultfolder + '/LWPR_1D_model_test_result.mat'

    result={}
    result['train_time'] = train_time
    result['test_time'] = end_test_time - start_test_time
    result['history_validation_error'] = hist_valid_error
    result['test_error'] = error
    result['model_output_on_test_data'] = pred_on_test
    result['joystick_command'] = test_data_x[:, (input_dim-2):(input_dim-2+250)]
    savemat(save_file_name, result)


    # plot (lots of) graphs
    if saveFigure:
        real_y_tmp, pred_y_tmp, joystick_cmd_tmp = test_data_y.reshape((num_test,125,2)), pred_on_test.reshape((num_test,125,2)), test_data_x[:, 2*(input_dim//2-1):2*(input_dim//2+124)].reshape((num_test,125,2))
        real_y_left, real_y_right = np.zeros((num_test,125)), np.zeros((num_test,125))
        pred_left_speed, pred_right_speed = np.zeros((num_test,125)), np.zeros((num_test,125))
        forward_cmd, left_cmd = np.zeros((num_test,125)), np.zeros((num_test,125))
        for cnt in range(num_test):
            real_y_left[cnt,:] = real_y_tmp[cnt,:,0]
            real_y_right[cnt,:] = real_y_tmp[cnt,:,1]
            pred_left_speed[cnt,:] = pred_y_tmp[cnt,:,0]
            pred_right_speed[cnt,:] = pred_y_tmp[cnt,:,1]
            forward_cmd[cnt,:] = joystick_cmd_tmp[cnt,:,0]
            left_cmd[cnt,:] = joystick_cmd_tmp[cnt,:,1]

        plot_path = os.getcwd()
        plot_path = plot_path + '/' + resultfolder
        if not('ResultPlot' in os.listdir(plot_path)):
            dir_path = plot_path + '/ResultPlot'
            os.mkdir(dir_path)

        pdf_name = plot_path + '/LWPR_test_plot.pdf'
        pp = PdfPages(pdf_name)

        for cnt in range(num_test // 25 + 1):
            if cnt < num_test//25:
                plot_cnt = cnt
            else:
                plot_cnt= max_index / 25.0
            plot_x = np.linspace(plot_cnt, plot_cnt+5, num=125, endpoint=True)
            plot_y1 = pred_left_speed[int(plot_cnt*25),:]
            plot_y2 = pred_right_speed[int(plot_cnt*25),:]
            plot_real_y1 = real_y_left[int(plot_cnt*25),:]
            plot_real_y2 = real_y_right[int(plot_cnt*25),:]
            plot_for_cmd = forward_cmd[int(plot_cnt*25),:]/plot_cmd_divider
            plot_left_cmd = left_cmd[int(plot_cnt*25),:]/plot_cmd_divider
            title1 = 'Left Wheel 5sec Prediction from ' + str(plot_cnt)
            title2 = 'Right Wheel 5sec Prediction from ' + str(plot_cnt)
            filename = plot_path + '/ResultPlot/5secLTS_' + str(plot_cnt) + '.png'

            fig1 = plt.figure()
            plt.subplot(211)
            plt.plot(plot_x, plot_y1, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_real_y1, 'b-.', label='Encoded Speed')
            if (robot_type == 'Vulcan'):
                plt.plot(plot_x, plot_for_cmd, 'k--', label='Forward Command')
                plt.plot(plot_x, plot_left_cmd, 'k-.', label='Left Command')
            elif (robot_type == 'Fetch') or (robot_type == 'Fetch2') or (robot_type == 'MagicBot'):
                plt.plot(plot_x, plot_for_cmd, 'k--', label='Left Command')
            plt.title(title1)
            #plt.legend(loc=2)
            plt.subplot(212)
            plt.plot(plot_x, plot_y2, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_real_y2, 'b-.', label='Encoded Speed')
            if (robot_type == 'Vulcan'):
                plt.plot(plot_x, plot_for_cmd, 'k--', label='Forward Command')
                plt.plot(plot_x, plot_left_cmd, 'k-.', label='Left Command')
            elif (robot_type == 'Fetch') or (robot_type == 'Fetch2') or (robot_type == 'MagicBot'):
                plt.plot(plot_x, plot_left_cmd, 'k--', label='Right Command')
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
def test_lwpr_ref(datafile, resultfolder, model_epoch, hist_window, saveFigure=False, modelfile='lwpr_model'):

    if (robot_type == 'Vulcan'):
        plot_cmd_divider = 100.0
    elif (robot_type == 'Fetch') or (robot_type == 'Fetch2') or (robot_type == 'MagicBot'):
        plot_cmd_divider = 1.0

    # load data file
    dataset = loadmat(datafile)
    test_data_x = dataset['reference_input']
    num_test, test_dim = test_data_x.shape[0], 125
    speed_hw, cmd_hw = hist_window[0], hist_window[1]
    input_dim = 2*(speed_hw+cmd_hw)

    # load model files
    modelfile_name = './' + resultfolder + '/' + modelfile + '_best_left_epoch' + str(model_epoch) + '.bin'
    best_left_model = LWPR(modelfile_name)
    print 'Read Left model (%d)' % (best_left_model.num_rfs[0])

    modelfile_name = './' + resultfolder + '/' + modelfile + '_best_right_epoch' + str(model_epoch) + '.bin'
    best_right_model = LWPR(modelfile_name)
    print 'Read Right model (%d)' % (best_right_model.num_rfs[0])

    # start making 5 seconds simulation
    pred_on_test = np.zeros((num_test, 2*test_dim))
    tmp_x = np.zeros((input_dim, 1))

    pred_left_speed = np.zeros((num_test, 125))
    pred_right_speed = np.zeros((num_test, 125))
    forward_cmd = np.zeros((num_test, 125))
    left_cmd = np.zeros((num_test, 125))

    print 'start prediction on reference input'
    start_test_time = timeit.default_timer()
    for data_cnt in range(num_test):
        for pred_cnt in range(test_dim):
            if pred_cnt < speed_hw:
                num_value_from_data = 2*(speed_hw-pred_cnt)
                tmp_x[0:num_value_from_data, 0] = test_data_x[data_cnt, 2*pred_cnt:2*speed_hw]
                tmp_x[num_value_from_data:2*speed_hw, 0] = pred_on_test[data_cnt, 0:2*pred_cnt]
            else:
                tmp_x[0:2*speed_hw, 0] = pred_on_test[data_cnt, 2*(pred_cnt-speed_hw):2*pred_cnt]

            tmp_x[2*speed_hw:input_dim, 0] = test_data_x[data_cnt, 2*(speed_hw+pred_cnt):2*(speed_hw+pred_cnt+cmd_hw)]

            pred_on_test[data_cnt, 2*pred_cnt], _ = best_left_model.predict_conf(tmp_x)
            pred_on_test[data_cnt, 2*pred_cnt+1], _ = best_right_model.predict_conf(tmp_x)

        tmp_cmd = test_data_x[data_cnt, (input_dim-2):(input_dim-2+250)].reshape((125, 2))
        tmp_pred = pred_on_test[data_cnt, :].reshape((125, 2))

        pred_left_speed[data_cnt,:] = tmp_pred[:,0]
        pred_right_speed[data_cnt,:] = tmp_pred[:,1]
        forward_cmd[data_cnt,:] = tmp_cmd[:,0]
        left_cmd[data_cnt,:] = tmp_cmd[:,1]

        if data_cnt%1000 == 0:
            print '\t\t', data_cnt
    end_test_time = timeit.default_timer()
    print 'finish prediction on reference input'

    save_file_name = './' + resultfolder + '/ref_input_result_LWPR.mat'

    result={}
    result['test_time'] = end_test_time - start_test_time
    result['model_output_on_test_data'] = pred_on_test
    result['joystick_command'] = test_data_x[:, (input_dim-2):(input_dim-2+250)]
    savemat(save_file_name, result)

    # plot (lots of) graphs
    if saveFigure:
        plot_path = os.getcwd()
        plot_path = plot_path + '/' + resultfolder
        if not('RefResponsePlot_LWPR' in os.listdir(plot_path)):
            dir_path = plot_path + '/RefResponsePlot_LWPR'
            os.mkdir(dir_path)

        pdf_name = plot_path + '/LWPR_ref_response_plot.pdf'
        pp = PdfPages(pdf_name)

        for cnt in range(num_test):
            plot_x = np.linspace(0, 5, num=125, endpoint=True)
            plot_y1 = pred_left_speed[cnt,:]
            plot_y2 = pred_right_speed[cnt,:]
            plot_cmd1 = forward_cmd[cnt,:]/plot_cmd_divider
            plot_cmd2 = left_cmd[cnt,:]/plot_cmd_divider
            title1 = 'Left Wheel response to ref input ' + str(cnt)
            title2 = 'Right Wheel response to ref input ' + str(cnt)
            filename = plot_path + '/RefResponsePlot_LWPR/' + str(cnt) + '.png'

            #ymax1 = max(np.amax(plot_y1)+0.1, np.amax(plot_cmd1)+0.1, np.amax(plot_cmd2)+0.1, 0.25)
            #ymin1 = min(np.amin(plot_y1)-0.1, np.amin(plot_cmd1)-0.1, np.amin(plot_cmd2)-0.1, -0.25)
            #ymax2 = max(np.amax(plot_y2)+0.1, np.amax(plot_cmd1)+0.1, np.amax(plot_cmd2)+0.1, 0.25)
            #ymin2 = min(np.amin(plot_y2)-0.1, np.amin(plot_cmd1)-0.1, np.amin(plot_cmd2)-0.1, -0.25)
            ymax1 = max(np.amax(plot_y1)+0.1, np.amax(plot_cmd1)+0.1, 0.25)
            ymin1 = min(np.amin(plot_y1)-0.1, np.amin(plot_cmd1)-0.1, -0.25)
            ymax2 = max(np.amax(plot_y2)+0.1, np.amax(plot_cmd2)+0.1, 0.25)
            ymin2 = min(np.amin(plot_y2)-0.1, np.amin(plot_cmd2)-0.1, -0.25)


            fig1 = plt.figure()
            plt.subplot(211)
            plt.plot(plot_x, plot_y1, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_cmd1, 'k--', label='Left Wheel Command')
            #plt.plot(plot_x, plot_cmd1, 'k--', label='Forward Command')
            #plt.plot(plot_x, plot_cmd2, 'k-.', label='Left Command')
            plt.ylim(ymin1, ymax1)
            plt.title(title1)
            #plt.legend(loc=2)
            plt.subplot(212)
            plt.plot(plot_x, plot_y2, 'r--', label='Predicted Speed')
            plt.plot(plot_x, plot_cmd2, 'k--', label='Right Wheel Command')
            #plt.plot(plot_x, plot_cmd1, 'k--', label='Forward Command')
            #plt.plot(plot_x, plot_cmd2, 'k-.', label='Left Command')
            plt.ylim(ymin2, ymax2)
            plt.title(title2)
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 1.0))

            fig1.savefig(filename, bbox_inches='tight', pad_inches=0)
            pp.savefig()
            plt.close()

        pp.close()
        #plt.show()


#from inspect import getmembers
#print getmembers(LWPR)


# Main function
if __name__ == '__main__':
    max_num_trials = 70
    patience_list = [10, 1.5]
    improvement_threshold = 0.99
    init_lwpr_setting = [0.06, 150, (10**-4), 0.1, 0.9, 40] #initD, initA, penalty, w_gen, w_prune, multiplier of initD, initA

    robot_type = 'Vulcan'
    #robot_type = 'MagicBot'

    str_mode = raw_input('Type mode of this program\t')

    if str_mode == "train":
        print "Train new model"
        data_file_name = raw_input('\tName of data file\t')
        speed_hw = input('\tNumber of steps of speed\t')
        cmd_hw = input('\tNumber of steps of command\t')
        data_file_name = data_file_name + '(hw' + str(speed_hw) + '_' + str(cmd_hw) + ').mat'
        result_folder_name = raw_input('\tname of subfolder for result files\t')

        best_model_epoch = train_lwpr(datafile=data_file_name, resultfolder=result_folder_name, max_num_train=max_num_trials, patience_list=patience_list, improvement_threshold=improvement_threshold, init_lwpr_setting=init_lwpr_setting, hist_window=[speed_hw, cmd_hw])

    elif str_mode == "train_conti":
        print "Train model continuously"
        data_file_name = raw_input('\tName of data file\t')
        speed_hw = input('\tNumber of steps of speed\t')
        cmd_hw = input('\tNumber of steps of command\t')
        data_file_name = data_file_name + '(hw' + str(speed_hw) + '_' + str(cmd_hw) + ').mat'
        result_folder_name = raw_input('\tname of subfolder for result files\t')
        start_epoch = input('\t# of epoch to start(epoch# of existing + 1)\t')

        train_lwpr(datafile=data_file_name, resultfolder=result_folder_name, max_num_train=max_num_trials, patience_list=patience_list, improvement_threshold=improvement_threshold, init_lwpr_setting=init_lwpr_setting, start_epoch=start_epoch, hist_window=[speed_hw, cmd_hw])

    elif str_mode == "test5sec":
        print "Test model for 5 sec simulation"
        data_file_name = raw_input('\tName of data file\t')
        speed_hw = input('\tNumber of steps of speed\t')
        cmd_hw = input('\tNumber of steps of command\t')
        data_file_name = data_file_name + '(hw' + str(speed_hw) + '_' + str(cmd_hw) + ').mat'
        result_folder_name = raw_input('\tname of subfolder for model files\t')
        result_model_epoch = input('\t# of epoch of model to test\t')
        save_fig = raw_input('\tsave the figure of responses?(T/F)\t')
        if save_fig == 'T':
            save_fig = True
        else:
            save_fig = False

        test_lwpr_5sec(datafile=data_file_name, resultfolder=result_folder_name, model_epoch=result_model_epoch, saveFigure=save_fig, hist_window=[speed_hw, cmd_hw], robot_type=robot_type)

    elif str_mode == "test_ref":
        print "Test model for reference input"
        data_file_name = raw_input('\tName of data file\t')
        speed_hw = input('\tNumber of steps of speed\t')
        cmd_hw = input('\tNumber of steps of command\t')
        data_file_name = data_file_name + '(hw' + str(speed_hw) + '_' + str(cmd_hw) + ').mat'
        result_folder_name = raw_input('\tname of subfolder for model files\t')
        result_model_epoch = input('\t# of epoch of model to test\t')
        save_fig = raw_input('\tsave the figure of responses?(T/F)\t')
        if save_fig == 'T':
            save_fig = True
        else:
            save_fig = False

        test_lwpr_ref(datafile=data_file_name, resultfolder=result_folder_name, model_epoch=result_model_epoch, saveFigure=save_fig, hist_window=[speed_hw, cmd_hw], robot_type=robot_type)

    else:
        print "\tWrong mode! Try again!\n"
