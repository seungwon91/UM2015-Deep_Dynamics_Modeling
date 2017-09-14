import six.moves.cPickle as pickle
import os
import sys

from scipy.io import loadmat
import numpy as np
import theano

##############################################################################################
# create data file for long term simulation
#   data_x : left wheel speed[ind-hist_window], right wheel speed[ind-hist_window]
#            / left wheel speed[ind-hist_window+1], right wheel speed[ind-hist_window+1]
#            ... / left wheel speed[ind-1], right wheel speed[ind-1]
#            / joystick forward[ind-1], joystick left[ind-1]
#            / joystick forward[ind], joystick left[ind]
#            ... / joystick forward[ind+pred_window-2], joystick left[ind+pred_window-2]
#
#   data_y : left wheel speed[ind], right wheel speed[ind] 
#            / left wheel speed[ind+1], right wheel speed[ind+1]
#            ... / left wheel speed[ind+pred_window-1], right wheel speed[ind+pred_window-1]

def process_data_vulcan_ffnn(mat_data, hist_window, pred_window, useSmoothed, mixData=True, makeTrainValid=False, zeroPadding=False):
    if zeroPadding:
        num_data = mat_data['Log']['plotTime'][0][0].shape[0]
        for_cmd, lat_cmd, left_wheel, right_wheel = np.zeros(num_data+hist_window), np.zeros(num_data+hist_window), np.zeros(num_data+hist_window), np.zeros(num_data+hist_window)
        for_cmd[hist_window:num_data+hist_window] = mat_data['Log']['forwardCommand'][0][0][:,0]
        lat_cmd[hist_window:num_data+hist_window] = mat_data['Log']['leftCommand'][0][0][:,0]

        # use/not use smoothed speed of both wheels
        if useSmoothed:
            left_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderLeftWheelSpeedSmoothed'][0][0][:,0]
            right_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderRightWheelSpeedSmoothed'][0][0][:,0]
        else:
            left_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderLeftWheelSpeed'][0][0][:,0]
            right_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderRightWheelSpeed'][0][0][:,0]

        num_data = num_data + 1 - pred_window

    else:
        num_data = mat_data['Log']['plotTime'][0][0].shape[0]+1-pred_window-hist_window
        for_cmd, lat_cmd = mat_data['Log']['forwardCommand'][0][0][:,0], mat_data['Log']['leftCommand'][0][0][:,0]

        # use/not use smoothed speed of both wheels
        if useSmoothed:
            left_wheel = mat_data['Log']['encoderLeftWheelSpeedSmoothed'][0][0][:,0]
            right_wheel = mat_data['Log']['encoderRightWheelSpeedSmoothed'][0][0][:,0]
        else:
            left_wheel = mat_data['Log']['encoderLeftWheelSpeed'][0][0][:,0]
            right_wheel = mat_data['Log']['encoderRightWheelSpeed'][0][0][:,0]


    # mix the order of data
    if mixData:
        data_index = np.arange(num_data)
        np.random.shuffle(data_index)

    # whether make only one data set or two disjoint data sets
    if makeTrainValid:
        data_split_cnt = int(num_data*0.90)
        train_data_in, train_data_out = np.zeros((data_split_cnt, 2*hist_window+2*pred_window)), np.zeros((data_split_cnt,2*pred_window))
        valid_data_in, valid_data_out = np.zeros((num_data-data_split_cnt, 2*hist_window+2*pred_window)), np.zeros((num_data-data_split_cnt,2*pred_window))
    else:
        data_in, data_out = np.zeros((num_data, 2*hist_window+2*pred_window)), np.zeros((num_data,2*pred_window))

    # data re-allocation
    for cnt in range(num_data):
        tmp, tmp2 = np.zeros(2*hist_window+2*pred_window), np.zeros(2*pred_window)
        if mixData:
            ind = data_index[cnt] + hist_window
        else:
            ind = cnt + hist_window


        for cnt2 in range(hist_window):
            tmp[2*cnt2] = left_wheel[ind-hist_window+cnt2]
            tmp[2*cnt2+1] = right_wheel[ind-hist_window+cnt2]
        '''
        tmp[0:hist_window] = left_wheel[ind-hist_window:ind]
        tmp[hist_window:2*hist_window] = right_wheel[ind-hist_window:ind]
        '''

        for cnt2 in range(pred_window):
            tmp[2*hist_window+2*cnt2] = for_cmd[ind-1+cnt2]
            tmp[2*hist_window+2*cnt2+1] = lat_cmd[ind-1+cnt2]
            tmp2[2*cnt2] = left_wheel[ind+cnt2]
            tmp2[2*cnt2+1] = right_wheel[ind+cnt2]

        if makeTrainValid:
            if ind < data_split_cnt+hist_window:
                train_data_in[ind-hist_window,:] = tmp
                train_data_out[ind-hist_window,:] = tmp2
            else:
                valid_data_in[ind-hist_window-data_split_cnt,:] = tmp
                valid_data_out[ind-hist_window-data_split_cnt,:] = tmp2
        else:
            data_in[ind-hist_window,:] = tmp
            data_out[ind-hist_window,:] = tmp2

    # result return
    if makeTrainValid:
        return [(train_data_in,train_data_out),(valid_data_in,valid_data_out)]
    else:
        return (data_in, data_out)


##############################################################################################
# create data file for long term simulation
#   data_x : linear velocity[ind-hist_window], angular velocity[ind-hist_window]
#            / linear velocity[ind-hist_window+1], angular velocity[ind-hist_window+1]
#            ... / linear velocity[ind-1], angular velocity[ind-1]
#            / joystick forward[ind-1], joystick lateral[ind-1]
#            / joystick forward[ind], joystick lateral[ind]
#            ... / joystick forward[ind+pred_window-2], joystick lateral[ind+pred_window-2]
#
#   data_y : linear velocity[ind], angular velocity[ind] 
#            / linear velocity[ind+1], angular velocity[ind+1]
#            ... / linear velocity[ind+pred_window-1], angular velocity[ind+pred_window-1]

def process_data_fetch_ffnn(mat_data, hist_window, pred_window, useSmoothed, mixData=True, makeTrainValid=False, zeroPadding=False):
    if zeroPadding:
        num_data = mat_data['Log']['plotTime'][0][0].shape[0]
        for_cmd, lat_cmd, lin_vel, ang_vel = np.zeros(num_data+hist_window), np.zeros(num_data+hist_window), np.zeros(num_data+hist_window), np.zeros(num_data+hist_window)
        for_cmd[hist_window:num_data+hist_window] = mat_data['Log']['forwardCommand'][0][0][:,0]
        lat_cmd[hist_window:num_data+hist_window] = mat_data['Log']['lateralCommand'][0][0][:,0]

        # use/not use smoothed speed of both wheels
        if useSmoothed:
            lin_vel[hist_window:num_data+hist_window] = mat_data['Log']['odometryLinearSpeedSmoothed'][0][0][:,0]
            ang_vel[hist_window:num_data+hist_window] = mat_data['Log']['odometryAngularSpeedSmoothed'][0][0][:,0]
        else:
            lin_vel[hist_window:num_data+hist_window] = mat_data['Log']['odometryLinearSpeed'][0][0][:,0]
            ang_vel[hist_window:num_data+hist_window] = mat_data['Log']['odometryAngularSpeed'][0][0][:,0]

        num_data = num_data + 1 - pred_window

    else:
        num_data = mat_data['Log']['plotTime'][0][0].shape[0]+1-pred_window-hist_window
        for_cmd, lat_cmd = mat_data['Log']['forwardCommand'][0][0][:,0], mat_data['Log']['lateralCommand'][0][0][:,0]

        # use/not use smoothed speed of both wheels
        if useSmoothed:
            lin_vel = mat_data['Log']['odometryLinearSpeedSmoothed'][0][0][:,0]
            ang_vel = mat_data['Log']['odometryAngularSpeedSmoothed'][0][0][:,0]
        else:
            lin_vel = mat_data['Log']['odometryLinearSpeed'][0][0][:,0]
            ang_vel = mat_data['Log']['odometryAngularSpeed'][0][0][:,0]


    # mix the order of data
    if mixData:
        data_index = np.arange(num_data)
        np.random.shuffle(data_index)

    # whether make only one data set or two disjoint data sets
    if makeTrainValid:
        data_split_cnt = int(num_data*0.92)
        train_data_in, train_data_out = np.zeros((data_split_cnt, 2*hist_window+2*pred_window)), np.zeros((data_split_cnt,2*pred_window))
        valid_data_in, valid_data_out = np.zeros((num_data-data_split_cnt, 2*hist_window+2*pred_window)), np.zeros((num_data-data_split_cnt,2*pred_window))
    else:
        data_in, data_out = np.zeros((num_data, 2*hist_window+2*pred_window)), np.zeros((num_data,2*pred_window))

    # data re-allocation
    for cnt in range(num_data):
        tmp, tmp2 = np.zeros(2*hist_window+2*pred_window), np.zeros(2*pred_window)
        if mixData:
            ind = data_index[cnt] + hist_window
        else:
            ind = cnt + hist_window


        for cnt2 in range(hist_window):
            tmp[2*cnt2] = lin_vel[ind-hist_window+cnt2]
            tmp[2*cnt2+1] = ang_vel[ind-hist_window+cnt2]

        for cnt2 in range(pred_window):
            tmp[2*hist_window+2*cnt2] = for_cmd[ind-1+cnt2]
            tmp[2*hist_window+2*cnt2+1] = lat_cmd[ind-1+cnt2]
            tmp2[2*cnt2] = lin_vel[ind+cnt2]
            tmp2[2*cnt2+1] = ang_vel[ind+cnt2]

        if makeTrainValid:
            if ind < data_split_cnt+hist_window:
                train_data_in[ind-hist_window,:] = tmp
                train_data_out[ind-hist_window,:] = tmp2
            else:
                valid_data_in[ind-hist_window-data_split_cnt,:] = tmp
                valid_data_out[ind-hist_window-data_split_cnt,:] = tmp2
        else:
            data_in[ind-hist_window,:] = tmp
            data_out[ind-hist_window,:] = tmp2

    # result return
    if makeTrainValid:
        return [(train_data_in,train_data_out),(valid_data_in,valid_data_out)]
    else:
        return (data_in, data_out)


##############################################################################################
# create data file for long term simulation
#   data_x : left velocity[ind-hist_window], right velocity[ind-hist_window]
#            / left velocity[ind-hist_window+1], right velocity[ind-hist_window+1]
#            ... / left velocity[ind-1], right velocity[ind-1]
#            / left command[ind-1], right command[ind-1]
#            / left command[ind], right command[ind]
#            ... / left command[ind+pred_window-2], right command[ind+pred_window-2]
#
#   data_y : left velocity[ind], right velocity[ind] 
#            / left velocity[ind+1], right velocity[ind+1]
#            ... / left velocity[ind+pred_window-1], right velocity[ind+pred_window-1]

def process_data_magicbot_ffnn(mat_data, hist_window, pred_window, useSmoothed, mixData=True, makeTrainValid=False, zeroPadding=False):
    if zeroPadding:
        num_data = mat_data['Log']['plotTime'][0][0].shape[0]
        left_cmd, right_cmd, left_wheel, right_wheel = np.zeros(num_data+hist_window), np.zeros(num_data+hist_window), np.zeros(num_data+hist_window), np.zeros(num_data+hist_window)
        left_cmd[hist_window:num_data+hist_window] = mat_data['Log']['leftWheelCommand'][0][0][:,0]
        right_cmd[hist_window:num_data+hist_window] = mat_data['Log']['rightWheelCommand'][0][0][:,0]

        # use/not use smoothed speed of both wheels
        if useSmoothed:
            left_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderLeftWheelSpeedSmoothed'][0][0][:,0]
            right_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderRightWheelSpeedSmoothed'][0][0][:,0]
        else:
            left_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderLeftWheelSpeed'][0][0][:,0]
            right_wheel[hist_window:num_data+hist_window] = mat_data['Log']['encoderRightWheelSpeed'][0][0][:,0]

        num_data = num_data + 1 - pred_window

    else:
        num_data = mat_data['Log']['plotTime'][0][0].shape[0]+1-pred_window-hist_window
        left_cmd, right_cmd = mat_data['Log']['leftWheelCommand'][0][0][:,0], mat_data['Log']['rightWheelCommand'][0][0][:,0]

        # use/not use smoothed speed of both wheels
        if useSmoothed:
            left_wheel = mat_data['Log']['encoderLeftWheelSpeedSmoothed'][0][0][:,0]
            right_wheel = mat_data['Log']['encoderRightWheelSpeedSmoothed'][0][0][:,0]
        else:
            left_wheel = mat_data['Log']['encoderLeftWheelSpeed'][0][0][:,0]
            right_wheel = mat_data['Log']['encoderRightWheelSpeed'][0][0][:,0]


    # mix the order of data
    if mixData:
        data_index = np.arange(num_data)
        np.random.shuffle(data_index)

    # whether make only one data set or two disjoint data sets
    if makeTrainValid:
        data_split_cnt = int(num_data*0.92)
        train_data_in, train_data_out = np.zeros((data_split_cnt, 2*hist_window+2*pred_window)), np.zeros((data_split_cnt,2*pred_window))
        valid_data_in, valid_data_out = np.zeros((num_data-data_split_cnt, 2*hist_window+2*pred_window)), np.zeros((num_data-data_split_cnt,2*pred_window))
        train_cnt, valid_cnt = 0, 0
    else:
        data_in, data_out = np.zeros((num_data, 2*hist_window+2*pred_window)), np.zeros((num_data,2*pred_window))
        data_cnt = 0

    # data re-allocation
    for cnt in range(num_data):
        tmp, tmp2 = np.zeros(2*hist_window+2*pred_window), np.zeros(2*pred_window)
        if mixData:
            ind = data_index[cnt] + hist_window
        else:
            ind = cnt + hist_window


        for cnt2 in range(hist_window):
            tmp[2*cnt2] = left_wheel[ind-hist_window+cnt2]
            tmp[2*cnt2+1] = right_wheel[ind-hist_window+cnt2]
        '''
        tmp[0:hist_window] = left_wheel[ind-hist_window:ind]
        tmp[hist_window:2*hist_window] = right_wheel[ind-hist_window:ind]
        '''

        for cnt2 in range(pred_window):
            tmp[2*hist_window+2*cnt2] = left_cmd[ind-1+cnt2]
            tmp[2*hist_window+2*cnt2+1] = right_cmd[ind-1+cnt2]
            tmp2[2*cnt2] = left_wheel[ind+cnt2]
            tmp2[2*cnt2+1] = right_wheel[ind+cnt2]

        if makeTrainValid:
            if cnt < data_split_cnt:
                #train_data_in[ind-hist_window,:] = tmp
                #train_data_out[ind-hist_window,:] = tmp2
                train_data_in[train_cnt,:] = tmp
                train_data_out[train_cnt,:] = tmp2
                train_cnt += 1
            else:
                #valid_data_in[ind-hist_window-data_split_cnt,:] = tmp
                #valid_data_out[ind-hist_window-data_split_cnt,:] = tmp2
                valid_data_in[valid_cnt,:] = tmp
                valid_data_out[valid_cnt,:] = tmp2
                valid_cnt += 1
        else:
            #data_in[ind-hist_window,:] = tmp
            #data_out[ind-hist_window,:] = tmp2
            data_in[data_cnt,:] = tmp
            data_out[data_cnt,:] = tmp2
            data_cnt += 1

    # result return
    if makeTrainValid:
        return [(train_data_in,train_data_out),(valid_data_in,valid_data_out)]
    else:
        return (data_in, data_out)


##############################################################################################
def load_data_ffnn(data_file_name, hist_window, pred_window, useSmoothed=True, datadir=None, testfilelist=None, excludefilelist=None, robot_type='Vulcan', model_type='FFNN'):
    data_path = os.getcwd()
    datafilelist = os.listdir(data_path)

    if (robot_type == 'Vulcan') and (model_type == 'FFNN'):
        process_data_func = process_data_vulcan_ffnn
    elif (robot_type == 'Fetch') and (model_type == 'FFNN'):
        process_data_func = process_data_fetch_ffnn
    elif (robot_type == 'MagicBot') and (model_type == 'FFNN'):
        process_data_func = process_data_magicbot_ffnn
    else:
        print "Data Loading Fails - Robot and Model type is wrong"

    if ((datadir is None) or (testfilelist is None)) or (data_file_name in datafilelist):
        print "Load existing data file"
        with open(data_file_name, 'rb') as f:
            try:
                train_set, valid_set, test_set, final_test_set = pickle.load(f, encoding='latin1')
                print "Data Loading Complete!"
                print "\ttrain data %d, valid data %d, test data %d" % (train_set[0].shape[0], valid_set[0].shape[0], test_set[0].shape[0])
            except:
                train_set, valid_set, test_set, final_test_set = pickle.load(f)
                print "Data Loading Complete!"
                print "\ttrain data %d, valid data %d, test data %d" % (train_set[0].shape[0], valid_set[0].shape[0], test_set[0].shape[0])

    else:
        print "Make new data file"
        data_path = os.getcwd() + '/' + datadir + '/'
        datafilelist = os.listdir(data_path)
        num_datafile = len(datafilelist)

        train_cnt, test_cnt = 0, 0
        for data_file_cnt in range(num_datafile):
            print 'File ', datafilelist[data_file_cnt], ' is processed'
            if (datafilelist[data_file_cnt] in testfilelist) and (os.path.isfile(data_path+datafilelist[data_file_cnt])):
                test_data_mat = loadmat(data_path+datafilelist[data_file_cnt])
                if test_cnt < 1:
                    test_data_in, test_data_out = process_data_func(test_data_mat, hist_window, pred_window, useSmoothed, mixData=False, makeTrainValid=False, zeroPadding=True)
                    final_test_data_in, final_test_data_out = process_data_func(test_data_mat, hist_window, 125, useSmoothed, mixData=False, makeTrainValid=False, zeroPadding=True)
                    test_cnt = test_cnt + 1
                else:
                    test_data_tmp_in, test_data_tmp_out = process_data_func(test_data_mat, hist_window, pred_window, useSmoothed, mixData=False, makeTrainValid=False, zeroPadding=True)
                    test_data_in = np.concatenate((test_data_in, test_data_tmp_in), axis=0)
                    test_data_out = np.concatenate((test_data_out, test_data_tmp_out), axis=0)

                    final_test_data_tmp_in, final_test_data_tmp_out = process_data_func(test_data_mat, hist_window, 125, useSmoothed, mixData=False, makeTrainValid=False, zeroPadding=True)
                    final_test_data_in = np.concatenate((final_test_data_in, final_test_data_tmp_in), axis=0)
                    final_test_data_out = np.concatenate((final_test_data_out, final_test_data_tmp_out), axis=0)
                print '\t', test_data_in.shape, test_data_out.shape
                print '\t', final_test_data_in.shape, final_test_data_out.shape

            elif not(datafilelist[data_file_cnt] in excludefilelist) and (os.path.isfile(data_path+datafilelist[data_file_cnt])):
                train_data_mat = loadmat(data_path+datafilelist[data_file_cnt])
                if train_cnt < 0.5:
                    datasets = process_data_func(train_data_mat, hist_window, pred_window, useSmoothed, mixData=True, makeTrainValid=True, zeroPadding=False)
                    train_data_in, train_data_out = datasets[0]
                    valid_data_in, valid_data_out = datasets[1]
                    train_cnt = train_cnt + 1
                else:
                    datasets = process_data_func(train_data_mat, hist_window, pred_window, useSmoothed, mixData=True, makeTrainValid=True, zeroPadding=False)
                    train_data_tmp_in, train_data_tmp_out = datasets[0]
                    valid_data_tmp_in, valid_data_tmp_out = datasets[1]
                    train_data_in = np.concatenate((train_data_in, train_data_tmp_in), axis=0)
                    train_data_out = np.concatenate((train_data_out, train_data_tmp_out), axis=0)
                    valid_data_in = np.concatenate((valid_data_in, valid_data_tmp_in), axis=0)
                    valid_data_out = np.concatenate((valid_data_out, valid_data_tmp_out), axis=0)
                print '\t', train_data_in.shape, train_data_out.shape
                print '\t', valid_data_in.shape, valid_data_out.shape

        train_set = (train_data_in, train_data_out)
        valid_set = (valid_data_in, valid_data_out)
        test_set = (test_data_in, test_data_out)
        final_test_set = (final_test_data_in, final_test_data_out)

        # save processed data
        with open(data_file_name, 'wb') as f:
            pickle.dump([train_set, valid_set, test_set, final_test_set], f)
        print 'New Data File is Saved'


    # train_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix, number of data x number of feature)
    # target is a numpy.ndarray of 2 dimension (a matrix, number of data x number of outputs)

    # Needed to use GPU
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    final_test_set_x, final_test_set_y = shared_dataset(final_test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), (final_test_set_x, final_test_set_y)]
    return rval
