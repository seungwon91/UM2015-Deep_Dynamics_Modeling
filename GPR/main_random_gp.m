close all;  clear;	clc;

% for window
% addpath(genpath('C:\Users\Seungwon\Documents\MATLAB\lib\gpml-matlab-v3.6-2015-07-07'));
% for ubuntu
addpath(genpath('/media/leeswon/Data/Research/Robot_Modeling/GPR/gpml-matlab-v3.6-2015-07-07'));
% for lab computer
%addpath(genpath('/home/kuiperslab-desktop/Documents/robot_modeling/GPR/gpml-matlab-v3.6-2015-07-07'));
% for CAEN computer
% addpath(genpath('./gpml-matlab-v3.6-2015-07-07'));

obs_hist_window = 5;
cmd_hist_window = 1;

robot_type = 'vulcan';
% robot_type = 'fetch';
% robot_type = 'magicbot';

load(sprintf('train_data_%s(hw%d_%d).mat', robot_type, obs_hist_window, cmd_hist_window));
load(sprintf('test_data_%s(hw%d_%d).mat', robot_type, obs_hist_window, cmd_hist_window));
num_train_data = size(train_data_x, 1);
num_valid_data = size(valid_data_x, 1);
num_test_data = size(test_data_x, 1);
num_dim = size(train_data_x, 2);


size_model = 2000;
num_epoch = 30;

folder_name = sprintf('./subsampled_model_%s(size%d_hw%d_%d)', robot_type, size_model, obs_hist_window, cmd_hist_window);

if exist(folder_name)~= 7
    fprintf(1, 'Folder does NOT exist. Make new folder\n');
    mkdir(folder_name);
else
    fprintf(1, 'Folder exists\n');
end


for train_epoch = 1:num_epoch
    fprintf(1, 'Start Process for %d (%s)\n', train_epoch, datestr(now));
    
    % initialization
    GP_model_left = [];
    GP_model_right = [];
    
    GP_model_left.num_total = num_train_data;
    GP_model_right.num_total = num_train_data;
    
    data_order_index = randperm(num_train_data);
    GP_model_left.data_index = data_order_index(1:size_model);
    data_order_index = randperm(num_train_data);
    GP_model_right.data_index = data_order_index(1:size_model);
    
    
    % hyperparameter setting
    h_init = 1.2*ones(num_dim+2,1);          % [cov_kernel in_exponential hyps;cov_kernel amplitude hyp;lik hyp]
    GP_model_left = hyp_setting(GP_model_left, h_init);
    GP_model_right = hyp_setting(GP_model_right, h_init);
    
    
    GP_model_left.kernel = @(x1,x2,h) kernFunc(x1,x2,h);
    GP_model_right.kernel = @(x1,x2,h) kernFunc(x1,x2,h);
    
    % gp type setting
    GP_model_left.covfunc = @covSEard;
    GP_model_left.meanfunc = @meanZero;
    GP_model_left.likfunc = @likGauss;
    
    GP_model_right.covfunc = @covSEard;
    GP_model_right.meanfunc = @meanZero;
    GP_model_right.likfunc = @likGauss;
    
    % parameter for slice-sampling
    GP_model_left.slicesample.num_hyp = 35;
    GP_model_left.slicesample.burnin = 31;
    
    GP_model_right.slicesample.num_hyp = 35;
    GP_model_right.slicesample.burnin = 31;

    % marginal likelihood function
    GP_model_left.log_marg_lik = @(h) gp_feval_transform(h, GP_model_left.meanfunc, ...
                                                     GP_model_left.covfunc, GP_model_left.likfunc, ...
                                                     train_data_x(GP_model_left.data_index,:), ...
                                                     train_data_y(GP_model_left.data_index,1));

    GP_model_right.log_marg_lik = @(h) gp_feval_transform(h, GP_model_right.meanfunc, ...
                                                     GP_model_right.covfunc, GP_model_right.likfunc, ...
                                                     train_data_x(GP_model_right.data_index,:), ...
                                                     train_data_y(GP_model_right.data_index,2));
    
	% hyperparameter optimization
	model_hyp_para = [];
	new_hyp = hyp_opt_slice_sampling(GP_model_left);
	GP_model_left = hyp_setting(GP_model_left,new_hyp);
	if size(new_hyp,1) > size(new_hyp,2)
        model_hyp_para = new_hyp';
    else
        model_hyp_para = new_hyp;
    end
	new_hyp = hyp_opt_slice_sampling(GP_model_right);
	GP_model_right = hyp_setting(GP_model_right,new_hyp);
	if size(new_hyp,1) > size(new_hyp,2)
        model_hyp_para = [model_hyp_para;new_hyp'];
    else
        model_hyp_para = [model_hyp_para;new_hyp];
	end
    fprintf(1, '\t\tEnd optimization, Start prediction on validation %s\n', datestr(now));
                                                 
	% Start Prediction on Test data
        % kernel matrix
	[GP_model_left.K_approx, GP_model_left.L_approx] = GP_KL_generator(GP_model_left, ...
                                                                       train_data_x(GP_model_left.data_index,:));
	GP_model_left.L_appAndNoise = jitChol( (GP_model_left.L_approx * GP_model_left.L_approx' + ...
                                            GP_model_left.hyp_para(end)* ...
                                            eye(length(GP_model_left.data_index))), 5, 'lower' );
	%GP_model_left.inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
	tmp_inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
	GP_model_left.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * tmp_inv_L_appAndNoise;
    GP_model_left.inv_L_appAndNoise_sqr_y = GP_model_left.inv_L_appAndNoise_sqr * ...
                                            train_data_y(GP_model_left.data_index, 1);
    
    [GP_model_right.K_approx, GP_model_right.L_approx] = GP_KL_generator(GP_model_right, ...
                                                                         train_data_x(GP_model_right.data_index,:));
    GP_model_right.L_appAndNoise = jitChol( (GP_model_right.L_approx * GP_model_right.L_approx' + ...
                                             GP_model_right.hyp_para(end)* ...
                                             eye(length(GP_model_right.data_index))), 5, 'lower' );
    %GP_model_right.inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
    tmp_inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
    GP_model_right.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * tmp_inv_L_appAndNoise;
    GP_model_right.inv_L_appAndNoise_sqr_y = GP_model_right.inv_L_appAndNoise_sqr * ...
                                             train_data_y(GP_model_right.data_index, 2);
    clear tmp_inv_L_appAndNoise;
    
    
    num_pred = 4;
    pred_on_test_data = zeros(size(test_data_y, 1), 2*num_pred);
    tic;
    for data_cnt = 1:num_test_data
        for pred_cnt = 1:num_pred
            if pred_cnt <= obs_hist_window
                tmp_input = [test_data_x(data_cnt, 2*(pred_cnt-1)+1:2*obs_hist_window), ... % input_history
                             pred_on_test_data(data_cnt, 1:2*(pred_cnt-1)), ...           % input_history
                             test_data_x(data_cnt, 2*(obs_hist_window+pred_cnt-1)+1:2*(obs_hist_window+pred_cnt+cmd_hist_window-1))];  % command_signal
            else
                tmp_input = [pred_on_test_data(data_cnt, 2*(pred_cnt-obs_hist_window)-1:2*(pred_cnt-1)), ... % input_history
                             test_data_x(data_cnt, 2*(obs_hist_window+pred_cnt-1)+1:2*(obs_hist_window+pred_cnt+cmd_hist_window-1))];  % command_signal
            end

            % left_wheel_case
            [pred_on_test_data(data_cnt, 2*pred_cnt-1), ~] = GPR_predict(GP_model_left, train_data_x, ...
                                                                         train_data_y(:, 1), tmp_input, 2);

            % right_wheel_case
            [pred_on_test_data(data_cnt, 2*pred_cnt), ~] = GPR_predict(GP_model_right, train_data_x, ...
                                                                       train_data_y(:, 2), tmp_input, 2);
        end

        if mod(data_cnt, floor(num_test_data/5)) == 0
            fprintf('\t\t%.3f - %s\n', data_cnt*100/num_test_data, datestr(now));
        end
    end
    pred_time = toc;

    diff = pred_on_test_data - test_data_y(:, 1:2*num_pred);
    error = zeros(3,1);
    error(1) = sum(sum(abs(diff)))/num_test_data;
    error(2) = sqrt(sum(sum(diff.^2))/num_test_data);
    error(3) = max(max(abs(diff), [], 2));

    fprintf(1, '\tL1 Error(MAE) : %.5f, L2 Error(RMS) : %.5f, Linf Error : %.5f\n', error(1), error(2), error(3));
    fprintf(1, '\tComputational Time Consumption : %.5f(avg. %f)\n', pred_time, pred_time/num_test_data);
  
    % Save Result Model & Prediction Result
	file_name = sprintf('%s/RandomGPRModel(%d).mat', folder_name, train_epoch);
	model_data_index = [GP_model_left.data_index; GP_model_right.data_index];
	save(file_name, 'model_data_index', 'model_hyp_para', 'pred_on_test_data', 'test_data_y', 'pred_time', 'error');
    fprintf(1, 'End %d-th model generation\n\n', train_epoch);
end