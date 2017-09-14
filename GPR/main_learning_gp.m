%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.08.28 Updated
% Main Program for Learning Robot's Physics Model using Gaussian
% Process Regression method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

% robot_type = 'vulcan';
% robot_type = 'fetch';
robot_type = 'magicbot';

if strcmp(robot_type, 'vulcan')
    cmd_scaler = 1/100.0;
elseif strcmp(robot_type, 'fetch') || strcmp(robot_type, 'magicbot')
    cmd_scaler = 1.0;
end

load(sprintf('train_data_%s(hw%d_%d).mat', robot_type, obs_hist_window, cmd_hist_window));
load(sprintf('test_data_%s(hw%d_%d).mat', robot_type, obs_hist_window, cmd_hist_window));

% train_data_x = [train_data_x(:, 1:2*obs_hist_window), train_data_x(:, 2*obs_hist_window+1:2*(obs_hist_window+cmd_hist_window))*cmd_scaler];
% valid_data_x = [valid_data_x(:, 1:2*obs_hist_window), valid_data_x(:, 2*obs_hist_window+1:2*(obs_hist_window+cmd_hist_window))*cmd_scaler];
% test_data_x = [test_data_x(:, 1:2*obs_hist_window), test_data_x(:, 2*obs_hist_window+1:end)*cmd_scaler];


num_train_data = size(train_data_x, 1);
num_valid_data = size(valid_data_x, 1);
num_test_data = size(test_data_x, 1);
num_dim = size(train_data_x, 2);

train_boost = false;
    num_models = 15;    % matters only when train_boost == True
size_model = 4000;
num_epoch = 30;

if train_boost
    folder_name = sprintf('./subsampled_boost_model_%s(size%d_%d_hw%d_%d)/', robot_type, num_models, size_model, obs_hist_window, cmd_hist_window);
else
    %folder_name = sprintf('./subsampled_scaled_model_%s(size%d_hw%d_%d)', robot_type, size_model, obs_hist_window, cmd_hist_window);
    folder_name = sprintf('./subsampled_model_%s(size%d_hw%d_%d)', robot_type, size_model, obs_hist_window, cmd_hist_window);
end

if exist(folder_name)~= 7
    fprintf(1, 'Folder does NOT exist. Make new folder\n');
    mkdir(folder_name);
else
    fprintf(1, 'Folder exists\n');
end


%% Make GPR model

if train_boost
    fprintf('Boost mode\n');
    % initialization
    GP_boost_model_left = [];
    GP_boost_model_right = [];

    h_init = 1.2*ones(num_dim+2,1);          % [cov_kernel in_exponential hyps;cov_kernel amplitude hyp;lik hyp]

    % variables to save information about models
    model_data_index = zeros(num_models, size_model);
    left_hyper_parameter = zeros(num_models, num_dim+2);
    right_hyper_parameter = zeros(num_models, num_dim+2);

    for a = 1:num_models
        data_order_index = randperm(num_train_data);
        model_data_index(a,:) = data_order_index(1:size_model);

        GP_boost_model_left{a}.num_total = num_train_data;
        GP_boost_model_left{a}.data_index = data_order_index(1:size_model);

        GP_boost_model_right{a}.num_total = num_train_data;
        GP_boost_model_right{a}.data_index = data_order_index(1:size_model);

        % hyperparameter setting
        GP_boost_model_left{a} = hyp_setting(GP_boost_model_left{a},h_init);
        GP_boost_model_left{a}.kernel = @(x1,x2,h) kernFunc(x1,x2,h);

        GP_boost_model_right{a} = hyp_setting(GP_boost_model_right{a},h_init);
        GP_boost_model_right{a}.kernel = @(x1,x2,h) kernFunc(x1,x2,h);

        % gp type setting
        GP_boost_model_left{a}.covfunc = @covSEard;
        GP_boost_model_left{a}.meanfunc = @meanZero;
        GP_boost_model_left{a}.likfunc = @likGauss;

        GP_boost_model_right{a}.covfunc = @covSEard;
        GP_boost_model_right{a}.meanfunc = @meanZero;
        GP_boost_model_right{a}.likfunc = @likGauss;

        % parameter for slice-sampling
        GP_boost_model_left{a}.slicesample.num_hyp = 20;
        GP_boost_model_left{a}.slicesample.burnin = 15;

        GP_boost_model_right{a}.slicesample.num_hyp = 20;
        GP_boost_model_right{a}.slicesample.burnin = 15;

        % marginal likelihood function
        GP_boost_model_left{a}.log_marg_lik = @(h) gp_feval_transform(h, GP_boost_model_left{a}.meanfunc, ...
                                                          GP_boost_model_left{a}.covfunc, GP_boost_model_left{a}.likfunc, ...
                                                          train_data_x(GP_boost_model_left{a}.data_index,:), ...
                                                          train_data_y(GP_boost_model_left{a}.data_index,1));

        GP_boost_model_right{a}.log_marg_lik = @(h) gp_feval_transform(h, GP_boost_model_right{a}.meanfunc, ...
                                                          GP_boost_model_right{a}.covfunc, GP_boost_model_right{a}.likfunc, ...
                                                          train_data_x(GP_boost_model_right{a}.data_index,:), ...
                                                          train_data_y(GP_boost_model_right{a}.data_index,2));
    end

    % Slice Sampling for Hyperparameter Optimization
    fprintf(1, 'Initialization Ends / Hyperparameter Optimization Starts!\n');
%     folder_name = './result/';
    if ~exist(folder_name)
        mkdir(folder_name);
    end
    file_name = sprintf('%s/LearnedModel_GPR_%d.mat', folder_name, num_models);

    for a = 1:num_models
        % left wheel
        tic;
        new_hyp = hyp_opt_slice_sampling(GP_boost_model_left{a});
    %     new_hyp = hyp_opt_gp_minimize(GP_boost_model_left{a});
        GP_boost_model_left{a} = hyp_setting(GP_boost_model_left{a},new_hyp);
        left_hyper_parameter(a, :) = new_hyp;
        t = toc;
        fprintf(1, 'Left Wheel(%d) - Opt Ends(time : %f)\n', a, t);

        % right wheel
        tic;
        new_hyp = hyp_opt_slice_sampling(GP_boost_model_right{a});
    %     new_hyp = hyp_opt_gp_minimize(GP_boost_model_right{a});
        GP_boost_model_right{a} = hyp_setting(GP_boost_model_right{a},new_hyp);
        right_hyper_parameter(a, :) = new_hyp;
        t = toc;
        fprintf(1, 'Right Wheel(%d) - Opt Ends(time : %f)\n', a, t);

        %save(file_name,'GP_boost_model_left','GP_boost_model_right', '-v7.3');
        save(file_name,'model_data_index','left_hyper_parameter','right_hyper_parameter', '-v7.3');
        fprintf(1, 'Successfully Save File\n\n');
    end

    clear a t new_hyp folder_name file_name data_order_index h_init;
else
    fprintf('Single model mode\n');
    start_epoch = input('epoch of model to start training(first learned model has epoch+1)  ');
    
    % initialization
    GP_model_left = [];
    GP_model_right = [];
    
    GP_model_left.num_total = num_train_data;
    GP_model_right.num_total = num_train_data;
    
    if start_epoch < 0
        data_order_index = randperm(num_train_data);
        GP_model_left.data_index = data_order_index(1:size_model);
        data_order_index = randperm(num_train_data);
        GP_model_right.data_index = data_order_index(1:size_model);
    else
        file_name = sprintf('%s/LearnedModel_GPR(size%d_epoch%d).mat', folder_name, size_model, start_epoch);
        load(file_name);
        
        GP_model_left.data_index = model_data_index(1,:);
        GP_model_right.data_index = model_data_index(2,:);
    end
    
    % hyperparameter setting
    if start_epoch < 0
        h_init = 1.2*ones(num_dim+2,1);          % [cov_kernel in_exponential hyps;cov_kernel amplitude hyp;lik hyp]
        GP_model_left = hyp_setting(GP_model_left, h_init);
        GP_model_right = hyp_setting(GP_model_right, h_init);
    else
        GP_model_left = hyp_setting(GP_model_left, model_hyp_para(1,:));
        GP_model_right = hyp_setting(GP_model_right, model_hyp_para(2,:));
    end
    
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
    GP_model_left.slicesample.num_hyp = 25;
    GP_model_left.slicesample.burnin = 23;
    
    GP_model_right.slicesample.num_hyp = 25;
    GP_model_right.slicesample.burnin = 23;

    % marginal likelihood function
    GP_model_left.log_marg_lik = @(h) gp_feval_transform(h, GP_model_left.meanfunc, ...
                                                     GP_model_left.covfunc, GP_model_left.likfunc, ...
                                                     train_data_x(GP_model_left.data_index,:), ...
                                                     train_data_y(GP_model_left.data_index,1));

    GP_model_right.log_marg_lik = @(h) gp_feval_transform(h, GP_model_right.meanfunc, ...
                                                     GP_model_right.covfunc, GP_model_right.likfunc, ...
                                                     train_data_x(GP_model_right.data_index,:), ...
                                                     train_data_y(GP_model_right.data_index,2));

    % Slice Sampling for Hyperparameter Optimization
    if start_epoch < 0
        fprintf(1, 'Initialization Ends / Hyperparameter Optimization Starts!(%s)\n', datestr(now));
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
        
        % initialize kernel matrix for new subsamples & hyperparameter
        [GP_model_left.K_approx, GP_model_left.L_approx] = GP_KL_generator(GP_model_left, ...
                                                                           train_data_x(GP_model_left.data_index,:));
        GP_model_left.L_appAndNoise = jitChol( (GP_model_left.L_approx * GP_model_left.L_approx' + ...
                                                GP_model_left.hyp_para(end)* ...
                                                eye(length(GP_model_left.data_index))), 5, 'lower' );
        tmp_inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
        GP_model_left.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                              tmp_inv_L_appAndNoise;
        GP_model_left.inv_L_appAndNoise_sqr_y = GP_model_left.inv_L_appAndNoise_sqr * ...
                                                train_data_y(GP_model_left.data_index, 1);

        [GP_model_right.K_approx, GP_model_right.L_approx] = GP_KL_generator(GP_model_right, ...
                                                                             train_data_x(GP_model_right.data_index,:));
        GP_model_right.L_appAndNoise = jitChol( (GP_model_right.L_approx * GP_model_right.L_approx' + ...
                                                GP_model_right.hyp_para(end)* ...
                                                eye(length(GP_model_right.data_index))), 5, 'lower' );
        tmp_inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
        GP_model_right.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                               tmp_inv_L_appAndNoise;
        GP_model_right.inv_L_appAndNoise_sqr_y = GP_model_right.inv_L_appAndNoise_sqr * ...
                                                 train_data_y(GP_model_right.data_index, 1);
        
        output_on_valid = zeros(num_valid_data, 2);
        for valid_cnt = 1:num_valid_data
            [output_on_valid(valid_cnt, 1), ~] = GPR_predict(GP_model_left, train_data_x, ...
                                                             train_data_y(:, 1), valid_data_x(valid_cnt, :), 2);
            [output_on_valid(valid_cnt, 2), ~] = GPR_predict(GP_model_right, train_data_x, ...
                                                             train_data_y(:, 2), valid_data_x(valid_cnt, :), 2);
        end
        valid_error = sum(sum((valid_data_y(1:num_valid_data,:) - output_on_valid).^2))/(2*num_valid_data);
        
        if ~exist(folder_name)
            mkdir(folder_name);
        end
        file_name = sprintf('%s/LearnedModel_GPR(size%d_epoch0).mat', folder_name, size_model);
        model_data_index = [GP_model_left.data_index; GP_model_right.data_index];
        fprintf(1, '\t\tEnd prediction on validation %s/error %f\n', datestr(now), valid_error);
        save(file_name, 'model_data_index', 'model_hyp_para', 'valid_error');
    elseif ~exist('valid_error', 'var')
        % initialize kernel matrix for new subsamples & hyperparameter
        [GP_model_left.K_approx, GP_model_left.L_approx] = GP_KL_generator(GP_model_left, ...
                                                                           train_data_x(GP_model_left.data_index,:));
        GP_model_left.L_appAndNoise = jitChol( (GP_model_left.L_approx * GP_model_left.L_approx' + ...
                                                GP_model_left.hyp_para(end)* ...
                                                eye(length(GP_model_left.data_index))), 5, 'lower' );
        tmp_inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
        GP_model_left.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                              tmp_inv_L_appAndNoise;
        GP_model_left.inv_L_appAndNoise_sqr_y = GP_model_left.inv_L_appAndNoise_sqr * ...
                                                train_data_y(GP_model_left.data_index, 1);

        [GP_model_right.K_approx, GP_model_right.L_approx] = GP_KL_generator(GP_model_right, ...
                                                                             train_data_x(GP_model_right.data_index,:));
        GP_model_right.L_appAndNoise = jitChol( (GP_model_right.L_approx * GP_model_right.L_approx' + ...
                                                GP_model_right.hyp_para(end)* ...
                                                eye(length(GP_model_right.data_index))), 5, 'lower' );
        tmp_inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
        GP_model_right.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                               tmp_inv_L_appAndNoise;
        GP_model_right.inv_L_appAndNoise_sqr_y = GP_model_right.inv_L_appAndNoise_sqr * ...
                                                 train_data_y(GP_model_right.data_index, 1);
        
        fprintf(1, '\t\tNo validation error. Start prediction %s\n', datestr(now));
        output_on_valid = zeros(num_valid_data, 2);
        for valid_cnt = 1:num_valid_data
            [output_on_valid(valid_cnt, 1), ~] = GPR_predict(GP_model_left, train_data_x, ...
                                                             train_data_y(:, 1), valid_data_x(valid_cnt, :), 2);
            [output_on_valid(valid_cnt, 2), ~] = GPR_predict(GP_model_right, train_data_x, ...
                                                             train_data_y(:, 2), valid_data_x(valid_cnt, :), 2);
        end
        valid_error = sum(sum((valid_data_y(1:num_valid_data,:) - output_on_valid).^2))/(2*num_valid_data);
        fprintf(1, '\t\tEnd prediction on validation %s/error %f\n', datestr(now), valid_error);
        save(file_name, 'model_data_index', 'model_hyp_para', 'valid_error');
    end
    fprintf(1, 'End Initial Set-up(%s)\n', datestr(now));
    
    
    % update subsamples of data
    for epoch_cnt = start_epoch+1:num_epoch
        % change subsampled data at random
        data_update_cnt = 0;
        while data_update_cnt <= 14
            % make temporary models
            left_tmp_model = GP_model_left; left_tmp_model.log_marg_lik = [];
            right_tmp_model = GP_model_right;   right_tmp_model.log_marg_lik = [];
            
            % change 20 samples randomly
            for update_cnt = 1:int16(size_model/200)
                ind_in_subsample = randi([1, size_model]);
                ind_out_subsample = randi([1, num_train_data]);
                while ~isempty(find(left_tmp_model.data_index == ind_out_subsample, 1))
                    ind_out_subsample = randi([1, num_train_data]);
                end
                left_tmp_model.data_index(ind_in_subsample) = ind_out_subsample;
            end
            for update_cnt = 1:int16(size_model/200)
                ind_in_subsample = randi([1, size_model]);
                ind_out_subsample = randi([1, num_train_data]);
                while ~isempty(find(right_tmp_model.data_index == ind_out_subsample, 1))
                    ind_out_subsample = randi([1, num_train_data]);
                end
                right_tmp_model.data_index(ind_in_subsample) = ind_out_subsample;
            end
            
            % make kernel matrices to speed up the prediction
            [left_tmp_model.K_approx, left_tmp_model.L_approx] = GP_KL_generator(left_tmp_model, ...
                                                                               train_data_x(left_tmp_model.data_index,:));
            left_tmp_model.L_appAndNoise = jitChol( (left_tmp_model.L_approx * left_tmp_model.L_approx' + ...
                                                    left_tmp_model.hyp_para(end)* ...
                                                    eye(length(left_tmp_model.data_index))), 5, 'lower' );
            tmp_inv_L_appAndNoise = inv(left_tmp_model.L_appAndNoise);
            left_tmp_model.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                                  tmp_inv_L_appAndNoise;
            left_tmp_model.inv_L_appAndNoise_sqr_y = left_tmp_model.inv_L_appAndNoise_sqr * ...
                                                    train_data_y(left_tmp_model.data_index, 1);

            [right_tmp_model.K_approx, right_tmp_model.L_approx] = GP_KL_generator(right_tmp_model, ...
                                                                                 train_data_x(right_tmp_model.data_index,:));
            right_tmp_model.L_appAndNoise = jitChol( (right_tmp_model.L_approx * right_tmp_model.L_approx' + ...
                                                    right_tmp_model.hyp_para(end)* ...
                                                    eye(length(right_tmp_model.data_index))), 5, 'lower' );
            tmp_inv_L_appAndNoise = inv(right_tmp_model.L_appAndNoise);
            right_tmp_model.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                                   tmp_inv_L_appAndNoise;
            right_tmp_model.inv_L_appAndNoise_sqr_y = right_tmp_model.inv_L_appAndNoise_sqr * ...
                                                     train_data_y(right_tmp_model.data_index, 1);
            
            % prediction on validation data with new models
            output_on_valid = zeros(num_valid_data, 2);
            for valid_cnt = 1:num_valid_data
                [output_on_valid(valid_cnt, 1), ~] = GPR_predict(left_tmp_model, train_data_x, ...
                                                                 train_data_y(:, 1), valid_data_x(valid_cnt, :), 2);
                [output_on_valid(valid_cnt, 2), ~] = GPR_predict(right_tmp_model, train_data_x, ...
                                                                 train_data_y(:, 2), valid_data_x(valid_cnt, :), 2);
            end
            new_valid_error = sum(sum((valid_data_y(1:num_valid_data,:) - output_on_valid).^2))/(2*num_valid_data);
            
            % if new model has better validation error => update model
            if new_valid_error < valid_error
                data_update_cnt = data_update_cnt + 1;
                GP_model_left.data_index = left_tmp_model.data_index;
                GP_model_right.data_index = right_tmp_model.data_index;
                fprintf(1, '\tUpdate subsamples of model\n');
            else
                fprintf(1, '\tFail to update subsamples of model\n');
            end
        end
        
        % update log marginal likelihood function of models
        GP_model_left.log_marg_lik = @(h) gp_feval_transform(h, GP_model_left.meanfunc, ...
                                                         GP_model_left.covfunc, GP_model_left.likfunc, ...
                                                         train_data_x(GP_model_left.data_index,:), ...
                                                         train_data_y(GP_model_left.data_index,1));

        GP_model_right.log_marg_lik = @(h) gp_feval_transform(h, GP_model_right.meanfunc, ...
                                                         GP_model_right.covfunc, GP_model_right.likfunc, ...
                                                         train_data_x(GP_model_right.data_index,:), ...
                                                         train_data_y(GP_model_right.data_index,2));
        
        % change hyperparameter
        fprintf(1, '\t\tStart hyperparameter tuning %d-%s\n', epoch_cnt, datestr(now));
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
        
        % initialize kernel matrix for new subsamples & hyperparameter
        [GP_model_left.K_approx, GP_model_left.L_approx] = GP_KL_generator(GP_model_left, ...
                                                                           train_data_x(GP_model_left.data_index,:));
        GP_model_left.L_appAndNoise = jitChol( (GP_model_left.L_approx * GP_model_left.L_approx' + ...
                                                GP_model_left.hyp_para(end)* ...
                                                eye(length(GP_model_left.data_index))), 5, 'lower' );
        tmp_inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
        GP_model_left.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                              tmp_inv_L_appAndNoise;
        GP_model_left.inv_L_appAndNoise_sqr_y = GP_model_left.inv_L_appAndNoise_sqr * ...
                                                train_data_y(GP_model_left.data_index, 1);

        [GP_model_right.K_approx, GP_model_right.L_approx] = GP_KL_generator(GP_model_right, ...
                                                                             train_data_x(GP_model_right.data_index,:));
        GP_model_right.L_appAndNoise = jitChol( (GP_model_right.L_approx * GP_model_right.L_approx' + ...
                                                GP_model_right.hyp_para(end)* ...
                                                eye(length(GP_model_right.data_index))), 5, 'lower' );
        tmp_inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
        GP_model_right.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                               tmp_inv_L_appAndNoise;
        GP_model_right.inv_L_appAndNoise_sqr_y = GP_model_right.inv_L_appAndNoise_sqr * ...
                                                 train_data_y(GP_model_right.data_index, 1);
        
        fprintf(1, '\t\tEnd optimization, Start prediction on validation %s\n', datestr(now));
        output_on_valid = zeros(num_valid_data, 2);
        for valid_cnt = 1:num_valid_data
            [output_on_valid(valid_cnt, 1), ~] = GPR_predict(GP_model_left, train_data_x, ...
                                                             train_data_y(:, 1), valid_data_x(valid_cnt, :), 2);
            [output_on_valid(valid_cnt, 2), ~] = GPR_predict(GP_model_right, train_data_x, ...
                                                             train_data_y(:, 2), valid_data_x(valid_cnt, :), 2);
        end
        valid_error = sum(sum((valid_data_y(1:num_valid_data,:) - output_on_valid).^2))/(2*num_valid_data);
        fprintf(1, '\t\tEnd prediction on validation %s/error %f\n', datestr(now), valid_error);

        % Save Intermediate Result
        file_name = sprintf('%s/LearnedModel_GPR(size%d_epoch%d).mat', folder_name, size_model, epoch_cnt);
        model_data_index = [GP_model_left.data_index; GP_model_right.data_index];
        save(file_name, 'model_data_index', 'model_hyp_para', 'valid_error');
        fprintf(1, 'End training epoch %d(%s)\n\n', epoch_cnt, datestr(now));
    end
end

%% Prediction using several models

if train_boost
    fprintf(1, 'Start Kernel setup\n');
    load(sprintf('./GPR/result/LearnedModel_GPR_%d.mat', num_models));

    %initial set-up for fast-computation
    GP_boost_model_left = [];
    GP_boost_model_right = [];

    for a = 1:num_models
        % left wheel
        GP_boost_model_left{a}.num_total = num_train_data;
        GP_boost_model_left{a}.data_index = model_data_index(a,:);

            % hyperparameter setting
        GP_boost_model_left{a} = hyp_setting(GP_boost_model_left{a},left_hyper_parameter(a, :));
        GP_boost_model_left{a}.kernel = @(x1,x2,h) kernFunc(x1,x2,h);

            % gp type setting
        GP_boost_model_left{a}.covfunc = @covSEard;
        GP_boost_model_left{a}.meanfunc = @meanZero;
        GP_boost_model_left{a}.likfunc = @likGauss;

            % marginal likelihood function
        GP_boost_model_left{a}.log_marg_lik = @(h) gp_feval_transform(h, GP_boost_model_left{a}.meanfunc, ...
                                                          GP_boost_model_left{a}.covfunc, GP_boost_model_left{a}.likfunc, ...
                                                          train_data_x(GP_boost_model_left{a}.data_index,:), ...
                                                          train_data_y(GP_boost_model_left{a}.data_index,1));

            % kernel matrix
        [GP_boost_model_left{a}.K_approx, GP_boost_model_left{a}.L_approx] = GP_KL_generator(GP_boost_model_left{a}, ...
                                                                                             train_data_x(GP_boost_model_left{a}.data_index,:));
        GP_boost_model_left{a}.L_appAndNoise = jitChol( (GP_boost_model_left{a}.L_approx * GP_boost_model_left{a}.L_approx' + ...
                                                         GP_boost_model_left{a}.hyp_para(end)* ...
                                                         eye(length(GP_boost_model_left{a}.data_index))), 5, 'lower' );
            %GP_boost_model_left{a}.inv_L_appAndNoise = inv(GP_boost_model_left{a}.L_appAndNoise);
        tmp_inv_L_appAndNoise = inv(GP_boost_model_left{a}.L_appAndNoise);
        GP_boost_model_left{a}.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                                       tmp_inv_L_appAndNoise;
        GP_boost_model_left{a}.inv_L_appAndNoise_sqr_y = GP_boost_model_left{a}.inv_L_appAndNoise_sqr * ...
                                                         train_data_y(GP_boost_model_left{a}.data_index, 1);


        % right wheel
        GP_boost_model_right{a}.num_total = num_train_data;
        GP_boost_model_right{a}.data_index = model_data_index(a,:);

            % hyperparameter setting
        GP_boost_model_right{a} = hyp_setting(GP_boost_model_right{a},right_hyper_parameter(a, :));
        GP_boost_model_right{a}.kernel = @(x1,x2,h) kernFunc(x1,x2,h);

            % gp type setting
        GP_boost_model_right{a}.covfunc = @covSEard;
        GP_boost_model_right{a}.meanfunc = @meanZero;
        GP_boost_model_right{a}.likfunc = @likGauss;

            % marginal likelihood function
        GP_boost_model_right{a}.log_marg_lik = @(h) gp_feval_transform(h, GP_boost_model_right{a}.meanfunc, ...
                                                          GP_boost_model_right{a}.covfunc, GP_boost_model_right{a}.likfunc, ...
                                                          train_data_x(GP_boost_model_right{a}.data_index,:), ...
                                                          train_data_y(GP_boost_model_right{a}.data_index,2));

            % kernel matrix
        [GP_boost_model_right{a}.K_approx, GP_boost_model_right{a}.L_approx] = GP_KL_generator(GP_boost_model_right{a}, ...
                                                                                               train_data_x(GP_boost_model_right{a}.data_index,:));
        GP_boost_model_right{a}.L_appAndNoise = jitChol( (GP_boost_model_right{a}.L_approx * GP_boost_model_right{a}.L_approx' + ...
                                                          GP_boost_model_right{a}.hyp_para(end)* ...
                                                          eye(length(GP_boost_model_right{a}.data_index))), 5, 'lower' );
            %GP_boost_model_right{a}.inv_L_appAndNoise = inv(GP_boost_model_right{a}.L_appAndNoise);
        tmp_inv_L_appAndNoise = inv(GP_boost_model_right{a}.L_appAndNoise);
        GP_boost_model_right{a}.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                                        tmp_inv_L_appAndNoise;
        GP_boost_model_right{a}.inv_L_appAndNoise_sqr_y = GP_boost_model_right{a}.inv_L_appAndNoise_sqr * ...
                                                          train_data_y(GP_boost_model_right{a}.data_index, 2);

        fprintf(1,'%d-th Model kernel computation Finish\n', a);
    end

    clear tmp_inv_L_appAndNoise;
    %save(sprintf('./GPR/result/LearnedModel_GPR_%d.mat', num_models), 'GP_boost_model_left', 'GP_boost_model_right', '-v7.3');
    %fprintf(1, 'Successfully Save File\tCommence Prediction on Test Data\n');
    fprintf(1, 'Successfully Set-up GP Models\tCommence Prediction on Test Data\n');


    % make simulation on test data
    num_input_dim = (num_dim-2)/2;
    tmp_output = zeros(num_models, 1);
    tmp_std_of_output = zeros(num_models, 1);
    tmp_weight_of_output = zeros(num_models, 1);
    num_pred = int32(size(test_data_y, 2) / 2);
    pred_on_test_data = zeros(size(test_data_y));

    tic;
    for data_cnt = 1:num_test_data
        for pred_cnt = 1:num_pred
            if pred_cnt <= num_input_dim
                tmp_input = [test_data_x(data_cnt, 2*(pred_cnt-1)+1:2*num_input_dim), ... % input_history
                             pred_on_test_data(data_cnt, 1:2*(pred_cnt-1)), ...           % input_history
                             test_data_x(data_cnt, 2*(num_input_dim+pred_cnt-1)+1:2*(num_input_dim+pred_cnt))];  % command_signal
            else
                tmp_input = [pred_on_test_data(data_cnt, 2*(pred_cnt-num_input_dim)-1:2*(pred_cnt-1)), ... % input_history
                             test_data_x(data_cnt, 2*(num_input_dim+pred_cnt-1)+1:2*(num_input_dim+pred_cnt))];  % command_signal
            end

            % left_wheel_case
            for model_cnt = 1:num_models
                [tmp_output(model_cnt), tmp_std_of_output(model_cnt)] = GPR_predict(GP_boost_model_left{a}, train_data_x, ...
                                                                                    train_data_y(:, 1), tmp_input, 2);
                tmp_weight_of_output(model_cnt) = 1/tmp_std_of_output(model_cnt);
            end
            pred_on_test_data(data_cnt, 2*pred_cnt-1) = tmp_output' * tmp_weight_of_output/sum(tmp_weight_of_output);

            % right_wheel_case
            for model_cnt = 1:num_models
                [tmp_output(model_cnt), tmp_std_of_output(model_cnt)] = GPR_predict(GP_boost_model_right{a}, train_data_x, ...
                                                                                    train_data_y(:, 2), tmp_input, 2);
                tmp_weight_of_output(model_cnt) = 1/tmp_std_of_output(model_cnt);
            end
            pred_on_test_data(data_cnt, 2*pred_cnt) = tmp_output' * tmp_weight_of_output/sum(tmp_weight_of_output);
        end

        if mod(data_cnt, 100) == 0
            fprintf('\t%d\n', data_cnt);
        end
    end
    pred_time = toc;

    diff = pred_on_test_data - test_data_y;
    error = zeros(3,1);
    error(1) = sum(sum(abs(diff)))/num_test_data;
    error(2) = sqrt(sum(sum(diff.^2))/num_test_data);
    error(3) = max(max(abs(diff), [], 2));

    fprintf(1, 'L1 Error(MAE) : %f, L2 Error(RMS) : %f, Linf Error : %f\n', error(1), error(2), error(3));
    fprintf(1, 'Computational Time Consumption : %f(avg. %f)\n', pred_time, pred_time/num_test_data);

    save(sprintf('./GPR/result/Result_of_GPR_boost_%d.mat', num_models), 'pred_on_test_data', 'test_data_y', 'pred_time', 'error', '-v7.3');
else
    num_pred = int32(size(test_data_y, 2) / 2);
    %num_pred = 4;
    pred_on_test_data = zeros(size(test_data_y));
    
    epoch_start = input('epoch of first model to test  ');
    epoch_end = input('epoch of last model to test  ');
    epoch_increment = input('increase of epoch  ');
    for epoch_cnt = epoch_start:epoch_increment:min([epoch_end, num_epoch])
        fprintf(1, 'Start Prediction on test data(%d / %s)\n', epoch_cnt, datestr(now));
        file_name = sprintf('%s/LearnedModel_GPR(size%d_epoch%d).mat', folder_name, size_model, epoch_cnt);
        load(file_name);
        
        if exist(sprintf('%s/Result_of_GPR(size%d_epoch%d).mat', folder_name, size_model, epoch_cnt), 'file')
            load(sprintf('%s/Result_of_GPR(size%d_epoch%d).mat', folder_name, size_model, epoch_cnt));
            start_data_index = data_cnt+1;
            prev_time = pred_time;
            fprintf(1, 'Load Intermediate Result File. Start at %d\n', start_data_index);
        else
            prev_time = 0;
            start_data_index = 1;
        end
    
        %initial set-up for fast-computation
        GP_model_left = [];
        GP_model_right = [];

        % left wheel
        GP_model_left.num_total = num_train_data;
        GP_model_left.data_index = model_data_index(1,:);

            % hyperparameter setting
        GP_model_left = hyp_setting(GP_model_left, model_hyp_para(1, :));
        GP_model_left.kernel = @(x1,x2,h) kernFunc(x1,x2,h);

            % gp type setting
        GP_model_left.covfunc = @covSEard;
        GP_model_left.meanfunc = @meanZero;
        GP_model_left.likfunc = @likGauss;

            % marginal likelihood function
        GP_model_left.log_marg_lik = @(h) gp_feval_transform(h, GP_model_left.meanfunc, ...
                                                             GP_model_left.covfunc, GP_model_left.likfunc, ...
                                                             train_data_x(GP_model_left.data_index,:), ...
                                                             train_data_y(GP_model_left.data_index,1));

            % kernel matrix
        [GP_model_left.K_approx, GP_model_left.L_approx] = GP_KL_generator(GP_model_left, ...
                                                                           train_data_x(GP_model_left.data_index,:));
        GP_model_left.L_appAndNoise = jitChol( (GP_model_left.L_approx * GP_model_left.L_approx' + ...
                                                GP_model_left.hyp_para(end)* ...
                                                eye(length(GP_model_left.data_index))), 5, 'lower' );
        %GP_model_left.inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
        tmp_inv_L_appAndNoise = inv(GP_model_left.L_appAndNoise);
        GP_model_left.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                              tmp_inv_L_appAndNoise;
        GP_model_left.inv_L_appAndNoise_sqr_y = GP_model_left.inv_L_appAndNoise_sqr * ...
                                                train_data_y(GP_model_left.data_index, 1);


        % right wheel
        GP_model_right.num_total = num_train_data;
        GP_model_right.data_index = model_data_index(2,:);

            % hyperparameter setting
        GP_model_right = hyp_setting(GP_model_right, model_hyp_para(2, :));
        GP_model_right.kernel = @(x1,x2,h) kernFunc(x1,x2,h);

            % gp type setting
        GP_model_right.covfunc = @covSEard;
        GP_model_right.meanfunc = @meanZero;
        GP_model_right.likfunc = @likGauss;

            % marginal likelihood function
        GP_model_right.log_marg_lik = @(h) gp_feval_transform(h, GP_model_right.meanfunc, ...
                                                          GP_model_right.covfunc, GP_model_right.likfunc, ...
                                                          train_data_x(GP_model_right.data_index,:), ...
                                                          train_data_y(GP_model_right.data_index,2));

            % kernel matrix
        [GP_model_right.K_approx, GP_model_right.L_approx] = GP_KL_generator(GP_model_right, ...
                                                                             train_data_x(GP_model_right.data_index,:));
        GP_model_right.L_appAndNoise = jitChol( (GP_model_right.L_approx * GP_model_right.L_approx' + ...
                                                 GP_model_right.hyp_para(end)* ...
                                                 eye(length(GP_model_right.data_index))), 5, 'lower' );
        %GP_model_right.inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
        tmp_inv_L_appAndNoise = inv(GP_model_right.L_appAndNoise);
        GP_model_right.inv_L_appAndNoise_sqr = tmp_inv_L_appAndNoise' * ...
                                               tmp_inv_L_appAndNoise;
        GP_model_right.inv_L_appAndNoise_sqr_y = GP_model_right.inv_L_appAndNoise_sqr * ...
                                                 train_data_y(GP_model_right.data_index, 2);

        clear tmp_inv_L_appAndNoise;
        fprintf(1, 'Successfully Set-up GP Model(%d)\nCommence Prediction on Test Data(%s)\n\n', epoch_cnt, datestr(now));
    
        % completed model set-up above
        % start prediction on test data
        tic;
        for data_cnt = start_data_index:num_test_data
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

%             if mod(data_cnt, 500) == 0
            if mod(data_cnt, 2000) == 0
            %if mod(data_cnt, floor(num_test_data/10)) == 0
                fprintf('\t%.3f - %s\n', data_cnt*100/num_test_data, datestr(now));
                pred_time = toc + prev_time;
                save(sprintf('%s/Result_of_GPR(size%d_epoch%d).mat', folder_name, size_model, epoch_cnt), 'pred_on_test_data', 'pred_time', 'data_cnt');
            end
        end

        pred_time = toc + prev_time;

        diff = pred_on_test_data(:, 1:2*num_pred) - test_data_y(:, 1:2*num_pred);
        error = zeros(3,1);
        error(1) = sum(sum(abs(diff)))/num_test_data;
        error(2) = sqrt(sum(sum(diff.^2))/num_test_data);
        error(3) = max(max(abs(diff), [], 2));

        fprintf(1, '\nGPR model(%d)-Test result\n', epoch_cnt);
        fprintf(1, 'L1 Error(MAE) : %f, L2 Error(RMS) : %f, Linf Error : %f\n', error(1), error(2), error(3));
        fprintf(1, 'Computational Time Consumption : %f(avg. %f)\n', pred_time, pred_time/num_test_data);

        save(sprintf('%s/Result_of_GPR(size%d_epoch%d).mat', folder_name, size_model, epoch_cnt), 'pred_on_test_data', 'test_data_y', 'pred_time', 'error');
        fprintf(1, '\tSave result of GPR model(%d) / %s\n\n', epoch_cnt, datestr(now));
    end
end
