%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.07.08 Updated
% Program to optimize the (Jongjin Park's) dynamics model of wheelchair
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Manually tune physics model

% make Data Log file
close all;  clear; clc;
% addpath('./Physics-based');

plot_lts_on_whole_data = false;

dataType = 'umich16';
set_params();

% model_param = [RobotModelParams.mu;RobotModelParams.beta;
%                RobotModelParams.gamma;RobotModelParams.alpha;
%                RobotModelParams.mu;RobotModelParams.beta;
%                RobotModelParams.gamma;RobotModelParams.alpha;
%                RobotModelParams.turnRateInPlace;RobotModelParams.turnReductionRate];

% model_param = [0.2128;6.1026;
%                3.6299;0.1474;
%                0.2128;6.1026;
%                3.6299;0.1474;
%                0.3932;0.0037];

model_param = [0.2106;5.9118;
               3.5973;0.1451;
               0.2128;6.1026;
               3.6299;0.1474;
               0.4175;0.0051];

if plot_lts_on_whole_data
    make_new_data = false;

    if make_new_data
        load('./DataFile/result_log_data_log_061716.mat');
        Log_test = Log;

        load('./DataFile/result_log_daily_run_101615.mat');
        Log_test = log_connector(Log_test, Log);
        clear Log;

        load('./DataFile/result_log_data_log_061716_2.mat');
        Log_test = log_connector(Log_test, Log);
        clear Log;

        save('test_data_for_phys_model_tuning.mat','Log_test');
    else
        load('test_data_for_phys_model_tuning.mat');
    end

    dataBegin = 1;  dataEnd = length(Log_test.plotTime);

    [~, vs, omegas, us_joystick, qs, us_motor, ~] = model_error_func2(model_param, [], ...
                                                                       Quantum6000Params.wheelBase, ...
                                                                       Log_test, dataBegin, dataEnd, true);

    plot_for_robot_model(Log_test, qs, vs, omegas, us_motor, us_joystick, [dataBegin; dataEnd], ...
                         [480, 680], true, false, false);
end

%% evaluate 5 second Long-Term Simulation
% make Data Log file
close all;  clear; clc;
% addpath('./Physics-based');

% folder_name = './Physics-based/result';
folder_name = './new_result2';
if ~exist(folder_name)
    mkdir(folder_name);
end
save_figure = false;

use_reference_input = false;

dataType = 'umich16';
set_params();

% model_param = [RobotModelParams.mu;RobotModelParams.beta;
%                RobotModelParams.gamma;RobotModelParams.alpha;
%                RobotModelParams.mu;RobotModelParams.beta;
%                RobotModelParams.gamma;RobotModelParams.alpha;
%                RobotModelParams.turnRateInPlace;RobotModelParams.turnReductionRate];

% model_param = [0.2128;6.1026;
%                3.6299;0.1474;
%                0.2128;6.1026;
%                3.6299;0.1474;
%                0.3932;0.0037];

model_param = [0.2106;5.9118;
               3.5973;0.1451;
               0.2128;6.1026;
               3.6299;0.1474;
               0.4175;0.0051];

if use_reference_input
    fprintf(1, 'Response to Reference Input\n');
    load('test_data_reference_input_phys.mat');
    num_test_data = size(reference_input, 1);
    
    tic;
    %%% MATLAB version of simulation function
    % test_data_y = zeros(num_test_data, 250);
    % [err, vs, omegas, us_joystick, qs, us_motor, fs] = model_error_func(model_param, [], Quantum6000Params.wheelBase, ...
    %                                                                     reference_input, test_data_y);

    %%% C version of simulation function 
    [vs, omegas, us_joystick, qs, us_motor, fs] = model_predict([model_param;Quantum6000Params.wheelBase], ...
                                                               reference_input, 125);

    time_for_simulation = toc;

    model_prediction = qs;
    save_file_name = sprintf('%s/ref_input_result(Physics_model).mat', folder_name);
    save(save_file_name, 'time_for_simulation', 'model_prediction');
    
    if save_figure
        set(0,'DefaultFigureVisible','off');
        
        folder_name = sprintf('%s/ref_input_plot', folder_name);
        if ~exist(folder_name)
            mkdir(folder_name);
        end
        
        plot_x = 0.04:0.04:5;
        for cnt = 1:num_test_data
            save_file_name = sprintf('%s/%d.png', folder_name, cnt);
            forw_cmd = reference_input(cnt, 5:2:end)/100;
            left_cmd = reference_input(cnt, 6:2:end)/100;
            
            left_wheel = model_prediction(cnt, 1:2:end);
            right_wheel = model_prediction(cnt, 2:2:end);
            
            fig = figure('Position', [5, 5, 1550, 850]);
            subplot(2,1,1);
            y_max = max([0.25, max(forw_cmd), max(left_cmd), max(left_wheel)]);
            y_min = min([-0.25, min(forw_cmd), min(left_cmd), min(left_wheel)]);
            hold on;
            plot(plot_x, left_wheel, 'r-');
            plot(plot_x, forw_cmd, 'k--');
            plot(plot_x, left_cmd, 'k-.');
            hold off;
            xlabel('time(sec)');    ylabel('speed of left wheel(m/s)');
            legend('model', 'forw cmd', 'left cmd', 'Location', 'EastOutside');
            ylim([y_min y_max]);
            
            subplot(2,1,2);
            y_max = max([0.25, max(forw_cmd), max(left_cmd), max(right_wheel)]);
            y_min = min([-0.25, min(forw_cmd), min(left_cmd), min(right_wheel)]);
            
            hold on;
            plot(plot_x, right_wheel, 'r-');
            plot(plot_x, forw_cmd, 'k--');
            plot(plot_x, left_cmd, 'k-.');
            hold off;
            xlabel('time(sec)');    ylabel('speed of right wheel(m/s)');
            legend('model', 'forw cmd', 'left cmd', 'Location', 'EastOutside');
            ylim([y_min y_max]);
            
            print(fig, save_file_name, '-dpng');
        end
    end
else
    fprintf(1, 'Response to Test Data\n');
    load('test_data_set_phys.mat');
    num_test_data = size(test_data_x, 1);

    tic;
    %%% MATLAB version of simulation function
    % [err, vs, omegas, us_joystick, qs, us_motor, fs] = model_error_func(model_param, [], Quantum6000Params.wheelBase, ...
    %                                                                     test_data_x, test_data_y);

    %%% C version of simulation function 
    [vs, omegas, us_joystick, qs, us_motor, fs] = model_predict([model_param;Quantum6000Params.wheelBase], ...
                                                               test_data_x, 125);

    time_for_simulation = toc;

    model_prediction = qs;
    observed_speed = test_data_y;
    difference_btw_prediction_data = qs - test_data_y;

    error = [sum(sum(abs(difference_btw_prediction_data)))/num_test_data, ...
             sqrt(sum(sum(difference_btw_prediction_data.^2))/num_test_data), ...
             max(max(abs(difference_btw_prediction_data), [], 2))];

    fprintf(1, 'test MAE error : %f\n', error(1));
    fprintf(1, 'test RMS error : %f\n', error(2));
    fprintf(1, 'test Maximum error : %f\n', error(3));

    %joystick_command = test_data_x(:, 41:290);
    joystick_command = test_data_x(:, 5:254);

    save(sprintf('%s/5sec_LTS_result(Physics_model).mat', folder_name), 'model_prediction', 'observed_speed', ...
          'difference_btw_prediction_data', 'joystick_command', 'error', 'time_for_simulation', '-v7.3');


    error_per_case = sum(difference_btw_prediction_data.^2, 2);
    [~, ind] = sort(error_per_case, 1, 'ascend');

    [~, max_ind] = max(max(abs(difference_btw_prediction_data),[],2));

    num_plot = 5;

    for cnt = 1:num_plot+1
        switch cnt
            case 1
                data_ind = ind(cnt);
                title_str1 = sprintf('Simulation vs Observation(left wheel, Simple Dynamics, 100%%-best)');
                title_str2 = sprintf('Simulation vs Observation(right wheel, Simple Dynamics, 100%%-best)');
                fig_file_name = sprintf('%s/model_5sec_simulation(physics, best)', folder_name);
            case 2
                data_ind = ind(int64(size(error_per_case,1)/4));
                title_str1 = sprintf('Simulation vs Observation(left wheel, Simple Dynamics, 75%%-good)');
                title_str2 = sprintf('Simulation vs Observation(right wheel, Simple Dynamics, 75%%-good)');
                fig_file_name = sprintf('%s/model_5sec_simulation(physics, good)', folder_name);
            case 3
                data_ind = ind(int64(size(error_per_case,1)/2));
                title_str1 = sprintf('Simulation vs Observation(left wheel, Simple Dynamics, 50%%-fine)');
                title_str2 = sprintf('Simulation vs Observation(right wheel, Simple Dynamics, 50%%-fine)');
                fig_file_name = sprintf('%s/model_5sec_simulation(physics, fine)', folder_name);
            case 4
                data_ind = ind(int64(3*size(error_per_case,1)/4));
                title_str1 = sprintf('Simulation vs Observation(left wheel, Simple Dynamics, 25%%-not good)');
                title_str2 = sprintf('Simulation vs Observation(right wheel, Simple Dynamics, 25%%-not good)');
                fig_file_name = sprintf('%s/model_5sec_simulation(physics, not_good)', folder_name);
            case 5
                data_ind = ind(end);
                title_str1 = sprintf('Simulation vs Observation(left wheel, Simple Dynamics, 0%%-worst)');
                title_str2 = sprintf('Simulation vs Observation(right wheel, Simple Dynamics, 0%%-worst)');
                fig_file_name = sprintf('%s/model_5sec_simulation(physics, worst)', folder_name);
            otherwise
                data_ind = max_ind;
                title_str1 = sprintf('Simulation vs Observation(left wheel, Simple Dynamics, max)');
                title_str2 = sprintf('Simulation vs Observation(right wheel, Simple Dynamics, max)');
                fig_file_name = sprintf('%s/model_5sec_simulation(physics, max_diff)', folder_name);
        end

        forw_cmd = joystick_command(data_ind, 1:2:250)/100;
        left_cmd = joystick_command(data_ind, 2:2:250)/100;

        left_speed = qs(data_ind, 1:2:250);
        obs_left_speed = test_data_y(data_ind, 1:2:250);
        right_speed = qs(data_ind, 2:2:250);
        obs_right_speed = test_data_y(data_ind, 2:2:250);

        fig = figure('Position', [10, 10, 1550, 850]);
        subplot(2,1,1);
        hold on;
        plot(0:0.04:4.96, left_speed, 'r-');
        plot(0:0.04:4.96, obs_left_speed, 'b-');
        stairs(0:0.04:4.96, forw_cmd, 'k--');
        stairs(0:0.04:4.96, left_cmd, 'k-.');
        hold off;
        grid on;    grid minor;
        title(title_str1);
        xlabel('time(sec)'); ylabel('wheel speed(m/s)');


        subplot(2,1,2);
        hold on;
        plot(0:0.04:4.96, right_speed, 'r-');
        plot(0:0.04:4.96, obs_right_speed, 'b-');
        stairs(0:0.04:4.96, forw_cmd, 'k--');
        stairs(0:0.04:4.96, left_cmd, 'k-.');
        hold off;
        grid on;    grid minor;
        title(title_str2);
        xlabel('time(sec)'); ylabel('wheel speed(m/s)');

        if save_figure
            print(fig, fig_file_name, '-dpng');
        end
    end
end


%% Use Optimization tool for tuning of physics model

clear;  clc;

%%% select a log
% Data Loading
hist_window = 1;   pred_window = 25;
load(sprintf('train_data_set(hw%d_pw%d).mat', hist_window, pred_window));
load('test_data_set.mat');

num_train_data = size(train_data_x, 1);
num_valid_data = size(valid_data_x, 1);
num_test_data = size(test_data_x, 1);
dim_input = size(train_data_x, 2);

%%% import parameters
dataType = 'umich16';
set_params();


%%% optimize part
x0 = [RobotModelParams.mu;RobotModelParams.beta;
      RobotModelParams.gamma;RobotModelParams.alpha;
      RobotModelParams.mu;RobotModelParams.beta;
      RobotModelParams.gamma;RobotModelParams.alpha;
      RobotModelParams.turnRateInPlace;RobotModelParams.turnReductionRate];

[err, vs, omegas, us_joystick, qs, us_motor, fs] = model_error_func(x0, [], Quantum6000Params.wheelBase, ...
                                                                    test_data_x, test_data_y);

fprintf(1, 'test MAE error : %f\n', err(1)/(num_test_data*250));
fprintf(1, 'test RMS error : %f\n', sqrt(err(2)/(num_test_data*250)));
fprintf(1, 'test Maximum error : %f\n', err(3));

  
  
lb = [0  0  0  0  0  0  0  0  0  0];
ub = [5 100 100  5  5 100 100  5  1  1];

num_opt_process = 6;
num_trials_per_process = 100;

parameter_history = zeros(size(x0,1),num_opt_process+1);
parameter_history(:,1) = x0;
valid_error_history = zeros(num_opt_process,1);

options = optimset('Algorithm', 'interior-point', ...
                       'MaxFunEvals', num_trials_per_process, ...
                       'TolFun', 1e-3, ...
                       'TolX',   1e-3);

% show results
disp('Initial Values');
disp(x0);


for cnt = 1:num_opt_process
    tic;
    [x, ~, ~, ~] = fmincon(@(x)model_error_func(x, [], Quantum6000Params.wheelBase, ...
                                                train_data_x, train_data_y), ...
                           parameter_history(:,cnt), [], [], [], [], lb, ub, [], options);
                       
    parameter_history(:, cnt+1) = x;
    
    disp('Intermediate Values');
    disp(parameter_history(:, cnt+1));

    [err, ~, ~, ~, ~, ~, ~] = model_error_func(x, [], Quantum6000Params.wheelBase, ...
                                                    valid_data_x, valid_data_y);
    t = toc;

    valid_error_history(cnt) = sqrt(err/num_valid_data);
    fprintf(1, '%d-th validation error : %f, time consumption : %f\n', cnt, sqrt(err/num_valid_data), t);
end

% batch_size = 10000;
% for cnt = 1:num_opt_process
%     num_batch = floor(num_train_data / batch_size);
%     for batch_cnt = 1:num_batch
%         if batch_cnt < 2
%             para_init = parameter_history(:,cnt);
%         else
%             para_init = x;
%         end
%         [x, ~, ~, ~] = fmincon(@(x)model_error_func(x, [], Quantum6000Params.wheelBase, ...
%                                                 train_data_x(batch_size*(batch_cnt-1)+1:batch_size*batch_cnt,:), ...
%                                                 train_data_y(batch_size*(batch_cnt-1)+1:batch_size*batch_cnt,:)), ...
%                                para_init, [], [], [], [], lb, ub, [], options);
%     end
% 
%     disp(' ');
%     disp('Optimization Result');
%     disp(x');
%     
%     [err, ~, ~, ~, ~, ~, ~] = model_error_func(x, [], Quantum6000Params.wheelBase, ...
%                                                valid_data_x, valid_data_y);
% 	fprintf(1, '\tvalidation error : %f\n', err);
%     valid_error_history(cnt) = err;
%                                                                  
% %     [~, vs, omegas, us_joystick, qs, us_motor, ~] = model_error_func(x, [], Quantum6000Params.wheelBase, ...
% %                                                                      train_data_x, train_data_y);
% %     plot_for_fit_robot_model(Log, qs, vs, omegas, us_motor, us_joystick, [dataBegin;dataEnd], plot_range, use_smoothed_data);
%     
%     parameter_history(:,cnt+1) = x;
% end

[err, vs, omegas, us_joystick, qs, us_motor, fs] = model_error_func(x, [], Quantum6000Params.wheelBase, ...
                                                                         test_data_x, test_data_y);
                                                                
fprintf(1, 'test error : %f\n', sqrt(err/num_test_data));


%%% plot results
%plot_model_predictions(Log, vs, omegas, us_joystick, [0 200]);

