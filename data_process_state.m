%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.08.08 Updated
% Program to read several log files in \DataFile\
%         and make dataset appropriate to each model
%         (This code makes dataset as state(speed, acceleration, etc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;  clc;
addpath('.\DataFile');

pred_window = 1;
ratio_of_valid = 0.95;

% list of data log for training data set
train_data_list = [];
train_data_list{1} = 'result_log_data_log_060616.mat';
train_data_list{2} = 'result_log_bbb3_060116.mat';
train_data_list{3} = 'result_log_daily_run_102215_1.mat';
train_data_list{4} = 'result_log_daily_run_dow_101615.mat';
train_data_list{5} = 'result_log_bbb3_060116_2.mat';
train_data_list{6} = 'result_log_backward_driving3_062916.mat';
train_data_list{7} = 'result_log_daily_run_101515.mat';
train_data_list{8} = 'result_log_daily_run_101415_1.mat';
train_data_list{9} = 'result_log_backward_driving_062916.mat';
train_data_list{10} = 'result_log_data_log_060616_2.mat';
train_data_list{11} = 'result_log_bbb3_060716_2.mat';
train_data_list{12} = 'result_log_rotational_driving_062916.mat';
train_data_list{13} = 'result_log_daily_run_101415_2.mat';
train_data_list{14} = 'result_log_daily_run_101315_1.mat';
train_data_list{15} = 'result_log_daily_run_101315_2.mat';
train_data_list{16} = 'result_log_bbb3_060716_1.mat';
train_data_list{17} = 'result_log_bbb3_062716.mat';
train_data_list{18} = 'result_log_bbb3_062716_fast.mat';
train_data_list{19} = 'result_log_backward_driving2_062916.mat';
train_data_list{20} = 'result_log_daily_run_102215_2.mat';


train_data_x = []; train_data_y = [];
valid_data_x = []; valid_data_y = [];
for list_cnt = 1:length(train_data_list)
    fprintf(1, '%d-th training data start\n', list_cnt);
    load(train_data_list{list_cnt});
    log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1)-pred_window-4;
    left_speed = Log.encoderLeftWheelSpeedSmoothed;
    right_speed = Log.encoderRightWheelSpeedSmoothed;
    forw_cmd = Log.forwardCommand;
    left_cmd = Log.leftCommand;
    
    split_cnt = int64(log_file_size * ratio_of_valid);
    data_index = randperm(log_file_size);
    
    train_data_x_tmp = zeros(split_cnt-1, 2*pred_window+4);
    train_data_y_tmp = zeros(split_cnt-1, 4*pred_window);
    valid_data_x_tmp = zeros(log_file_size+1-split_cnt, 2*pred_window+4);
    valid_data_y_tmp = zeros(log_file_size+1-split_cnt, 4*pred_window);
    
    
    for data_cnt = 1:log_file_size
        if data_cnt < 2
            tmptmp_x = []; tmptmp_y = [];
            for cnt2 = 0:4
                tmptmp_x = [tmptmp_x, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
                tmptmp_y = [tmptmp_y, left_speed(data_cnt+cnt2+1), right_speed(data_cnt+cnt2+1)];
            end
            tmptmp_x = [tmptmp_x, forw_cmd(data_cnt+2), left_cmd(data_cnt+2)];
        else
            tmptmp_x = [tmptmp_x(3:end-2), left_speed(data_cnt+4), right_speed(data_cnt+4), ...
                     forw_cmd(data_cnt+2), left_cmd(data_cnt+2)];
            tmptmp_y = [tmptmp_y(3:end), left_speed(data_cnt+5), right_speed(data_cnt+5)];
        end
        left_accel = (tmptmp_x(1)-8*tmptmp_x(3)+8*tmptmp_x(7)-tmptmp_x(9))*25/12;
        right_accel = (tmptmp_x(2)-8*tmptmp_x(4)+8*tmptmp_x(8)-tmptmp_x(10))*25/12;
        tmp_x = [tmptmp_x(5), left_accel, tmptmp_x(6), right_accel, tmptmp_x(end-1:end)];
        
        left_accel = (tmptmp_y(1)-8*tmptmp_y(3)+8*tmptmp_y(7)-tmptmp_y(9))*25/12;
        right_accel = (tmptmp_y(2)-8*tmptmp_y(4)+8*tmptmp_y(8)-tmptmp_y(10))*25/12;
        tmp_y = [tmptmp_y(5), left_accel, tmptmp_y(6), right_accel];
        
        perm_ind = data_index(data_cnt);
        if perm_ind < split_cnt
            train_data_x_tmp(perm_ind, :) = tmp_x;
            train_data_y_tmp(perm_ind, :) = tmp_y;
        else
            valid_data_x_tmp(perm_ind+1-split_cnt, :) = tmp_x;
            valid_data_y_tmp(perm_ind+1-split_cnt, :) = tmp_y;
        end
    end
    train_data_x = [train_data_x; train_data_x_tmp];
    train_data_y = [train_data_y; train_data_y_tmp];
    valid_data_x = [valid_data_x; valid_data_x_tmp];
    valid_data_y = [valid_data_y; valid_data_y_tmp];
end

clear tmp_x tmp_y tmptmp_x tmptmp_y train_data_x_tmp train_data_y_tmp valid_data_x_tmp valid_data_y_tmp;
save_file_name = 'train_data_set(state).mat';
save(save_file_name, 'train_data_x', 'train_data_y', 'valid_data_x', 'valid_data_y');


%%
% list of data log for test data set
clear;  clc;
addpath('.\DataFile');

test_data_list = [];
test_data_list{1} = 'result_log_data_log_061716.mat';
test_data_list{2} = 'result_log_daily_run_101615.mat';
test_data_list{3} = 'result_log_data_log_061716_2.mat';

% hist_window = 20;
pred_window = 5*25; % (25Hz 5seconds)
test_data_x = []; test_data_y = [];

for list_cnt = 1:length(test_data_list)
    fprintf(1, '%d-th test data start\n', list_cnt);
    load(test_data_list{list_cnt});
    log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1);
    
    left_speed = zeros(log_file_size+2, 1);
    right_speed = zeros(log_file_size+2, 1);
    forw_cmd = zeros(log_file_size+2, 1);
    left_cmd = zeros(log_file_size+2, 1);
    left_speed(3:end) = Log.encoderLeftWheelSpeedSmoothed;
    right_speed(3:end) = Log.encoderRightWheelSpeedSmoothed;
    forw_cmd(3:end) = Log.forwardCommand;
    left_cmd(3:end) = Log.leftCommand;
    
    
    log_file_size = log_file_size - pred_window -2;

    test_data_x_tmp = zeros(log_file_size, 2*(2+pred_window));
    test_data_y_tmp = zeros(log_file_size, 4*pred_window);    
    for data_cnt = 1:log_file_size
        if data_cnt < 2
            tmptmp_x = []; tmp_y = [];
            for cnt2 = 0:4
                tmptmp_x = [tmptmp_x, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
            
            for cnt2 = 1:pred_window
                tmptmp_x = [tmptmp_x, forw_cmd(data_cnt+cnt2+1), left_cmd(data_cnt+cnt2+1)];
                
                left_accel = (left_speed(data_cnt+cnt2)-8*left_speed(data_cnt+cnt2+1)+ ...
                              8*left_speed(data_cnt+cnt2+3)-left_speed(data_cnt+cnt2+4))*25/12;
                right_accel = (right_speed(data_cnt+cnt2)-8*right_speed(data_cnt+cnt2+1)+ ...
                              8*right_speed(data_cnt+cnt2+3)-right_speed(data_cnt+cnt2+4))*25/12;
                tmp_y = [tmp_y, left_speed(data_cnt+cnt2+2), left_accel, right_speed(data_cnt+cnt2+2), right_accel];
            end
        else
            tmptmp_x = [tmptmp_x(3:10), left_speed(data_cnt+4), right_speed(data_cnt+4), ...
                        tmptmp_x(11:end-2), forw_cmd(data_cnt+pred_window+2), left_cmd(data_cnt+pred_window+2)];
                 
            left_accel = (left_speed(data_cnt+pred_window)-8*left_speed(data_cnt+pred_window+1)+ ...
                          8*left_speed(data_cnt+pred_window+3)-left_speed(data_cnt+pred_window+4))*25/12;
            right_accel = (right_speed(data_cnt+pred_window)-8*right_speed(data_cnt+pred_window+1)+ ...
                           8*right_speed(data_cnt+pred_window+3)-right_speed(data_cnt+pred_window+4))*25/12;
            tmp_y = [tmp_y(5:end), left_speed(data_cnt+pred_window+2), left_accel, ...
                     right_speed(data_cnt+pred_window+2), right_accel];
        end
        left_accel = (tmptmp_x(1)-8*tmptmp_x(3)+8*tmptmp_x(7)-tmptmp_x(9))*25/12;
        right_accel = (tmptmp_x(2)-8*tmptmp_x(4)+8*tmptmp_x(8)-tmptmp_x(10))*25/12;
        tmp_x = [tmptmp_x(5), left_accel, tmptmp_x(6), right_accel, tmptmp_x(11:end)];
        
        test_data_x_tmp(data_cnt, :) = tmp_x;
        test_data_y_tmp(data_cnt, :) = tmp_y;
    end
    test_data_x = [test_data_x; test_data_x_tmp];
    test_data_y = [test_data_y; test_data_y_tmp];
end

save('test_data_set(state).mat', 'test_data_x', 'test_data_y');

%% Make data file for reference input

clear;	clc;
reference_input = [];
phys_data = false;
if phys_data
	num_state = 4;
else
	num_state = 40;
end

for forw_cnt=0:5:100
	for left_cnt = -100:5:100
		tmp_x = zeros(1, 250+num_state);

		for cnt = (num_state+51):2:(num_state+250)
			tmp_x(cnt) = forw_cnt;
			tmp_x(cnt+1) = left_cnt;
		end
		reference_input = [reference_input;tmp_x];
	end
end


for forw_cnt=0:5:100
	for left_cnt = -100:5:100
		tmp_x = zeros(1, 250+num_state);
                forw_slope = forw_cnt/100;
		left_slope = left_cnt/100;

		for cnt = (num_state+51):2:(num_state+250)
			tmp_x(cnt) = forw_slope * (cnt-num_state-49)/2;
			tmp_x(cnt+1) = left_slope * (cnt-num_state-49)/2;
		end
		reference_input = [reference_input;tmp_x];
	end
end

if phys_data
	save('test_data_reference_input_phys.mat', 'reference_input');
else
	save('test_data_reference_input.mat', 'reference_input');
end