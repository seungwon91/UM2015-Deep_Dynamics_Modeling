%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.07.26 Updated
% Program to read several log files in \DataFile\
%         and make dataset appropriate to each model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;  clc;
addpath('.\DataFile');

hist_window = 10;
pred_window = 50;
ratio_of_valid = 0.9;

phys_data = true;
if phys_data
    hist_window = 3;
end

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
    log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1)-hist_window-pred_window+1;
    left_speed = Log.encoderLeftWheelSpeedSmoothed;
    right_speed = Log.encoderRightWheelSpeedSmoothed;
    forw_cmd = Log.forwardCommand;
    left_cmd = Log.leftCommand;
    
    split_cnt = int64(log_file_size * ratio_of_valid);
    data_index = randperm(log_file_size);
    if phys_data
        train_data_x_tmp = zeros(split_cnt-1, 2*(hist_window-1+pred_window));
        train_data_y_tmp = zeros(split_cnt-1, 2*pred_window);
        valid_data_x_tmp = zeros(log_file_size+1-split_cnt, 2*(hist_window-1+pred_window));
        valid_data_y_tmp = zeros(log_file_size+1-split_cnt, 2*pred_window);
    else
        train_data_x_tmp = zeros(split_cnt-1, 2*(hist_window+pred_window));
        train_data_y_tmp = zeros(split_cnt-1, 2*pred_window);
        valid_data_x_tmp = zeros(log_file_size+1-split_cnt, 2*(hist_window+pred_window));
        valid_data_y_tmp = zeros(log_file_size+1-split_cnt, 2*pred_window);
    end
    for data_cnt = 1:log_file_size
%         if data_cnt < 2
%             tmp_x = []; tmp_y = [];
%             for cnt2 = 0:hist_window-1
%                 tmp_x = [tmp_x, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
%             end
%             for cnt2 = hist_window:hist_window+pred_window-1
%                 tmp_x = [tmp_x, forw_cmd(data_cnt+cnt2-1), left_cmd(data_cnt+cnt2-1)];
%                 tmp_y = [tmp_y, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
%             end
%         else
%             tmptmp_x = tmp_x;
%             tmptmp_y = tmp_y;
%             tmp_x(1:2*hist_window-2) = tmptmp_x(3:2*hist_window);
%             tmp_x(2*hist_window-1:2*hist_window) = tmptmp_y(1:2);
%             tmp_x(2*hist_window+1:2*(hist_window+pred_window-1)) = tmptmp_x(2*hist_window+3:2*(hist_window+pred_window));
%             tmp_x(2*(hist_window+pred_window)-1:2*(hist_window+pred_window)) = [forw_cmd(data_cnt+hist_window+pred_window-2), left_cmd(data_cnt+hist_window+pred_window-2)];
%             tmp_y(1:2*pred_window-2) = tmptmp_y(3:2*pred_window);
%             tmp_y(2*pred_window-1:2*pred_window) = [left_speed(data_cnt+hist_window+pred_window-1), right_speed(data_cnt+hist_window+pred_window-1)];
%         end
%         
%         perm_ind = data_index(data_cnt);
%         if phys_data
%             tmptmp_x = tmp_x;
%             acc1 = (tmp_x(3)-tmp_x(1))/(Log.plotTime(data_cnt+1)-Log.plotTime(data_cnt));
%             acc2 = (tmp_x(4)-tmp_x(2))/(Log.plotTime(data_cnt+1)-Log.plotTime(data_cnt));
%             tmptmp_x(1) = tmptmp_x(3);  tmptmp_x(3) = tmptmp_x(4);
%             tmptmp_x(2) = acc1;         tmptmp_x(4) = acc2;
%             
%             if perm_ind < split_cnt
%                 train_data_x_tmp(perm_ind, :) = tmptmp_x;
%                 train_data_y_tmp(perm_ind, :) = tmp_y;
%             else
%                 valid_data_x_tmp(perm_ind+1-split_cnt, :) = tmptmp_x;
%                 valid_data_y_tmp(perm_ind+1-split_cnt, :) = tmp_y;
%             end
%         else
%             if perm_ind < split_cnt
%                 train_data_x_tmp(perm_ind, :) = tmp_x;
%                 train_data_y_tmp(perm_ind, :) = tmp_y;
%             else
%                 valid_data_x_tmp(perm_ind+1-split_cnt, :) = tmp_x;
%                 valid_data_y_tmp(perm_ind+1-split_cnt, :) = tmp_y;
%             end
%         end
        
        
        if (data_cnt < 2) && (~phys_data)
            tmp_x = []; tmp_y = [];
            for cnt2 = 0:hist_window-1
                tmp_x = [tmp_x, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
            for cnt2 = hist_window:hist_window+pred_window-1
                tmp_x = [tmp_x, forw_cmd(data_cnt+cnt2-1), left_cmd(data_cnt+cnt2-1)];
                tmp_y = [tmp_y, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
        elseif ~phys_data
            tmptmp_x = tmp_x;
            tmptmp_y = tmp_y;
            tmp_x(1:2*hist_window-2) = tmptmp_x(3:2*hist_window);
            tmp_x(2*hist_window-1:2*hist_window) = tmptmp_y(1:2);
            tmp_x(2*hist_window+1:2*(hist_window+pred_window-1)) = tmptmp_x(2*hist_window+3:2*(hist_window+pred_window));
            tmp_x(2*(hist_window+pred_window)-1:2*(hist_window+pred_window)) = [forw_cmd(data_cnt+hist_window+pred_window-2), left_cmd(data_cnt+hist_window+pred_window-2)];
            tmp_y(1:2*pred_window-2) = tmptmp_y(3:2*pred_window);
            tmp_y(2*pred_window-1:2*pred_window) = [left_speed(data_cnt+hist_window+pred_window-1), right_speed(data_cnt+hist_window+pred_window-1)];
        elseif (data_cnt < 2) && phys_data
            tmp_x = []; tmp_y = [];
            for cnt2 = 0:hist_window+1
                tmp_x = [tmp_x, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
            for cnt2 = hist_window:hist_window+pred_window-1
                tmp_x = [tmp_x, forw_cmd(data_cnt+cnt2-1), left_cmd(data_cnt+cnt2-1)];
                tmp_y = [tmp_y, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
        else
            tmptmp_x = tmp_x;
            tmptmp_y = tmp_y;
            tmp_x(1:2*hist_window+2) = tmptmp_x(3:2*hist_window+4);
            tmp_x(2*hist_window+3:2*hist_window+4) = tmptmp_y(1:2);
            tmp_x(2*hist_window+5:2*(hist_window+pred_window+1)) = tmptmp_x(2*hist_window+7:2*(hist_window+pred_window+2));
            tmp_x(2*(hist_window+pred_window+1)+1:2*(hist_window+pred_window+2)) = [forw_cmd(data_cnt+hist_window+pred_window-2), left_cmd(data_cnt+hist_window+pred_window-2)];
            tmp_y(1:2*pred_window-2) = tmptmp_y(3:2*pred_window);
            tmp_y(2*pred_window-1:2*pred_window) = [left_speed(data_cnt+hist_window+pred_window-1), right_speed(data_cnt+hist_window+pred_window-1)];
        end
        
        perm_ind = data_index(data_cnt);
        if phys_data
            tmptmp_x = tmp_x(7:end);
            dT = (Log.plotTime(data_cnt+4)-Log.plotTime(data_cnt))/4;
            % (-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h))/(12h)
            acc1 = (-tmp_x(9)+8*tmp_x(7)-8*tmp_x(3)+tmp_x(1))/(12*dT);
            acc2 = (-tmp_x(10)+8*tmp_x(8)-8*tmp_x(4)+tmp_x(2))/(12*dT);
            tmptmp_x(1) = tmp_x(9); tmptmp_x(2) = acc1;
            tmptmp_x(3) = tmp_x(10);    tmptmp_x(4) = acc2;
            
            if perm_ind < split_cnt
                train_data_x_tmp(perm_ind, :) = tmptmp_x;
                train_data_y_tmp(perm_ind, :) = tmp_y;
            else
                valid_data_x_tmp(perm_ind+1-split_cnt, :) = tmptmp_x;
                valid_data_y_tmp(perm_ind+1-split_cnt, :) = tmp_y;
            end
        else
            if perm_ind < split_cnt
                train_data_x_tmp(perm_ind, :) = tmp_x;
                train_data_y_tmp(perm_ind, :) = tmp_y;
            else
                valid_data_x_tmp(perm_ind+1-split_cnt, :) = tmp_x;
                valid_data_y_tmp(perm_ind+1-split_cnt, :) = tmp_y;
            end
        end
    end
    train_data_x = [train_data_x; train_data_x_tmp];
    train_data_y = [train_data_y; train_data_y_tmp];
    valid_data_x = [valid_data_x; valid_data_x_tmp];
    valid_data_y = [valid_data_y; valid_data_y_tmp];
end

clear tmp_x tmp_y tmptmp_x tmptmp_y train_data_x_tmp train_data_y_tmp valid_data_x_tmp valid_data_y_tmp;
if phys_data
    save_file_name = sprintf('train_data_set(hw%d_pw%d).mat', hist_window-1, pred_window);
else
    save_file_name = sprintf('train_data_set(hw%d_pw%d).mat', hist_window, pred_window);
end
save(save_file_name, 'train_data_x', 'train_data_y', 'valid_data_x', 'valid_data_y');


%%
% list of data log for test data set
clear;  clc;
addpath('.\DataFile');

test_data_list = [];
test_data_list{1} = 'result_log_data_log_061716.mat';
test_data_list{2} = 'result_log_daily_run_101615.mat';
test_data_list{3} = 'result_log_data_log_061716_2.mat';

hist_window = 20;
pred_window = 5*25; % (25Hz 5seconds)
test_data_x = []; test_data_y = [];

phys_data = false;
gpr_data = true;
gpr_hist_window = 10;

for list_cnt = 1:length(test_data_list)
    fprintf(1, '%d-th test data start\n', list_cnt);
    load(test_data_list{list_cnt});
    log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1);
    
    left_speed = zeros(log_file_size+hist_window, 1);
    right_speed = zeros(log_file_size+hist_window, 1);
    forw_cmd = zeros(log_file_size+hist_window, 1);
    left_cmd = zeros(log_file_size+hist_window, 1);
    left_speed(hist_window+1:end) = Log.encoderLeftWheelSpeedSmoothed;
    right_speed(hist_window+1:end) = Log.encoderRightWheelSpeedSmoothed;
    forw_cmd(hist_window+1:end) = Log.forwardCommand;
    left_cmd(hist_window+1:end) = Log.leftCommand;
    
    
    log_file_size = log_file_size - pred_window + 1;
    if phys_data
        test_data_x_tmp = zeros(log_file_size, 2*(2+pred_window));
        test_data_y_tmp = zeros(log_file_size, 2*pred_window);
    elseif gpr_data
        test_data_x_tmp = zeros(log_file_size, 2*(gpr_hist_window+pred_window));
        test_data_y_tmp = zeros(log_file_size, 2*pred_window);
    else
        test_data_x_tmp = zeros(log_file_size, 2*(hist_window+pred_window));
        test_data_y_tmp = zeros(log_file_size, 2*pred_window);
    end
    
    for data_cnt = 1:log_file_size
        if data_cnt < 2
            tmp_x = []; tmp_y = [];
            for cnt2 = 0:hist_window-1
                tmp_x = [tmp_x, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
            for cnt2 = hist_window:hist_window+pred_window-1
                tmp_x = [tmp_x, forw_cmd(data_cnt+cnt2-1), left_cmd(data_cnt+cnt2-1)];
                tmp_y = [tmp_y, left_speed(data_cnt+cnt2), right_speed(data_cnt+cnt2)];
            end
        else
            tmptmp_x = tmp_x;
            tmptmp_y = tmp_y;
            tmp_x(1:2*hist_window-2) = tmptmp_x(3:2*hist_window);
            tmp_x(2*hist_window-1:2*hist_window) = tmptmp_y(1:2);
            tmp_x(2*hist_window+1:2*(hist_window+pred_window-1)) = tmptmp_x(2*hist_window+3:2*(hist_window+pred_window));
            tmp_x(2*(hist_window+pred_window)-1:2*(hist_window+pred_window)) = [forw_cmd(data_cnt+hist_window+pred_window-2), left_cmd(data_cnt+hist_window+pred_window-2)];
            tmp_y(1:2*pred_window-2) = tmptmp_y(3:2*pred_window);
            tmp_y(2*pred_window-1:2*pred_window) = [left_speed(data_cnt+hist_window+pred_window-1), right_speed(data_cnt+hist_window+pred_window-1)];
        end

        if phys_data
            tmptmp_x = tmp_x(2*hist_window-3:2*(hist_window+pred_window));
            acc1 = (tmp_x(3)-tmp_x(1))/(Log.plotTime(data_cnt+hist_window-1)-Log.plotTime(data_cnt+hist_window-2));
            acc2 = (tmp_x(4)-tmp_x(2))/(Log.plotTime(data_cnt+hist_window-1)-Log.plotTime(data_cnt+hist_window-2));
            tmptmp_x(1) = tmptmp_x(3);  tmptmp_x(3) = tmptmp_x(4);
            tmptmp_x(2) = acc1;         tmptmp_x(4) = acc2;
            
            test_data_x_tmp(data_cnt, :) = tmptmp_x;
            test_data_y_tmp(data_cnt, :) = tmp_y;
        elseif gpr_data
            tmptmp_x = tmp_x(2*(hist_window-gpr_hist_window)+1:end);
            
            test_data_x_tmp(data_cnt, :) = tmptmp_x;
            test_data_y_tmp(data_cnt, :) = tmp_y;
        else
            test_data_x_tmp(data_cnt, :) = tmp_x;
            test_data_y_tmp(data_cnt, :) = tmp_y;
        end
    end
    test_data_x = [test_data_x; test_data_x_tmp];
    test_data_y = [test_data_y; test_data_y_tmp];
end

if phys_data
    save('test_data_set_phys.mat', 'test_data_x', 'test_data_y');
elseif gpr_data
    save('test_data_set_gpr.mat', 'test_data_x', 'test_data_y');
else
    save('test_data_set.mat', 'test_data_x', 'test_data_y');
end

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