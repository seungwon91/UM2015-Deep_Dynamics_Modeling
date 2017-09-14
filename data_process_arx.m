%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 17.01.11 Updated
% Program to read several log files in \DataFile\
%         and make dataset appropriate to each model
%         use history of observed speed and history of command as input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;  clc;
robot_type = 'vulcan';
%robot_type = 'fetch';
%robot_type = 'magicbot';

% list of data log for training data set
if strcmp(robot_type, 'vulcan')
    %addpath('.\DataVulcan');
    addpath('DataVulcan');
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
    train_data_list{21} = 'result_log_data_hoon_dec22.mat';
    train_data_list{22} = 'result_log_data_seungwon_dec15_carpet.mat';
    train_data_list{23} = 'result_log_data_seungwon_dec15_marble.mat';
    train_data_list{24} = 'result_log_data_seungwon_dec21_carpet.mat';
    train_data_list{25} = 'result_log_data_seungwon_dec21_marble.mat';
    train_data_list{26} = 'result_log_data_seungwon_dec21_marble_turning.mat';
    train_data_list{27} = 'result_log_data_seungwon_oct24_carpet.mat';
    train_data_list{28} = 'result_log_data_seungwon_oct24_cement.mat';
    train_data_list{29} = 'result_log_data_seungwon_oct24_marble_BBB.mat';
    train_data_list{30} = 'result_log_data_seungwon_oct24_marble_DOW.mat';
    train_data_list{31} = 'result_log_data_yeongjin_dec22.mat';
    train_data_list{32} = 'result_log_data_yeongjin_dec23.mat';
    train_data_list{33} = 'result_log_data_yeonjoon_dec22.mat';
    train_data_list{34} = 'result_log_data_yeonjoon_dec23.mat';

    dim_obs_out = 2;    % number of output observed from the system(or robot)
    dim_cmd = 2;    % number of command signal of the system(or robot)
elseif strcmp(robot_type, 'fetch')
    addpath('.\DataFetch\mat');
    train_data_list = [];
    train_data_list{1} = 'result_log_fetch_Dec21.mat';
    train_data_list{2} = 'result_log_fetch_Dec21_2.mat';
    train_data_list{3} = 'result_log_fetch_Dec28_1.mat';
    train_data_list{4} = 'result_log_fetch_Dec28_2.mat';
    train_data_list{5} = 'result_log_fetch_Dec28_3.mat';
    train_data_list{6} = 'result_log_fetch_Dec28_rotation.mat';
    
    dim_obs_out = 2;    % number of output observed from the system(or robot)
    dim_cmd = 2;    % number of command signal of the system(or robot)

elseif strcmp(robot_type, 'magicbot')
    %addpath('.\DataMagicBot');
    addpath('DataMagicBot');
    train_data_list = [];
    train_data_list{1} = 'result_log_010617_131150.mat';
    train_data_list{2} = 'result_log_011017_210041.mat';
    train_data_list{3} = 'result_log_011017_211108.mat';
    train_data_list{4} = 'result_log_011017_212211.mat';
    train_data_list{5} = 'result_log_011017_212647.mat';
    train_data_list{6} = 'result_log_011017_213417.mat';
    train_data_list{7} = 'result_log_011017_214602.mat';
    train_data_list{8} = 'result_log_011017_224423.mat';
    train_data_list{9} = 'result_log_011017_224949.mat';
    train_data_list{10} = 'result_log_011117_213031.mat';
    
    dim_obs_out = 2;    % number of output observed from the system(or robot)
    dim_cmd = 2;    % number of command signal of the system(or robot)
end

speed_hw = 5;   % number of time-steps of past speed for model input
cmd_hw = 1;     % number of time-steps of past command for model input
ratio_of_valid = 0.95;

train_data_x = []; train_data_y = [];
valid_data_x = []; valid_data_y = [];
for list_cnt = 1:length(train_data_list)    
    max_hw = max(speed_hw, cmd_hw);
    fprintf(1, '%d-th training data start\n', list_cnt);
    load(train_data_list{list_cnt});
    if strcmp(robot_type, 'vulcan')
        log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1)-max_hw;
        obs_out = [];
        obs_out{1} = Log.encoderLeftWheelSpeedSmoothed;
        obs_out{2} = Log.encoderRightWheelSpeedSmoothed;
        cmd = [];
        cmd{1} = Log.forwardCommand;
        cmd{2} = Log.leftCommand;
    elseif strcmp(robot_type, 'fetch')
        log_file_size = size(Log.odometryLinearSpeedSmoothed, 1)-max_hw;
        obs_out = [];
        obs_out{1} = Log.odometryLinearSpeedSmoothed;
        obs_out{2} = Log.odometryAngularSpeedSmoothed;
        cmd = [];
        cmd{1} = Log.forwardCommand;
        cmd{2} = Log.lateralCommand;
    elseif strcmp(robot_type, 'magicbot')
        log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1)-max_hw;
        obs_out = [];
        obs_out{1} = Log.encoderLeftWheelSpeedSmoothed;
        obs_out{2} = Log.encoderRightWheelSpeedSmoothed;
        cmd = [];
        cmd{1} = Log.leftWheelCommand;
        cmd{2} = Log.rightWheelCommand;
    end
    
%     data_index = randperm(log_file_size);
% 
%     train_data_x_tmp = zeros(split_cnt-1, dim_obs_out*speed_hw+dim_cmd*cmd_hw);
%     train_data_y_tmp = zeros(split_cnt-1, dim_obs_out);
%     valid_data_x_tmp = zeros(log_file_size+1-split_cnt, dim_obs_out*speed_hw+dim_cmd*cmd_hw);
%     valid_data_y_tmp = zeros(log_file_size+1-split_cnt, dim_obs_out);
    data_x_tmp = zeros(log_file_size, dim_obs_out*speed_hw+dim_cmd*cmd_hw);
    data_y_tmp = zeros(log_file_size, dim_obs_out);
    over_limit_cnt = 0;
    
    for data_cnt = 1:log_file_size
        if data_cnt < 2
            tmp_x = [];     tmp_y = [];
            for cnt2 = max(0, max_hw-speed_hw):(max_hw-1)
                for cnt3 = 1:dim_obs_out
                    tmp_x = [tmp_x, obs_out{cnt3}(data_cnt+cnt2)];
                end
            end
            for cnt2 = max(0, max_hw-cmd_hw):(max_hw-1)
                for cnt3 = 1:dim_cmd
                    tmp_x = [tmp_x, cmd{cnt3}(data_cnt+cnt2)];
                end
            end
            for cnt2 = 1:dim_obs_out
                tmp_y = [tmp_y, obs_out{cnt2}(data_cnt+max_hw)];
            end
        else
            tmptmp_x = tmp_x;
            tmp_x(1:dim_obs_out*(speed_hw-1)) = tmptmp_x(dim_obs_out+1:dim_obs_out*speed_hw);
            tmp_x(dim_obs_out*(speed_hw-1)+1:dim_obs_out*speed_hw) = tmp_y;
            tmp_x(dim_obs_out*speed_hw+1:dim_obs_out*speed_hw+dim_cmd*(cmd_hw-1)) = tmptmp_x(dim_obs_out*speed_hw+dim_cmd+1:dim_obs_out*speed_hw+dim_cmd*cmd_hw);
            for cnt2 = 1:dim_cmd
                tmp_x(dim_obs_out*speed_hw+dim_cmd*(cmd_hw-1)+cnt2) = cmd{cnt2}(data_cnt+max_hw-1);
            end
            
            tmp_y = [];
            for cnt2 = 1:dim_obs_out
                tmp_y = [tmp_y, obs_out{cnt2}(data_cnt+max_hw)];
            end
        end

        over_limit = false;
        if strcmp(robot_type, 'fetch')
            for cnt2=1:cmd_hw
                ind = (dim_obs_out*speed_hw)+dim_cmd*(cnt2-1)+1;
                if (1.001*tmp_x(ind)-0.1832*tmp_x(ind+1)>1) || (1.001*tmp_x(ind)+0.1832*tmp_x(ind+1)>1)
                    over_limit = true;
                    break;
                end
            end
        end
        
        if over_limit
            over_limit_cnt = over_limit_cnt + 1;
        else
            data_x_tmp(data_cnt-over_limit_cnt, :) = tmp_x;
            data_y_tmp(data_cnt-over_limit_cnt, :) = tmp_y;
        end
%         perm_ind = data_cnt;
%         if perm_ind < split_cnt
%             train_data_x_tmp(perm_ind, :) = tmp_x;
%             train_data_y_tmp(perm_ind, :) = tmp_y;
%         else
%             valid_data_x_tmp(perm_ind+1-split_cnt, :) = tmp_x;
%             valid_data_y_tmp(perm_ind+1-split_cnt, :) = tmp_y;
%         end
    end
    
    data_index = randperm(log_file_size-over_limit_cnt);
    split_cnt = int64(length(data_index) * ratio_of_valid);
    train_data_x = [train_data_x; data_x_tmp(data_index(1:split_cnt-1),:)];
    train_data_y = [train_data_y; data_y_tmp(data_index(1:split_cnt-1),:)];
    valid_data_x = [valid_data_x; data_x_tmp(data_index(split_cnt:log_file_size-over_limit_cnt),:)];
    valid_data_y = [valid_data_y; data_y_tmp(data_index(split_cnt:log_file_size-over_limit_cnt),:)];
%     train_data_x = [train_data_x; train_data_x_tmp];
%     train_data_y = [train_data_y; train_data_y_tmp];
%     valid_data_x = [valid_data_x; valid_data_x_tmp];
%     valid_data_y = [valid_data_y; valid_data_y_tmp];
end

clear tmp_x tmp_y tmptmp_x tmptmp_y train_data_x_tmp train_data_y_tmp valid_data_x_tmp valid_data_y_tmp;
save_file_name = sprintf('train_data_%s(hw%d_%d).mat', robot_type, speed_hw, cmd_hw);
save(save_file_name, 'train_data_x', 'train_data_y', 'valid_data_x', 'valid_data_y');

%%
clear;  clc;
robot_type = 'vulcan';
%robot_type = 'fetch';
%robot_type = 'magicbot';

% list of data log for test data set
if strcmp(robot_type, 'vulcan')
    %addpath('.\DataVulcan');
    addpath('DataVulcan');
    test_data_list = [];
    test_data_list{1} = 'result_log_data_log_061716.mat';
    test_data_list{2} = 'result_log_daily_run_101615.mat';
    test_data_list{3} = 'result_log_data_log_061716_2.mat';
    
    dim_obs_out = 2;    % number of output observed from the system(or robot)
    dim_cmd = 2;    % number of command signal of the system(or robot)
elseif strcmp(robot_type, 'fetch')
    addpath('.\DataFetch\mat');
    test_data_list = [];
    test_data_list{1} = 'result_log_fetch_test_1.mat';
    
    dim_obs_out = 2;    % number of output observed from the system(or robot)
    dim_cmd = 2;    % number of command signal of the system(or robot)
elseif strcmp(robot_type, 'magicbot')
    %addpath('.\DataMagicBot');
    addpath('DataMagicBot');
    test_data_list = [];
    test_data_list{1} = 'result_test_log_011017_221849.mat';
    test_data_list{2} = 'result_test_log_011117_220413.mat';
    
    dim_obs_out = 2;    % number of output observed from the system(or robot)
    dim_cmd = 2;    % number of command signal of the system(or robot)
end

speed_hw = 5;
cmd_hw = 1;
test_data_x = []; test_data_y = [];

for list_cnt = 1:length(test_data_list)
    max_hw = max(speed_hw, cmd_hw);
    fprintf(1, '%d-th test data start\n', list_cnt);
    load(test_data_list{list_cnt});
    if strcmp(robot_type, 'vulcan')
        log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1);
        obs_out = [];
        obs_out{1} = zeros(log_file_size+max_hw, 1);
        obs_out{1}(max_hw+1:end) = Log.encoderLeftWheelSpeedSmoothed;
        obs_out{2} = zeros(log_file_size+max_hw, 1);
        obs_out{2}(max_hw+1:end) = Log.encoderRightWheelSpeedSmoothed;
        cmd = [];
        cmd{1} = zeros(log_file_size+max_hw, 1);
        cmd{1}(max_hw+1:end) = Log.forwardCommand;
        cmd{2} = zeros(log_file_size+max_hw, 1);
        cmd{2}(max_hw+1:end) = Log.leftCommand;
    elseif strcmp(robot_type, 'fetch')
        log_file_size = size(Log.odometryLinearSpeedSmoothed, 1);
        obs_out = [];
        obs_out{1} = zeros(log_file_size+max_hw, 1);
        obs_out{1}(max_hw+1:end) = Log.odometryLinearSpeedSmoothed;
        obs_out{2} = zeros(log_file_size+max_hw, 1);
        obs_out{2}(max_hw+1:end) = Log.odometryAngularSpeedSmoothed;
        cmd = [];
        cmd{1} = zeros(log_file_size+max_hw, 1);
        cmd{1}(max_hw+1:end) = Log.forwardCommand;
        cmd{2} = zeros(log_file_size+max_hw, 1);
        cmd{2}(max_hw+1:end) = Log.lateralCommand;
    elseif strcmp(robot_type, 'magicbot')
        log_file_size = size(Log.encoderLeftWheelSpeedSmoothed, 1);
        obs_out = [];
        obs_out{1} = zeros(log_file_size+max_hw, 1);
        obs_out{1}(max_hw+1:end) = Log.encoderLeftWheelSpeedSmoothed;
        obs_out{2} = zeros(log_file_size+max_hw, 1);
        obs_out{2}(max_hw+1:end) = Log.encoderRightWheelSpeedSmoothed;
        cmd = [];
        cmd{1} = zeros(log_file_size+max_hw, 1);
        cmd{1}(max_hw+1:end) = Log.leftWheelCommand;
        cmd{2} = zeros(log_file_size+max_hw, 1);
        cmd{2}(max_hw+1:end) = Log.rightWheelCommand;
    end
    log_file_size = log_file_size - 124;
    test_data_x_tmp = zeros(log_file_size, dim_obs_out*speed_hw+dim_cmd*(cmd_hw + 124));
    test_data_y_tmp = zeros(log_file_size, dim_obs_out*125);
    over_limit_cnt = 0;
    
    for data_cnt = 1:log_file_size
        if data_cnt < 2
            tmp_x = []; tmp_y = [];
            
            for cnt2 = max(0, max_hw-speed_hw):(max_hw-1)
                for cnt3 = 1:dim_obs_out
                    tmp_x = [tmp_x, obs_out{cnt3}(data_cnt+cnt2)];
                end
            end
            for cnt2 = max(0, max_hw-cmd_hw):(max_hw-2)
                for cnt3 = 1:dim_cmd
                    tmp_x = [tmp_x, cmd{cnt3}(data_cnt+cnt2)];
                end
            end
            
            for cnt2 = max_hw-1:max_hw+123
                for cnt3 = 1:dim_cmd
                    tmp_x = [tmp_x, cmd{cnt3}(data_cnt+cnt2)];
                end
                for cnt3 = 1:dim_obs_out
                    tmp_y = [tmp_y, obs_out{cnt3}(data_cnt+cnt2+1)];
                end
            end
        else
            tmptmp_x = tmp_x;
            tmptmp_y = tmp_y;
            tmp_x(1:dim_obs_out*(speed_hw-1)) = tmptmp_x(dim_obs_out+1:dim_obs_out*speed_hw);
            tmp_x(dim_obs_out*(speed_hw-1)+1:dim_obs_out*speed_hw) = tmp_y(1:dim_obs_out);
            tmp_x(dim_obs_out*speed_hw+1:dim_obs_out*speed_hw+dim_cmd*(cmd_hw+123)) = tmptmp_x(dim_obs_out*speed_hw+dim_cmd+1:dim_obs_out*speed_hw+dim_cmd*(cmd_hw+124));
            for cnt2 = 1:dim_cmd
                tmp_x(dim_obs_out*speed_hw+dim_cmd*(cmd_hw+123)+cnt2) = cmd{cnt2}(data_cnt+max_hw+123);
            end
            tmp_y(1:dim_obs_out*124) = tmptmp_y(dim_obs_out+1:dim_obs_out*125);
            for cnt2 = 1:dim_obs_out
                tmp_y(dim_obs_out*124+cnt2) = obs_out{cnt2}(data_cnt+max_hw+124);
            end
        end
        
        over_limit = false;
        if strcmp(robot_type, 'fetch')
            for cnt2=1:cmd_hw
                ind = (dim_obs_out*speed_hw)+dim_cmd*(cnt2-1)+1;
                if (1.001*tmp_x(ind)-0.1832*tmp_x(ind+1)>1) || (1.001*tmp_x(ind)+0.1832*tmp_x(ind+1)>1)
                    over_limit = true;
                    break;
                end
            end
        end
        
        if over_limit
            over_limit_cnt = over_limit_cnt + 1;
        else
            test_data_x_tmp(data_cnt-over_limit_cnt, :) = tmp_x;
            test_data_y_tmp(data_cnt-over_limit_cnt, :) = tmp_y;
        end
%         test_data_x_tmp(data_cnt, :) = tmp_x;
%         test_data_y_tmp(data_cnt, :) = tmp_y;
    end
    test_data_x = [test_data_x; test_data_x_tmp(1:(log_file_size-over_limit_cnt),:)];
    test_data_y = [test_data_y; test_data_y_tmp(1:(log_file_size-over_limit_cnt),:)];
%     test_data_x = [test_data_x; test_data_x_tmp];
%     test_data_y = [test_data_y; test_data_y_tmp];
end
save_file_name = sprintf('test_data_%s(hw%d_%d).mat', robot_type, speed_hw, cmd_hw);
save(save_file_name, 'test_data_x', 'test_data_y');

%% Make data file for reference input

clear;	clc;
reference_input = [];
phys_data = false;
if phys_data
	num_state = 4;
else
	num_state = 40;
end

speed_hw = 4;
cmd_hw = 4;

for forw_cnt=0:5:100
	for left_cnt = -100:5:100
		tmp_x = zeros(1, 2*(speed_hw+cmd_hw+124));

		for cnt = 2*(speed_hw+cmd_hw+25)+1:2:2*(speed_hw+cmd_hw+124)
			tmp_x(cnt) = forw_cnt;
			tmp_x(cnt+1) = left_cnt;
		end
		reference_input = [reference_input;tmp_x];
	end
end


for forw_cnt=0:5:100
	for left_cnt = -100:5:100
		tmp_x = zeros(1, 2*(speed_hw+cmd_hw+124));
        forw_slope = forw_cnt/100;
		left_slope = left_cnt/100;

		for cnt = 2*(speed_hw+cmd_hw+25)+1:2:2*(speed_hw+cmd_hw+124)
			tmp_x(cnt) = forw_slope * (cnt+1-2*(speed_hw+cmd_hw+25))/2;
			tmp_x(cnt+1) = left_slope * (cnt+1-2*(speed_hw+cmd_hw+25))/2;
		end
		reference_input = [reference_input;tmp_x];
	end
end

save_file_name = sprintf('test_data_reference_input(hw%d_%d).mat', speed_hw, cmd_hw);
save(save_file_name, 'reference_input');
