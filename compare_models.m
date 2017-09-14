%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.07.31 Updated
% Program to compare physics-based model, LWPR-based model, GPR-based
%                    model, 3-layer FeedForward Neural Net,
%                    and Deep CNN-based model
%
% train_time / test_time / train_error_history / model_prediction
% test_data_output / difference_btw_prediction_data / test_error / joystick_command
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% error on different length of simulation

close all; clear; clc;

plot_all = true;
mode = 5;
single_step_cost = false;
plot_on_steps = false;
plot_legend = true;

if plot_all
    for mode=1:5
        if mode == 1
            % load('./Physics-based/result/5sec_LTS_result(Physics_model).mat', 'difference_btw_prediction_data');
            load('./Physics-based/new_result/5sec_LTS_result(Physics_model).mat', 'difference_btw_prediction_data');
            num_data = size(difference_btw_prediction_data, 1);
            error_phys = zeros(125, 3);
            for pred_cnt = 1:125
                error_phys(pred_cnt, 1) = sum(sum(abs(difference_btw_prediction_data(:,1:2*pred_cnt))))/(2*num_data*pred_cnt);
                error_phys(pred_cnt, 2) = sqrt(sum(sum(difference_btw_prediction_data(:,1:2*pred_cnt).^2))/(2*num_data*pred_cnt));
                error_phys(pred_cnt, 3) = max(max(abs(difference_btw_prediction_data(:,1:2*pred_cnt)), [], 2));
                
                if mod(pred_cnt, 25) == 0
                    fprintf(1, '\t%d\n', pred_cnt);
                end
            end
        elseif mode == 2
            % load('./LWPR/train(D06F_40_hw3_1)/LWPR_1D_model_test_result.mat');
            load('./LWPR/model(D04F_40_hw4_1)/LWPR_1D_model_test_result.mat', 'model_output_on_test_data');
            load('./LWPR/test_data_set(hw4_1).mat');
            difference_btw_prediction_data = model_output_on_test_data - test_data_y;
            error_lwpr = zeros(125, 3);
            for pred_cnt = 1:125
                error_lwpr(pred_cnt, 1) = sum(sum(abs(difference_btw_prediction_data(:,1:2*pred_cnt))))/(2*num_data*pred_cnt);
                error_lwpr(pred_cnt, 2) = sqrt(sum(sum(difference_btw_prediction_data(:,1:2*pred_cnt).^2))/(2*num_data*pred_cnt));
                error_lwpr(pred_cnt, 3) = max(max(abs(difference_btw_prediction_data(:,1:2*pred_cnt)), [], 2));
                
                if mod(pred_cnt, 25) == 0
                    fprintf(1, '\t%d\n', pred_cnt);
                end
            end
        elseif mode == 3
            load('./GPR/subsampled_model(size4000_hw4_1)/Test_Result_of_GPR(size4000_epoch45).mat', 'pred_on_test_data', 'test_data_y');
            difference_btw_prediction_data = pred_on_test_data - test_data_y;
            error_gpr = zeros(125, 3);
            for pred_cnt = 1:125
                error_gpr(pred_cnt, 1) = sum(sum(abs(difference_btw_prediction_data(:,1:2*pred_cnt))))/(2*num_data*pred_cnt);
                error_gpr(pred_cnt, 2) = sqrt(sum(sum(difference_btw_prediction_data(:,1:2*pred_cnt).^2))/(2*num_data*pred_cnt));
                error_gpr(pred_cnt, 3) = max(max(abs(difference_btw_prediction_data(:,1:2*pred_cnt)), [], 2));
                
                if mod(pred_cnt, 25) == 0
                    fprintf(1, '\t%d\n', pred_cnt);
                end
            end
        elseif mode == 4
            load('./FeedForwardNet/5sec_LTS_result(hw20_pw25_b64_epoch1720).mat', 'difference_btw_prediction_data');
            error_ffnn = zeros(125, 3);
            for pred_cnt = 1:125
                error_ffnn(pred_cnt, 1) = sum(sum(abs(difference_btw_prediction_data(:,1:2*pred_cnt))))/(2*num_data*pred_cnt);
                error_ffnn(pred_cnt, 2) = sqrt(sum(sum(difference_btw_prediction_data(:,1:2*pred_cnt).^2))/(2*num_data*pred_cnt));
                error_ffnn(pred_cnt, 3) = max(max(abs(difference_btw_prediction_data(:,1:2*pred_cnt)), [], 2));
                
                if mod(pred_cnt, 25) == 0
                    fprintf(1, '\t%d\n', pred_cnt);
                end
            end
        else
            if single_step_cost
                load('./DeepLearning/cnn3_k128_96_hid128_hw20_pw1_resultplots/5sec_LTS_result(hw20_pw1_b64_epoch6000).mat', 'difference_btw_prediction_data');
            else
                load('./DeepLearning/cnn3_k128_96_hid128_hw20_pw25_resultplots/5sec_LTS_result(hw20_pw25_b64_epoch650).mat', 'difference_btw_prediction_data');
            end
            error_cnn = zeros(125, 3);
            for pred_cnt = 1:125
                error_cnn(pred_cnt, 1) = sum(sum(abs(difference_btw_prediction_data(:,1:2*pred_cnt))))/(2*num_data*pred_cnt);
                error_cnn(pred_cnt, 2) = sqrt(sum(sum(difference_btw_prediction_data(:,1:2*pred_cnt).^2))/(2*num_data*pred_cnt));
                error_cnn(pred_cnt, 3) = max(max(abs(difference_btw_prediction_data(:,1:2*pred_cnt)), [], 2));
                
                if mod(pred_cnt, 25) == 0
                    fprintf(1, '\t%d\n', pred_cnt);
                end
            end
        end
    end
    
    ind_x = 1:4:125;
    if plot_on_steps
        plot_x = ind_x;
    else
        plot_x = ind_x/25;
    end
    
    mark_size = 17;
    line_width = 1.5;
    leg_font_size = 24;
    lab_font_size = 24;
    
    fig = figure();
    hold on;
    plot(plot_x, error_phys(ind_x, 2), 'bo-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_lwpr(ind_x, 2), 'g^-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_gpr(ind_x, 2), 'cp-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_ffnn(ind_x, 2), 'md-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_cnn(ind_x, 2), 'rs-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    hold off;
    set(gca, 'FontSize', 18);
    if plot_legend
        leg = legend('Physics', 'LWPR', 'GPR', 'FFNN', 'DD-CNN', 'Location', 'SouthEast', 'Orientation', 'horizontal');
%         leg = legend('Physics', 'LWPR', 'FeedForward', 'DeepDynamics-CNN', 'Location', 'SouthEast');
        set(leg, 'FontSize', leg_font_size);
    end
    if plot_on_steps
        xlabel('# of Simulation Steps', 'FontSize', lab_font_size);
    else
        xlabel('Length of Simulation(sec)', 'FontSize', lab_font_size);
    end
    ylabel('RMS error', 'FontSize', lab_font_size);
    %title('Error according to length of simulation', 'FontSize', 16);
    
    fig2 = figure();
    hold on;
    plot(plot_x, error_phys(ind_x, 3), 'bo-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_lwpr(ind_x, 3), 'g^-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_gpr(ind_x, 3), 'cp-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_ffnn(ind_x, 3), 'md-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    plot(plot_x, error_cnn(ind_x, 3), 'rs-', 'LineWidth', line_width, 'MarkerSize', mark_size);
    hold off;
    set(gca, 'FontSize', 18);
    if plot_legend
        leg = legend('Physics', 'LWPR', 'GPR', 'FFNN', 'DD-CNN', 'Location', 'SouthEast', 'Orientation', 'horizontal');
%         leg = legend('Physics', 'LWPR', 'FeedForward', 'DeepDynamics-CNN', 'Location', 'SouthEast');
        set(leg, 'FontSize', leg_font_size);
    end
    if plot_on_steps
        xlabel('# of Simulation Steps', 'FontSize', lab_font_size);
    else
        xlabel('Length of Simulation(sec)', 'FontSize', lab_font_size);
    end
    ylabel('L_{\infty} error(m/s)', 'FontSize', lab_font_size);
    %title('Error according to length of simulation', 'FontSize', 16);
else
    if mode == 1
        % load('./Physics-based/result/5sec_LTS_result(Physics_model).mat', 'difference_btw_prediction_data');
        load('./Physics-based/new_result/5sec_LTS_result(Physics_model).mat', 'difference_btw_prediction_data');
    elseif mode == 2
        load('./LWPR/model(D04F_40_hw4_1)/LWPR_1D_model_test_result.mat', 'model_output_on_test_data');
        load('./LWPR/test_data_set(hw4_1).mat');
        difference_btw_prediction_data = model_output_on_test_data - test_data_y;
    elseif mode == 3
        load('./GPR/subsampled_model(size4000_hw4_1)/Test_Result_of_GPR(size4000_epoch45).mat', 'pred_on_test_data', 'test_data_y');
        difference_btw_prediction_data = pred_on_test_data - test_data_y;
    elseif mode == 4
        load('./FeedForwardNet/5sec_LTS_result(hw20_pw25_b64_epoch1720).mat', 'difference_btw_prediction_data');
    else
        if single_step_cost
            load('./DeepLearning/cnn3_k128_96_hid128_hw20_pw1_resultplots/5sec_LTS_result(hw20_pw1_b64_epoch6000).mat', 'difference_btw_prediction_data');
        else
            load('./DeepLearning/cnn3_k128_96_hid128_hw20_pw25_resultplots/5sec_LTS_result(hw20_pw25_b64_epoch650).mat', 'difference_btw_prediction_data');
        end
    end

    num_data = size(difference_btw_prediction_data, 1);
    error = zeros(125, 3);
    for pred_cnt = 1:125
        error(pred_cnt, 1) = sum(sum(abs(difference_btw_prediction_data(:,1:2*pred_cnt))))/(2*num_data*pred_cnt);
        error(pred_cnt, 2) = sqrt(sum(sum(difference_btw_prediction_data(:,1:2*pred_cnt).^2))/(2*num_data*pred_cnt));
        error(pred_cnt, 3) = max(max(abs(difference_btw_prediction_data(:,1:2*pred_cnt)), [], 2));

        if mod(pred_cnt, 25) == 0
            fprintf(1, '\t%d\n', pred_cnt);
        end
    end

    ind_x = 1:4:125;
    if plot_on_steps
        plot_x = ind_x;
    else
        plot_x = ind_x/25;
    end
    fig = figure();
    hold on;
    plot(plot_x, error(ind_x, 1), 'rx-', 'MarkerSize', 12);
    plot(plot_x, error(ind_x, 2), 'bo-', 'MarkerSize', 12);
    plot(plot_x, error(ind_x, 3), 'g^-', 'MarkerSize', 12);
    hold off;
    if plot_legend
        leg = legend('MAE', 'RMS', 'Linf');
    end
    if plot_on_steps
        xlabel('# of Simulation Steps', 'FontSize', 13);
    else
        xlabel('Length of Simulation(sec)', 'FontSize', 13);
    end
    ylabel('error(m/s)', 'FontSize', 13);
    %title('Error according to length of simulation', 'FontSize', 16);
end

%% Model Comparison(Error/Inference Time)

close all; clear; clc;

load('./Physics-based/result/5sec_LTS_result(Physics_model).mat');
num_test_data = size(difference_btw_prediction_data, 1);
phys_error = [sum(sum(abs(difference_btw_prediction_data)))/num_test_data, sqrt(sum(sum(difference_btw_prediction_data.^2))/num_test_data), max(max(abs(difference_btw_prediction_data),[],2))];
phys_time = time_for_simulation/num_test_data * 1000;


% load('./LWPR/LWPR_1D_model_test_result(epoch16).mat');
load('./LWPR/train(D06F_40_hw3)/LWPR_1D_model_test_result.mat');
lwpr_error = [test_error(1), test_error(2), test_error(3)];
lwpr_time = test_time/num_test_data * 1000;


load('./GPR/result/Result_of_GPR_boost_9.mat');
gpr_error = [error(1), error(2), error(3)];
gpr_time = pred_time/num_test_data * 1000;


load('./FeedForwardNet/5sec_LTS_result(hw20_pw25_b64_epoch1720).mat');
fc_error = [error(1), error(2), error(3)];
fc_time = time_for_simulation/num_test_data * 1000;


load('./DeepLearning/cnn3_k128_96_hid128_hw20_pw25_resultplots/5sec_LTS_result(hw20_pw25_b64_epoch650).mat');
cnn_error = [error(1), error(2), error(3)];
cnn_time = time_for_simulation/num_test_data * 1000;

set(0,'DefaultFigureVisible','on');
fig = figure();
hold on;
b1 = bar(1, phys_error(2), 'g');
b2 = bar(2, lwpr_error(2), 'b');
b3 = bar(3, gpr_error(2), 'c');
b4 = bar(4, fc_error(2), 'm');
b5 = bar(5, cnn_error(2), 'r');
hold off;
ylim([0 3]);
set(gca,'XTick',1:5);
set(gca, 'XTickLabel', {'Phys', 'LWPR', 'GPR', 'FC', 'CNN'}, 'FontSize', 13);
set(gca,'LooseInset',get(gca,'TightInset'))
ylabel('RMS error(m/s)', 'FontSize', 13);
title('RMS error on different models', 'FontSize', 16);

fig2 = figure();
hold on;
b1 = bar(1, phys_error(3), 'g');
b2 = bar(2, lwpr_error(3), 'b');
b3 = bar(3, gpr_error(3), 'c');
b4 = bar(4, fc_error(3), 'm');
b5 = bar(5, cnn_error(3), 'r');
hold off;
ylim([0 3]);
set(gca,'XTick',1:5);
set(gca, 'XTickLabel', {'Phys', 'LWPR', 'GPR', 'FC', 'CNN'}, 'FontSize', 13);
set(gca,'LooseInset',get(gca,'TightInset'))
ylabel('$$L_{\inf}$$ error(m/s)', 'FontSize', 14, 'interpreter','latex');
title('$$L_{\inf}$$ error on different models', 'FontSize', 17 ,'interpreter','latex');

fig3 = figure();
hold on;
b1 = bar(1, phys_time, 'g');
b2 = bar(2, lwpr_time, 'b');
b3 = bar(3, gpr_time, 'c');
b4 = bar(4, fc_time, 'm');
b5 = bar(5, cnn_time, 'r');
hold off;
ylim([0 10]);
set(gca,'XTick',1:5);
set(gca, 'XTickLabel', {'Phys', 'LWPR', 'GPR', 'FC', 'CNN'}, 'FontSize', 13);
set(gca,'LooseInset',get(gca,'TightInset'))
ylabel('Inference Time(millisecond)', 'FontSize', 13);
title('Inference Time for 5 seconds trajectory', 'FontSize', 16);


%% Model Comparison(Time-series plot at specific time)
close all; clear; clc;

load('./Physics-based/new_result/5sec_LTS_result(Physics_model).mat');
num_test_data = size(difference_btw_prediction_data, 1);
phys_output = model_prediction;
observed_speed = phys_output - difference_btw_prediction_data;

load('./LWPR/model(D04F_40_hw4_1)/LWPR_1D_model_test_result.mat');
lwpr_output = model_output_on_test_data;

load('./GPR/subsampled_model(size4000_hw4_1)/Test_Result_of_GPR(size4000_epoch45).mat');
gpr_output = pred_on_test_data;

load('./FeedForwardNet/5sec_LTS_result(hw20_pw25_b64_epoch1720).mat');
fc_output = model_prediction;

load('./DeepLearning/cnn_model_result/cnn3_k128_96_hid128_hw20_pw25_resultplots/5sec_LTS_result(hw20_pw25_b64_epoch650).mat');
cnn_output = model_prediction;

forw_cmd = joystick_command(:, 1:2:250)/100;
left_cmd = joystick_command(:, 2:2:250)/100;

% for cnt = 1:4
%     if cnt==1
%         [~, plot_cnt] = max(phys_error);
%         title_l = 'Model Prediction/Measurement(Left Wheel, worst case of Physcis)';
%         title_r = 'Model Prediction/Measurement(Right Wheel, worst case of Physcis)';
%     elseif cnt==2
%         [~, plot_cnt] = max(lwpr_error);
%         title_l = 'Model Prediction/Measurement(Left Wheel, worst case of LWPR)';
%         title_r = 'Model Prediction/Measurement(Right Wheel, worst case of LWPR)';
%     elseif cnt==3
%         [~, plot_cnt] = max(fc_error);
%         title_l = 'Model Prediction/Measurement(Left Wheel, worst case of FC)';
%         title_r = 'Model Prediction/Measurement(Right Wheel, worst case of FC)';
%     elseif cnt==4
%         [~, plot_cnt] = max(cnn_error);
%         title_l = 'Model Prediction/Measurement(Left Wheel, worst case of CNN)';
%         title_r = 'Model Prediction/Measurement(Right Wheel, worst case of CNN)';
%     end
%     
%     %     fig = figure('Position', [10, 10, 1550, 850]);
%     fig = figure();
%     subplot(2,1,1);
%     hold on;
%     plot(0:0.04:4.96, phys_output(plot_cnt, 1:2:250), 'g-');
%     plot(0:0.04:4.96, lwpr_output(plot_cnt, 1:2:250), 'b-');
%     plot(0:0.04:4.96, gpr_output(plot_cnt, 1:2:250), 'c-');
%     plot(0:0.04:4.96, fc_output(plot_cnt, 1:2:250), 'm-');
%     plot(0:0.04:4.96, cnn_output(plot_cnt, 1:2:250), 'r-');
%     plot(0:0.04:4.96, observed_speed(plot_cnt, 1:2:250), 'y-');
%     stairs(0:0.04:4.96, forw_cmd(plot_cnt,:), 'k--');
%     stairs(0:0.04:4.96, left_cmd(plot_cnt,:), 'k-.');
%     hold off;
%     legend('Phys', 'LWPR', 'GPR', 'FC', 'CNN', 'Measured', 'Forward CMD', 'Left CMD', 'Location', 'eastoutside');
%     grid on;    grid minor;
%     set(gca,'LooseInset',get(gca,'TightInset'))
%     title(title_l, 'FontSize', 16);
%     xlabel('time(sec)', 'FontSize', 13); ylabel('wheel speed(m/s)', 'FontSize', 13);
%     
%     
%     subplot(2,1,2);
%     hold on;
%     plot(0:0.04:4.96, phys_output(plot_cnt, 2:2:250), 'g-');
%     plot(0:0.04:4.96, lwpr_output(plot_cnt, 2:2:250), 'b-');
%     plot(0:0.04:4.96, gpr_output(plot_cnt, 2:2:250), 'c-');
%     plot(0:0.04:4.96, fc_output(plot_cnt, 2:2:250), 'm-');
%     plot(0:0.04:4.96, cnn_output(plot_cnt, 2:2:250), 'r-');
%     plot(0:0.04:4.96, observed_speed(plot_cnt, 2:2:250), 'y-');
%     stairs(0:0.04:4.96, forw_cmd(plot_cnt,:), 'k--');
%     stairs(0:0.04:4.96, left_cmd(plot_cnt,:), 'k-.');
%     hold off;
%     legend('Phys', 'LWPR', 'GPR', 'FC', 'CNN', 'Measured', 'Forward CMD', 'Left CMD', 'Location', 'eastoutside');
%     grid on;    grid minor;
%     set(gca,'LooseInset',get(gca,'TightInset'))
%     title(title_r, 'FontSize', 16);
%     xlabel('time(sec)', 'FontSize', 13); ylabel('wheel speed(m/s)', 'FontSize', 13);
% end

% plot_cnt = 537*25;
plot_cnt = 5195;
observed_speed_l = observed_speed(plot_cnt, 1:2:250);
observed_speed_r = observed_speed(plot_cnt, 2:2:250);

line_width = 2;   mark_size = 4;
leg_font_size = 22; lab_font_size = 22;

max_y_left = max([max(forw_cmd), max(left_cmd), max(phys_output(plot_cnt, 1:2:250)), ...
                  max(lwpr_output(plot_cnt, 1:2:250)), max(gpr_output(plot_cnt, 1:2:250)), ...
                  max(fc_output(plot_cnt, 1:2:250)), max(cnn_output(plot_cnt, 1:2:250)), ...
                  max(observed_speed_l)]);
min_y_left = min([min(forw_cmd), min(left_cmd), min(phys_output(plot_cnt, 1:2:250)), ...
                  min(lwpr_output(plot_cnt, 1:2:250)), min(gpr_output(plot_cnt, 1:2:250)), ...
                  min(fc_output(plot_cnt, 1:2:250)), min(cnn_output(plot_cnt, 1:2:250)), ...
                  min(observed_speed_l)]);
max_y_right = max([max(forw_cmd), max(left_cmd), max(phys_output(plot_cnt, 2:2:250)), ...
                   max(lwpr_output(plot_cnt, 2:2:250)), max(gpr_output(plot_cnt, 2:2:250)), ...
                   max(fc_output(plot_cnt, 2:2:250)), max(cnn_output(plot_cnt, 2:2:250)), ...
                   max(observed_speed_r)]);
min_y_right = min([min(forw_cmd), min(left_cmd), min(phys_output(plot_cnt, 2:2:250)), ...
                   min(lwpr_output(plot_cnt, 2:2:250)), min(gpr_output(plot_cnt, 2:2:250)), ...
                   min(fc_output(plot_cnt, 2:2:250)), min(cnn_output(plot_cnt, 2:2:250)), ...
                   min(observed_speed_r)]);

for cnt = 1:5
    if cnt==1
        model_output_l = phys_output(plot_cnt, 1:2:250);
        model_output_r = phys_output(plot_cnt, 2:2:250);
    elseif cnt==2
        model_output_l = lwpr_output(plot_cnt, 1:2:250);
        model_output_r = lwpr_output(plot_cnt, 2:2:250);
    elseif cnt==3
        model_output_l = gpr_output(plot_cnt, 1:2:250);
        model_output_r = gpr_output(plot_cnt, 2:2:250);
    elseif cnt==4
        model_output_l = fc_output(plot_cnt, 1:2:250);
        model_output_r = fc_output(plot_cnt, 2:2:250);
    else
        model_output_l = cnn_output(plot_cnt, 1:2:250);
        model_output_r = cnn_output(plot_cnt, 2:2:250);
    end
    diff_l = model_output_l - observed_speed_l;
    diff_r = model_output_r - observed_speed_r;
    
    %     fig = figure('Position', [10, 10, 1550, 850]);
    fig = figure();
    subplot(2,1,1);
    hold on;
    p1 = plot(0:0.04:4.96, model_output_l, 'b-', 'LineWidth', line_width);
    p2 = plot(0:0.04:4.96, observed_speed_l, 'm--', 'LineWidth', line_width);
    p3 = plot(0:0.04:4.96, diff_l, 'ro-', 'LineWidth', line_width, 'MarkerSize', mark_size,'MarkerFaceColor','red');
%     stairs(0:0.04:4.96, forw_cmd(plot_cnt,:), 'k:', 'LineWidth', line_width);
%     stairs(0:0.04:4.96, left_cmd(plot_cnt,:), 'k-.', 'LineWidth', line_width);
    hold off;
    grid on;    grid minor;
    ylim([-0.6 1.9]);
%     if max_y_left - min_y_left < 0.1
%         ylim([-0.2 0.2]);
%     else
%         ylim([min_y_left-0.2 max_y_left+0.2]);
%     end
    ylabel('Left Wheel Speed(m/s)', 'FontSize', lab_font_size);
    set(gca, 'FontSize', leg_font_size-4);
        
    subplot(2,1,2);
    hold on;
    p1 = plot(0:0.04:4.96, model_output_r, 'b-', 'LineWidth', line_width);
    p2 = plot(0:0.04:4.96, observed_speed_r, 'm--', 'LineWidth', line_width);
    p3 = plot(0:0.04:4.96, diff_r, 'ro-', 'LineWidth', line_width, 'MarkerSize', mark_size,'MarkerFaceColor','red');
%     p4 = stairs(0:0.04:4.96, forw_cmd(plot_cnt,:), 'k:', 'LineWidth', line_width);
%     p5 = stairs(0:0.04:4.96, left_cmd(plot_cnt,:), 'k-.', 'LineWidth', line_width);
    hold off;
    grid on;    grid minor;
    ylim([-0.6 1.9]);
%     if max_y_right - min_y_right < 0.1
%         ylim([-0.2 0.2]);
%     else
%         ylim([min_y_right-0.2 max_y_right+0.2]);
%     end
    xlabel('Length of Simulation(sec)', 'FontSize', lab_font_size);
    ylabel('Right Wheel Speed(m/s)', 'FontSize', lab_font_size);
%     leg = legend([p1, p2, p3, p4, p5], {'model prediction', 'measured speed', 'difference', 'joystick forward cmd', 'joystick left cmd'});
    leg = legend([p1, p2, p3], {'model prediction', 'measured speed', 'difference'});
    set(leg, 'Position', [0.62 0.35 0.35 0.33]);
    set(gca, 'FontSize', leg_font_size-4);
    set(leg, 'FontSize', leg_font_size);
end

%% Model Comparison on Reference Input
close all; clear; clc;

robot_resp_index = 1:33;
model_resp_index = [72;70;69;68;67;66;65;64;60;53;152;148;145;143;140;136;234;230;...
                    226;223;219;316;310;306;300;355;350;348;343;28;24;18;14];

load('./Physics-based/new_result/ref_input_result(Physics_model).mat', 'model_prediction');
ref_resp_phys = model_prediction;

load('./LWPR/model(D04F_40_hw4_1)/ref_input_result_LWPR.mat', 'model_output_on_test_data');
ref_resp_lwpr = model_output_on_test_data;

load('./GPR/subsampled_model(size4000_hw4_1)/ref_input_result_GPR(size4000_epoch45).mat');
ref_resp_gpr = pred_on_test_data;

load('./FeedForwardNet/ref_input_result_MLP(hw20_pw25_b64_epoch1720).mat', 'model_prediction');
ref_resp_ffnn = model_prediction;

load('./DeepLearning/cnn_model_result/cnn3_k128_96_hid128_hw20_pw25_resultplots/ref_input_result(hw20_pw25_b64_epoch650).mat', 'model_prediction', 'joystick_command');
ref_resp_ddcnn = model_prediction;

load('robot_response_to_reference_input.mat', 'real_response');
ref_resp_robot = real_response;
num_data = size(real_response, 1);

line_width = 1.5;
mark_size = 4;
leg_font_size = 22;
lab_font_size = 22;

plot_cnt = 18;
plot_x = 1:125/25;

forw_cmd = joystick_command(model_resp_index(plot_cnt), 1:2:end)/100;
left_cmd = joystick_command(model_resp_index(plot_cnt), 2:2:end)/100;
left_speed = ref_resp_robot(plot_cnt, 1:2:end);
right_speed = ref_resp_robot(plot_cnt, 2:2:end);

diff_phys_l = ref_resp_phys(model_resp_index(plot_cnt), 1:2:250) - left_speed;
diff_phys_r = ref_resp_phys(model_resp_index(plot_cnt), 2:2:250) - right_speed;
diff_lwpr_l = ref_resp_lwpr(model_resp_index(plot_cnt), 1:2:250) - left_speed;
diff_lwpr_r = ref_resp_lwpr(model_resp_index(plot_cnt), 2:2:250) - right_speed;
diff_gpr_l = ref_resp_gpr(model_resp_index(plot_cnt), 1:2:250) - left_speed;
diff_gpr_r = ref_resp_gpr(model_resp_index(plot_cnt), 2:2:250) - right_speed;
diff_ffnn_l = ref_resp_ffnn(model_resp_index(plot_cnt), 1:2:250) - left_speed;
diff_ffnn_r = ref_resp_ffnn(model_resp_index(plot_cnt), 2:2:250) - right_speed;
diff_ddcnn_l = ref_resp_ddcnn(model_resp_index(plot_cnt), 1:2:250) - left_speed;
diff_ddcnn_r = ref_resp_ddcnn(model_resp_index(plot_cnt), 2:2:250) - right_speed;

max_y_left = max([max(forw_cmd), max(left_cmd), max(left_speed), ...
                  max(ref_resp_phys(model_resp_index(plot_cnt), 1:2:250)), ...
                  max(ref_resp_lwpr(model_resp_index(plot_cnt), 1:2:250)), ...
                  max(ref_resp_gpr(model_resp_index(plot_cnt), 1:2:250)), ...
                  max(ref_resp_ffnn(model_resp_index(plot_cnt), 1:2:250)), ...
                  max(ref_resp_ddcnn(model_resp_index(plot_cnt), 1:2:250)), ...
                  max(diff_phys_l), max(diff_lwpr_l), max(diff_gpr_l), ...
                  max(diff_ffnn_l), max(diff_ddcnn_l)]);
min_y_left = min([min(forw_cmd), min(left_cmd), min(left_speed), ...
                  min(ref_resp_phys(model_resp_index(plot_cnt), 1:2:250)), ...
                  min(ref_resp_lwpr(model_resp_index(plot_cnt), 1:2:250)), ...
                  min(ref_resp_gpr(model_resp_index(plot_cnt), 1:2:250)), ...
                  min(ref_resp_ffnn(model_resp_index(plot_cnt), 1:2:250)), ...
                  min(ref_resp_ddcnn(model_resp_index(plot_cnt), 1:2:250)), ...
                  min(diff_phys_l), min(diff_lwpr_l), min(diff_gpr_l), ...
                  min(diff_ffnn_l), min(diff_ddcnn_l)]);
max_y_right = max([max(forw_cmd), max(left_cmd), max(right_speed), ...
                   max(ref_resp_phys(model_resp_index(plot_cnt), 2:2:250)), ...
                   max(ref_resp_lwpr(model_resp_index(plot_cnt), 2:2:250)), ...
                   max(ref_resp_gpr(model_resp_index(plot_cnt), 2:2:250)), ...
                   max(ref_resp_ffnn(model_resp_index(plot_cnt), 2:2:250)), ...
                   max(ref_resp_ddcnn(model_resp_index(plot_cnt), 2:2:250)), ...
                   max(diff_phys_r), max(diff_lwpr_r), max(diff_gpr_r), ...
                   max(diff_ffnn_r), max(diff_ddcnn_r)]);
min_y_right = min([min(forw_cmd), min(left_cmd), min(right_speed), ...
                   min(ref_resp_phys(model_resp_index(plot_cnt), 2:2:250)), ...
                   min(ref_resp_lwpr(model_resp_index(plot_cnt), 2:2:250)), ...
                   min(ref_resp_gpr(model_resp_index(plot_cnt), 2:2:250)), ...
                   min(ref_resp_ffnn(model_resp_index(plot_cnt), 2:2:250)), ...
                   min(ref_resp_ddcnn(model_resp_index(plot_cnt), 2:2:250)), ...
                   min(diff_phys_r), min(diff_lwpr_r), min(diff_gpr_r), ...
                   min(diff_ffnn_r), min(diff_ddcnn_r)]);

for cnt = 1:5
    if cnt==1
        model_output_l = ref_resp_phys(model_resp_index(plot_cnt), 1:2:250);
        model_output_r = ref_resp_phys(model_resp_index(plot_cnt), 2:2:250);
    elseif cnt==2
        model_output_l = ref_resp_lwpr(model_resp_index(plot_cnt), 1:2:250);
        model_output_r = ref_resp_lwpr(model_resp_index(plot_cnt), 2:2:250);
    elseif cnt==3
        model_output_l = ref_resp_gpr(model_resp_index(plot_cnt), 1:2:250);
        model_output_r = ref_resp_gpr(model_resp_index(plot_cnt), 2:2:250);
    elseif cnt==4
        model_output_l = ref_resp_ffnn(model_resp_index(plot_cnt), 1:2:250);
        model_output_r = ref_resp_ffnn(model_resp_index(plot_cnt), 2:2:250);
    else
        model_output_l = ref_resp_ddcnn(model_resp_index(plot_cnt), 1:2:250);
        model_output_r = ref_resp_ddcnn(model_resp_index(plot_cnt), 2:2:250);
    end
    diff_l = model_output_l - left_speed;
    diff_r = model_output_r - right_speed;
    
    fig = figure();
    subplot(2,1,1);
    hold on;
    plot(0:0.04:4.96, model_output_l, 'b-', 'LineWidth', line_width);
    plot(0:0.04:4.96, left_speed, 'm--', 'LineWidth', line_width);
    plot(0:0.04:4.96, diff_l, 'ro-', 'LineWidth', line_width, 'MarkerSize', mark_size,'MarkerFaceColor','red');
%     stairs(0:0.04:4.96, forw_cmd, 'k:', 'LineWidth', line_width);
%     stairs(0:0.04:4.96, left_cmd, 'k-.', 'LineWidth', line_width);
    hold off;
    grid on;    grid minor;
%     if max_y_left - min_y_left < 0.1
%         ylim([-0.2 0.2]);
%     else
%         ylim([min_y_left-0.15 max_y_left+0.15]);
%     end
    ylim([-0.5 1]);
    ylabel('Left Wheel Speed(m/s)', 'FontSize', lab_font_size);
    set(gca, 'FontSize', leg_font_size-4);
    
    subplot(2,1,2);
    hold on;
    p1 = plot(0:0.04:4.96, model_output_r, 'b-', 'LineWidth', line_width);
    p2 = plot(0:0.04:4.96, right_speed, 'm--', 'LineWidth', line_width);
    p3 = plot(0:0.04:4.96, diff_r, 'ro-', 'LineWidth', line_width, 'MarkerSize', mark_size,'MarkerFaceColor','red');
%     p4 = stairs(0:0.04:4.96, forw_cmd, 'k:', 'LineWidth', line_width);
%     p5 = stairs(0:0.04:4.96, left_cmd, 'k-.', 'LineWidth', line_width);
    hold off;
    grid on;    grid minor;
%     if max_y_right - min_y_right < 0.1
%         ylim([-0.2 0.2]);
%     else
%         ylim([min_y_right-0.15 max_y_right+0.15]);
%     end
    ylim([-0.5 1]);
    xlabel('Length of Simulation(sec)', 'FontSize', lab_font_size);
    ylabel('Right Wheel Speed(m/s)', 'FontSize', lab_font_size);
%     leg = legend([p1, p2, p3, p4, p5], {'model prediction', 'measured speed', 'difference', 'joystick forward cmd', 'joystick left cmd'});
    leg = legend([p1, p2, p3], {'model prediction', 'measured speed', 'difference'});
%     set(leg, 'Position', [0.71 0.34 0.35 0.35]);
    set(leg, 'Position', [0.62 0.35 0.35 0.33]);
    set(gca, 'FontSize', leg_font_size-4);
    set(leg, 'FontSize', leg_font_size);
end

%% Error Distribution of 5 seconds LTS error of models
close all; clear; clc;

% robot_type = 'vulcan';
robot_type = 'magicbot';

if strcmp(robot_type, 'vulcan')
    load('./GPR/test_data_vulcan(hw4_1).mat', 'test_data_y');
    num_test_data = size(test_data_y, 1);
elseif  strcmp(robot_type, 'magicbot')
    load('./GPR/test_data_magicbot(hw5_1).mat', 'test_data_y');
    num_test_data = size(test_data_y, 1);
end
% divisor = 1;
divisor = 2*125;

for mode=1:6
    if mode == 1 && strcmp(robot_type, 'vulcan')
        load('./result_files/phys_vulcan_para7.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_phys = difference_btw_prediction_data;
        error_per_case_phys = sqrt(sum(difference_btw_prediction_data_phys.^2, 2)/divisor);
    elseif mode == 2 && strcmp(robot_type, 'vulcan')
        load('./result_files/lwpr_vulcan.mat', 'model_output_on_test_data');
        difference_btw_prediction_data_lwpr = model_output_on_test_data - test_data_y;
        error_per_case_lwpr = sqrt(sum(difference_btw_prediction_data_lwpr.^2, 2)/divisor);
    elseif mode == 2 && strcmp(robot_type, 'magicbot')
        load('./result_files/lwpr_magicbot.mat', 'model_output_on_test_data');
        difference_btw_prediction_data_lwpr = model_output_on_test_data - test_data_y;
        error_per_case_lwpr = sqrt(sum(difference_btw_prediction_data_lwpr.^2, 2)/divisor);
    elseif mode == 3 && strcmp(robot_type, 'vulcan')
        load('./result_files/gpr_vulcan_sample3000_hw5.mat', 'pred_on_test_data');
        difference_btw_prediction_data_gpr = pred_on_test_data - test_data_y;
        error_per_case_gpr = sqrt(sum(difference_btw_prediction_data_gpr.^2, 2)/divisor);
    elseif mode == 3 && strcmp(robot_type, 'magicbot')
        load('./result_files/gpr_magicbot_sample4000_hw5.mat', 'pred_on_test_data');
        difference_btw_prediction_data_gpr = pred_on_test_data - test_data_y;
        error_per_case_gpr = sqrt(sum(difference_btw_prediction_data_gpr.^2, 2)/divisor);
    elseif mode == 4 && strcmp(robot_type, 'vulcan')
        load('./result_files/ffnn_vulcan_hid128.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_ffnn = difference_btw_prediction_data;
        error_per_case_ffnn = sqrt(sum(difference_btw_prediction_data_ffnn.^2, 2)/divisor);
    elseif mode == 4 && strcmp(robot_type, 'magicbot')
        load('./result_files/ffnn_magicbot_hid96.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_ffnn = difference_btw_prediction_data;
        error_per_case_ffnn = sqrt(sum(difference_btw_prediction_data_ffnn.^2, 2)/divisor);
    elseif mode == 5 && strcmp(robot_type, 'vulcan')
        load('./result_files/cnn_vulcan.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_cnn = difference_btw_prediction_data;
        error_per_case_cnn = sqrt(sum(difference_btw_prediction_data_cnn.^2, 2)/divisor);
    elseif mode == 5 && strcmp(robot_type, 'magicbot')
        load('./result_files/cnn_magicbot.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_cnn = difference_btw_prediction_data;
        error_per_case_cnn = sqrt(sum(difference_btw_prediction_data_cnn.^2, 2)/divisor);
    elseif mode == 6 && strcmp(robot_type, 'vulcan')
        load('./result_files/rnn_vulcan.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_rnn = difference_btw_prediction_data;
        error_per_case_rnn = sqrt(sum(difference_btw_prediction_data_rnn.^2, 2)/divisor);
    elseif mode == 6 && strcmp(robot_type, 'magicbot')
        load('./result_files/rnn_magicbot.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_rnn = difference_btw_prediction_data;
        error_per_case_rnn = sqrt(sum(difference_btw_prediction_data_rnn.^2, 2)/divisor);
    end
end


plot_box_whisker = true;
plot_title = false;
lab_font_size = 22;
title_font_size = 24;

if plot_box_whisker
    ind = [int32(num_test_data/10); int32(num_test_data/4); ...
           int32(num_test_data/2); int32(num_test_data*3/4); int32(num_test_data*0.9)];
    error_sort_rnn = sort(error_per_case_rnn, 'ascend');
    error_sort_cnn = sort(error_per_case_cnn, 'ascend');
    error_sort_ffnn = sort(error_per_case_ffnn, 'ascend');
    error_sort_gpr = sort(error_per_case_gpr, 'ascend');
    error_sort_lwpr = sort(error_per_case_lwpr, 'ascend');
    if strcmp(robot_type, 'vulcan')
        error_sort_phys = sort(error_per_case_phys, 'ascend');
    end
    b_w_error_pos = [];
    for cnt = 1:size(ind, 1)
%         b_w_error_pos = [b_w_error_pos;error_sort_cnn(ind(cnt)), error_sort_ffnn(ind(cnt)), ...
%                          error_sort_phys(ind(cnt)), error_sort_gpr(ind(cnt)), ...
%                          error_sort_lwpr(ind(cnt))];
        if strcmp(robot_type, 'vulcan')
            b_w_error_pos = [b_w_error_pos;error_sort_rnn(ind(cnt)), error_sort_cnn(ind(cnt)), ...
                             error_sort_ffnn(ind(cnt)), error_sort_lwpr(ind(cnt)), ...
                             error_sort_gpr(ind(cnt)), error_sort_phys(ind(cnt))];
        elseif strcmp(robot_type, 'magicbot')
            b_w_error_pos = [b_w_error_pos;error_sort_rnn(ind(cnt)), error_sort_cnn(ind(cnt)), ...
                             error_sort_ffnn(ind(cnt)), error_sort_lwpr(ind(cnt)), ...
                             error_sort_gpr(ind(cnt))];
        end
    end
%     error_models = [error_per_case_cnn, error_per_case_ffnn, error_per_case_phys, ...
%                     error_per_case_gpr, error_per_case_lwpr];
    if strcmp(robot_type, 'vulcan')
        error_models = [error_per_case_rnn, error_per_case_cnn, error_per_case_ffnn, ...
                        error_per_case_lwpr, error_per_case_gpr, error_per_case_phys];
    elseif strcmp(robot_type, 'magicbot')
        error_models = [error_per_case_rnn, error_per_case_cnn, error_per_case_ffnn, ...
                        error_per_case_lwpr, error_per_case_gpr];
    end
    b_w_error_pos = b_w_error_pos * 100;
    error_models = error_models * 100;
    
    fig = figure();
    h = boxplot(error_models, 'Symbol', '', 'Widths', 0.4, 'Orientation', 'horizontal');
    set(h(:,:), 'linewidth', 3);
    low_whisker = findobj(h, 'tag', 'Lower Whisker');
    set(low_whisker, {'Xdata'}, num2cell(b_w_error_pos(1:2,:),1)');
    upper_whisker = findobj(h, 'tag', 'Upper Whisker');
    set(upper_whisker, {'Xdata'}, num2cell(b_w_error_pos(end-1:end,:),1)');
    set(h(3,:),{'Xdata'}, num2cell(b_w_error_pos([end end],:),1)');
    set(h(4,:),{'Xdata'}, num2cell(b_w_error_pos([1 1],:),1)');
%     xlim([0, 3]);
%     xlim([0, 18]);
%     xlim([0, 2.5]);
    xlim([0, 14]);
    if strcmp(robot_type, 'vulcan')
        set(gca,'YTick',1:6);
        set(gca, 'YTickLabel', {'DD-RNN', 'DD-CNN', 'FFNN', 'LWPR', 'GPR', 'Physics'});
    elseif strcmp(robot_type, 'magicbot')
        set(gca,'YTick',1:5);
        set(gca, 'YTickLabel', {'DD-RNN', 'DD-CNN', 'FFNN', 'LWPR', 'GPR'});
    end
    set(gca, 'FontSize', lab_font_size);
    xlabel('RMS error (cm/s)', 'FontSize', lab_font_size);
    if plot_title
        title('Error Distribution of Models', 'FontSize', title_font_size);
    end
    
else
    x_lim_max = 10;
    y_lim_max = 10000;

    max_ub = max([max(error_per_case_phys), max(error_per_case_lwpr), max(error_per_case_gpr), max(error_per_case_ffnn), max(error_per_case_cnn)]);
    min_ub = min([max(error_per_case_phys), max(error_per_case_lwpr), max(error_per_case_gpr), max(error_per_case_ffnn), max(error_per_case_cnn)]);
    % Ctrs = [min_ub/50:min_ub/25:max_ub-min_ub/50];
    Ctrs = [min_ub/50:min_ub/25:x_lim_max-min_ub/50];
    Xcts_phys = hist(error_per_case_phys, Ctrs);
    Xcts_lwpr = hist(error_per_case_lwpr, Ctrs);
    Xcts_gpr = hist(error_per_case_gpr, Ctrs);
    Xcts_fc = hist(error_per_case_ffnn, Ctrs);
    Xcts_cnn = hist(error_per_case_cnn, Ctrs);

    r_ind_phys = zeros(3,1); sum_of_cnt = 0;
    ref = 44661/4; ind_cnt = 1;
    for cnt = 1:length(Xcts_phys)
        if (sum_of_cnt < ref*ind_cnt) && (sum_of_cnt+Xcts_phys(cnt) >= ref*ind_cnt)
            r_ind_phys(ind_cnt) = cnt;
            ind_cnt = ind_cnt + 1;
        end

        if ind_cnt >= 4
            break;
        end
        sum_of_cnt = sum_of_cnt + Xcts_phys(cnt);
    end

    r_ind_lwpr = zeros(3,1); sum_of_cnt = 0;
    ref = 44661/4; ind_cnt = 1;
    for cnt = 1:length(Xcts_lwpr)
        if (sum_of_cnt < ref*ind_cnt) && (sum_of_cnt+Xcts_lwpr(cnt) >= ref*ind_cnt)
            r_ind_lwpr(ind_cnt) = cnt;
            ind_cnt = ind_cnt + 1;
        end

        if ind_cnt >= 4
            break;
        end
        sum_of_cnt = sum_of_cnt + Xcts_lwpr(cnt);
    end

    r_ind_gpr = zeros(3,1); sum_of_cnt = 0;
    ref = 44661/4; ind_cnt = 1;
    for cnt = 1:length(Xcts_gpr)
        if (sum_of_cnt < ref*ind_cnt) && (sum_of_cnt+Xcts_gpr(cnt) >= ref*ind_cnt)
            r_ind_gpr(ind_cnt) = cnt;
            ind_cnt = ind_cnt + 1;
        end

        if ind_cnt >= 4
            break;
        end
        sum_of_cnt = sum_of_cnt + Xcts_gpr(cnt);
    end

    r_ind_fc = zeros(3,1); sum_of_cnt = 0;
    ref = 44661/4; ind_cnt = 1;
    for cnt = 1:length(Xcts_fc)
        if (sum_of_cnt < ref*ind_cnt) && (sum_of_cnt+Xcts_fc(cnt) >= ref*ind_cnt)
            r_ind_fc(ind_cnt) = cnt;
            ind_cnt = ind_cnt + 1;
        end

        if ind_cnt >= 4
            break;
        end
        sum_of_cnt = sum_of_cnt + Xcts_fc(cnt);
    end

    r_ind_cnn = zeros(3,1); sum_of_cnt = 0;
    ref = 44661/4; ind_cnt = 1;
    for cnt = 1:length(Xcts_cnn)
        if (sum_of_cnt < ref*ind_cnt) && (sum_of_cnt+Xcts_cnn(cnt) >= ref*ind_cnt)
            r_ind_cnn(ind_cnt) = cnt;
            ind_cnt = ind_cnt + 1;
        end

        if ind_cnt >= 4
            break;
        end
        sum_of_cnt = sum_of_cnt + Xcts_cnn(cnt);
    end

    fig = figure();
    hold on;
    bar(Ctrs, Xcts_phys, 'b');
    bar(Ctrs(r_ind_phys(1)), Xcts_phys(r_ind_phys(1)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_phys(2)), Xcts_phys(r_ind_phys(2)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_phys(3)), Xcts_phys(r_ind_phys(3)), min_ub/25, 'FaceColor', 'r');
    hold off;
    ylim([0 y_lim_max]);
    set(gca,'LooseInset',get(gca,'TightInset'))
    title('Error Distribution of 5 seconds LTS(Physics)', 'FontSize', 16);
    xlabel('RMS Error', 'FontSize', 13);

    fig2 = figure();
    hold on;
    bar(Ctrs, Xcts_lwpr, 'b');
    bar(Ctrs(r_ind_lwpr(1)), Xcts_lwpr(r_ind_lwpr(1)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_lwpr(2)), Xcts_lwpr(r_ind_lwpr(2)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_lwpr(3)), Xcts_lwpr(r_ind_lwpr(3)), min_ub/25, 'FaceColor', 'r');
    hold off;
    ylim([0 y_lim_max]);
    set(gca,'LooseInset',get(gca,'TightInset'))
    title('Error Distribution of 5 seconds LTS(LWPR)', 'FontSize', 16);
    xlabel('RMS Error', 'FontSize', 13);

    fig3 = figure();
    hold on;
    bar(Ctrs, Xcts_gpr, 'b');
    bar(Ctrs(r_ind_gpr(1)), Xcts_gpr(r_ind_gpr(1)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_gpr(2)), Xcts_gpr(r_ind_gpr(2)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_gpr(3)), Xcts_gpr(r_ind_gpr(3)), min_ub/25, 'FaceColor', 'r');
    hold off;
    ylim([0 y_lim_max]);
    set(gca,'LooseInset',get(gca,'TightInset'))
    title('Error Distribution of 5 seconds LTS(GPR)', 'FontSize', 16);
    xlabel('RMS Error', 'FontSize', 13);

    fig4 = figure();
    %h2 = histogram(error_per_case_lts);
    hold on;
    bar(Ctrs, Xcts_fc, 'b');
    bar(Ctrs(r_ind_fc(1)), Xcts_fc(r_ind_fc(1)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_fc(2)), Xcts_fc(r_ind_fc(2)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_fc(3)), Xcts_fc(r_ind_fc(3)), min_ub/25, 'FaceColor', 'r');
    hold off;
    ylim([0 y_lim_max]);
    set(gca,'LooseInset',get(gca,'TightInset'))
    title('Error Distribution of 5 seconds LTS(FC)', 'FontSize', 16);
    xlabel('RMS Error', 'FontSize', 13);

    fig5 = figure();
    %h2 = histogram(error_per_case_lts);
    hold on;
    bar(Ctrs, Xcts_cnn, 'b');
    bar(Ctrs(r_ind_cnn(1)), Xcts_cnn(r_ind_cnn(1)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_cnn(2)), Xcts_cnn(r_ind_cnn(2)), min_ub/25, 'FaceColor', 'r');
    bar(Ctrs(r_ind_cnn(3)), Xcts_cnn(r_ind_cnn(3)), min_ub/25, 'FaceColor', 'r');
    hold off;
    ylim([0 y_lim_max]);
    set(gca,'LooseInset',get(gca,'TightInset'))
    title('Error Distribution of 5 seconds LTS(CNN)', 'FontSize', 16);
    xlabel('RMS Error', 'FontSize', 13);
end




%% pairwise t-test
close all; clear; clc;

robot_type = 'vulcan';
% robot_type = 'magicbot';

if strcmp(robot_type, 'vulcan')
    load('./GPR/test_data_vulcan(hw4_1).mat', 'test_data_y');
    num_test_data = size(test_data_y, 1);
elseif  strcmp(robot_type, 'magicbot')
    load('./GPR/test_data_magicbot(hw5_1).mat', 'test_data_y');
    num_test_data = size(test_data_y, 1);
end
divisor = 1;
% divisor = 2*125;

for mode=1:6
    if mode == 1 && strcmp(robot_type, 'vulcan')
        load('./result_files/phys_vulcan_para7.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_phys = difference_btw_prediction_data;
        error_per_case_phys = sqrt(sum(difference_btw_prediction_data_phys.^2, 2)/divisor);
    elseif mode == 2 && strcmp(robot_type, 'vulcan')
        load('./result_files/lwpr_vulcan.mat', 'model_output_on_test_data');
        difference_btw_prediction_data_lwpr = model_output_on_test_data - test_data_y;
        error_per_case_lwpr = sqrt(sum(difference_btw_prediction_data_lwpr.^2, 2)/divisor);
    elseif mode == 2 && strcmp(robot_type, 'magicbot')
        load('./result_files/lwpr_magicbot.mat', 'model_output_on_test_data');
        difference_btw_prediction_data_lwpr = model_output_on_test_data - test_data_y;
        error_per_case_lwpr = sqrt(sum(difference_btw_prediction_data_lwpr.^2, 2)/divisor);
    elseif mode == 3 && strcmp(robot_type, 'vulcan')
        load('./result_files/gpr_vulcan_sample3000_hw5.mat', 'pred_on_test_data');
        difference_btw_prediction_data_gpr = pred_on_test_data - test_data_y;
        error_per_case_gpr = sqrt(sum(difference_btw_prediction_data_gpr.^2, 2)/divisor);
    elseif mode == 3 && strcmp(robot_type, 'magicbot')
        load('./result_files/gpr_magicbot_sample4000_hw5.mat', 'pred_on_test_data');
        difference_btw_prediction_data_gpr = pred_on_test_data - test_data_y;
        error_per_case_gpr = sqrt(sum(difference_btw_prediction_data_gpr.^2, 2)/divisor);
    elseif mode == 4 && strcmp(robot_type, 'vulcan')
        load('./result_files/ffnn_vulcan_hid128.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_ffnn = difference_btw_prediction_data;
        error_per_case_ffnn = sqrt(sum(difference_btw_prediction_data_ffnn.^2, 2)/divisor);
    elseif mode == 4 && strcmp(robot_type, 'magicbot')
        load('./result_files/ffnn_magicbot_hid96.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_ffnn = difference_btw_prediction_data;
        error_per_case_ffnn = sqrt(sum(difference_btw_prediction_data_ffnn.^2, 2)/divisor);
    elseif mode == 5 && strcmp(robot_type, 'vulcan')
        load('./result_files/cnn_vulcan.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_cnn = difference_btw_prediction_data;
        error_per_case_cnn = sqrt(sum(difference_btw_prediction_data_cnn.^2, 2)/divisor);
    elseif mode == 5 && strcmp(robot_type, 'magicbot')
        load('./result_files/cnn_magicbot.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_cnn = difference_btw_prediction_data;
        error_per_case_cnn = sqrt(sum(difference_btw_prediction_data_cnn.^2, 2)/divisor);
    elseif mode == 6 && strcmp(robot_type, 'vulcan')
        load('./result_files/rnn_vulcan.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_rnn = difference_btw_prediction_data;
        error_per_case_rnn = sqrt(sum(difference_btw_prediction_data_rnn.^2, 2)/divisor);
    elseif mode == 6 && strcmp(robot_type, 'magicbot')
        load('./result_files/rnn_magicbot.mat', 'difference_btw_prediction_data');
        difference_btw_prediction_data_rnn = difference_btw_prediction_data;
        error_per_case_rnn = sqrt(sum(difference_btw_prediction_data_rnn.^2, 2)/divisor);
    end
end