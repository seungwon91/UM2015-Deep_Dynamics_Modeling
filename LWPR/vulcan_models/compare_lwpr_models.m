%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.08.30 Updated
% Program to compare LWPR-based model with diverse model setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;  clc;    close all;

plot_title = false;
lab_font_size = 22;
title_font_size = 22;
num_test_data = 44661;

load('test_data_set(hw4_1).mat', 'test_data_y');

folder_list = {};
folder_list{1} = 'D04F_20_hw3';
folder_list{2} = 'D04F_30_hw3';
folder_list{3} = 'D04F_40_hw3';
folder_list{4} = 'D04F_50_hw3';
folder_list{5} = 'D05F_20_hw3';
folder_list{6} = 'D05F_30_hw3';
folder_list{7} = 'D05F_40_hw3';
folder_list{8} = 'D06F_20_hw3';
folder_list{9} = 'D06F_30_hw3';
folder_list{10} = 'D06F_40_hw3';
folder_list{11} = 'D06F_50_hw3';
folder_list{12} = 'D10F_40_hw3';
folder_list{13} = 'D025F_40_hw3';
folder_list{14} = '04F_10_hw4';
folder_list{15} = '04F_20_hw4';
folder_list{16} = '04F_30_hw4';
folder_list{17} = '04F_40_hw4';
folder_list{18} = '05F_10_hw4';
folder_list{19} = '05F_20_hw4';
folder_list{20} = '05F_30_hw4';
folder_list{21} = '05F_40_hw4';
folder_list{22} = '06F_20_hw4';
folder_list{23} = '06F_30_hw4';
folder_list{24} = '06F_40_hw4';

% Print error/computation time
total_test_error = [];  total_test_time = [];
error_models = [];  b_w_error_pos = [];
ind = [int32(num_test_data/10); int32(num_test_data/4); ...
       int32(num_test_data/2); int32(num_test_data*3/4); int32(num_test_data*0.9)];
for cnt=1:length(folder_list)
    load(sprintf('./%s/LWPR_1D_model_test_result.mat', folder_list{cnt}), 'test_error', 'test_time', 'model_output_on_test_data');
    fprintf('\tModel %s\n', folder_list{cnt});
    fprintf('test MAE error : %f\n', test_error(1));
    fprintf('test RMS error : %f\n', test_error(2));
    fprintf('test Maximum error : %f\n', test_error(3));
    fprintf('test time consumption per trajectory : %f\n\n', test_time/44661);
    
    diff = model_output_on_test_data - test_data_y;
    error_per_case = sort(sqrt(sum(diff.^2, 2)), 'ascend');
    error_models = [error_per_case, error_models];
    b_w_error_pos = [[error_per_case(ind(1)); error_per_case(ind(2)); error_per_case(ind(3)); error_per_case(ind(4)); error_per_case(ind(5))], ...
                     b_w_error_pos];

    total_test_error = [total_test_error; test_error, max(error_per_case)];
    total_test_time = [total_test_time; test_time/44661];
end

save('LWPR_comparison_result.mat', 'folder_list', 'total_test_error', 'total_test_time', 'error_models', 'b_w_error_pos');

fig1 = figure();    clf(fig1);
hold on;
plot(1:24, total_test_error(:, 1)/10, 'rx-', 'LineWidth', 2);
plot(1:24, total_test_error(:, 2), 'bo-', 'LineWidth', 2);
plot(1:24, total_test_error(:, 3), 'g^-', 'LineWidth', 2);
plot(1:24, total_test_error(:, 4)/3, 'md-', 'LineWidth', 2);
plot(1:24, total_test_time*10, 'kp-', 'LineWidth', 2);
hold off;
legend({'MAE/10', 'RMS', 'L infty', 'Max RMS traj/3', 'Comp Time*10'}, 'FontSize', lab_font_size);
ylim([0 6]);


fig2 = figure();    clf(fig2);
h = boxplot(error_models, 'Symbol', '', 'Widths', 0.2, 'Orientation', 'horizontal');
set(h(:,:), 'linewidth', 3);
low_whisker = findobj(h, 'tag', 'Lower Whisker');
set(low_whisker, {'Xdata'}, num2cell(b_w_error_pos(1:2,:),1)');
upper_whisker = findobj(h, 'tag', 'Upper Whisker');
set(upper_whisker, {'Xdata'}, num2cell(b_w_error_pos(end-1:end,:),1)');
set(h(3,:),{'Xdata'}, num2cell(b_w_error_pos([end end],:),1)');
set(h(4,:),{'Xdata'}, num2cell(b_w_error_pos([1 1],:),1)');
xlim([0, 7]);
set(gca,'YTick',1:24);
set(gca, 'YTickLabel', 24:-1:1, 'FontSize', lab_font_size);
%set(gca, 'YTickLabel', {'DeepDynamics-CNN', 'FeedForwardNet', 'LWPR', 'Physics'}, 'FontSize', lab_font_size);
xlabel('RMS error', 'FontSize', lab_font_size);
if plot_title
    title('Error Distribution of Models', 'FontSize', title_font_size);
end