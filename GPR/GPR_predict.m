%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to estimate function's value at new input
% using modified version of Sparse Gaussian Process Regression
% [input]
% gpmodel : structure which contains properties like hyperparameter, model setting
% train_x & train_y : data used for training
% new_x : new x to compute estimates
% [output]
% new_y : estimated value of function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [new_y, new_y_std] = GPR_predict(gpmodel,train_x,train_y,new_x,mode)
    if mode == 1
        [~,~,new_y,new_y_std] = gp(gpmodel.hyp_struct, @infExact, gpmodel.meanfunc, ...
                           gpmodel.covfunc, gpmodel.likfunc, ...
                           train_x(gpmodel.data_index,:), ...
                           train_y(gpmodel.data_index,:), ...
                           new_x);
    else
        K_new_x_induc = gpmodel.kernel(new_x, train_x(gpmodel.data_index,:), gpmodel.hyp_para);
        K_new_x_new_x = gpmodel.kernel(new_x, new_x, gpmodel.hyp_para);
        if mode == 2
            new_y = K_new_x_induc * gpmodel.inv_L_appAndNoise_sqr_y;
            new_y_std = K_new_x_new_x - K_new_x_induc * gpmodel.inv_L_appAndNoise_sqr * K_new_x_induc';
        else
            tmp_invL_K = gpmodel.L_appAndNoise \ (K_new_x_induc');

            % new estimate f* given inducing x, y and new data x*
            new_y = tmp_invL_K' * (gpmodel.L_appAndNoise \ train_y(gpmodel.data_index,:));
            new_y_std = K_new_x_new_x - tmp_invL_K' * tmp_invL_K;
        end
    end
end