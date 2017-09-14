% Generating Covariance Matrix & Cholesky Decomposed Matrix

function [K, L] = GP_KL_generator(gpmodel,trainx)

% Method of 'Efficient Optimization for Sparse Gaussian Process Regression'
%  - Yanshuai Cao, Marcus Brubaker and 2 people
%
%     % Kernel Mtx. whole data-by-inducing data
%     Knm = gpmodel.kernel(trainx(gpmodel.data_order_index,:), ...
%                          trainx(gpmodel.data_order_index(1:gpmodel.num_inducing),:), ...
%                          gpmodel.hyp_para);
%                      
% 	% Kernel Mtx. inducing data-by-inducing data
%     Kmm = gpmodel.kernel(trainx(gpmodel.data_order_index(1:gpmodel.num_inducing),:), ...
%                          trainx(gpmodel.data_order_index(1:gpmodel.num_inducing),:), ...
%                          gpmodel.hyp_para);
% 
% 	% Computing inverse of Kmm & L & K
%     L_tmp = jitChol(Kmm, 5, 'lower');       % jitChol : ./gp_cholqr-master/
%     
%     L = Knm * inv(L_tmp)';
%     
%     K = L * L';

    K = gpmodel.kernel(trainx, trainx, gpmodel.hyp_para);
%     K = gpmodel.kernel(trainx(gpmodel.data_order_index(1:gpmodel.num_inducing),:), ...
%                        trainx(gpmodel.data_order_index(1:gpmodel.num_inducing),:), ...
%                        gpmodel.hyp_para);
                     
	L = jitChol(K, 5, 'lower');         % jitChol : ./gp_cholqr-master/
                                        % method to avoid minor errors
                                        % during cholesky decomposition

end