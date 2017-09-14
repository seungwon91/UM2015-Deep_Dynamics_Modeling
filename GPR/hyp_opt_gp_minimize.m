%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to optimize hyperparameter using minimize.m of GPML
% hyperparameter and function to minimize is stored in gp model
% [input]
% gpmodel : Gaussian Process model
% [output]
% new_hyp : column vector of newly chosen hyperparameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_hyp = hyp_opt_gp_minimize(gpmodel)
    new_hyp = minimize(gpmodel.hyp_para, gpmodel.log_marg_lik,-20);
end