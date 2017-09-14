%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to set hyperparameter with new values
% hyperparameter : [covariance hyps;amplitude of kernel;noise]
% setting gp model.hyp_para & .hyp_struct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gpmodel = hyp_setting(gpmodel, new_values)
    if isrow(new_values)
        gpmodel.hyp_para = new_values';
    else
        gpmodel.hyp_para = new_values;
    end
    
    gpmodel.hyp_struct.mean = [];
    gpmodel.hyp_struct.cov = log(gpmodel.hyp_para(1:end-1))';
    gpmodel.hyp_struct.lik = log(gpmodel.hyp_para(end));
end