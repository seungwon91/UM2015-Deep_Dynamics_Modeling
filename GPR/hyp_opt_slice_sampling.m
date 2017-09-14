%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to optimize hyperparameter using slice sampling
% hyperparameter and slice sampling settings are stored in gp model
% [input]
% gpmodel : Gaussian Process model
% [output]
% new_hyp : column vector of newly chosen hyperparameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_hyp = hyp_opt_slice_sampling(gpmodel)
    hyps = slicesample(gpmodel.hyp_para', gpmodel.slicesample.num_hyp, 'logpdf', ...
                       gpmodel.log_marg_lik, 'burnin', gpmodel.slicesample.burnin);

    new_hyp = [];
    for a=1:size(hyps,2)
        [p_kd, theta] = ksdensity(hyps(:,a));
        [~,mx_ind] = max(p_kd);
        new_hyp = [new_hyp;theta(mx_ind)];
    end
end