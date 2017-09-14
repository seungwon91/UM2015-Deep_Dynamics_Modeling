%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2015.08.30 updated
% To change gp function format for slice sampling
% if one of hyperparameters is smaller than 0, probability = -inf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [result,result2] = gp_feval_transform(hyper, meanfunc, covfunc, likfunc, train_input, train_output)
	if nargout <2
        if any(hyper<0)
            result = -inf;
        else
            num = length(hyper);
            hyp = [];
            hyp.mean = [];
            hyp.cov = log(hyper(1:num-1));
            hyp.lik = log(hyper(num));

            result = -gp(hyp, @infExact, meanfunc, covfunc, likfunc, train_input, train_output) ...
                    + priorGaussMulti(ones(num,1),eye(num),hyper');
        end
    else
        num = length(hyper);
        hyp = [];
        hyp.mean = [];
        hyp.cov = log(hyper(1:num-1));
        hyp.lik = log(hyper(num));
            
        [result, tmp] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, train_input, train_output);
        result2 = [tmp.cov;tmp.lik];
    end
end