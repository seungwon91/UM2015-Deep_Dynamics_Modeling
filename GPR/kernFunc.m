% GP radial basis exponential kernel
% [input]
% x1, x2 : scalar or vector to compute kernel value or matrix
% h : hyperparameter
% output k(x1,x2) = h(D+1) * exp(-( (x1-x2)/hyp(1:D) )^2 /2)
%                   + hyp(D+2) *(x1==x2)
%               where D : # of feature(each data : Dx1 vector)

function K = kernFunc(x1,x2,h)
	tmp = sq_dist(bsxfun(@rdivide, x1, h(1:end-2)')', bsxfun(@rdivide, x2, h(1:end-2)')');
    K = h(end-1) * exp(-tmp/2);
end