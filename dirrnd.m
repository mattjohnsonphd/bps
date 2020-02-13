function r = dirrnd(alpha,n)
% function r = dirrnd(alpha,n)
%
% Samples n vectors r from the Dirichlet distribution with parameter vector
% alpha. Output is a matrix with n rows and length(alpha) columns.

p = length(alpha);

if size(alpha,1) > size(alpha,2)
    alpha = alpha';
end

alpha = repmat(alpha,n,1);
r = gamrnd(alpha,1);
r = r ./ repmat(sum(r,2),1,p);