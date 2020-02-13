function u = bps_dirVB(Q,u0,tol)
% bps_dirVB approximates the distribution of Q with a Dirichlet
% distribution by minimizing the KL divergence of the approximation from
% the Monte Carlo samples.
% Q is a J x nmc matrix of probability vectors
% u0 is an initial J vector of Dirichlet parameters
% tol is a tolerance for use in Newton-Raphson

J = size(Q,1);

% possible to get something that's zero to machine level precision
% causes a problem with logs in the next step, so just replace with eps
Q(Q==0) = eps;

Elogq = mean(log(Q),2);
u = u0(:); % make sure it's a column vector

% Newton Raphson to solve for optimal Dirichlet parameters
f = psi(u) - psi(sum(u)) - Elogq;
while any(abs(f)>tol)
    
    Jac = -psi(1,sum(u)) * ones(J) + diag(psi(1,u));
    u = max(u- Jac\eye(J) * f, eps);
    f = psi(u) - psi(sum(u)) - Elogq;
    
end