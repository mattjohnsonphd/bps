function [q, nSamples] = bps_qSampler(x,z,beta,Sigma,r1,r2,d,f0,u)
% function [q, nSamples] = bps_qSampler(x,z,mu,Sigma,lambda)
%
% Samples (q|x,y,z,beta,Sigma) in bps Gibbs sampler.
% x = J-vector of samples from p(x|q,y,z,mu,Sigma)
% z = component of mixture, sampled from p(z|q,y,mu,Sigma)
% mu = J-vector of agent weight locations
% Sigma = JxJ covariance matrix of agent dependencies
% u = (1+J)-vector of Dirichlet parameters (prior for q_{0:J})

nSamples = 0;
    
if z == 0
    
    mu = f0+beta;
    J = length(mu);
    
    alpha = zeros(J,1);
    for j = 1:J
        phi=Sigma(:,j); phi(j)=[];
        Sigmamj=Sigma; Sigmamj(j,:)=[]; Sigmamj(:,j)=[];
        gamma = Sigmamj\eye(J-1)*phi;
        delta = Sigma(j,j)-phi'*gamma;
        
        xmj = x; xmj(j) = [];
        mumj = mu; mumj(j) = [];
        ej = x(j) - mu(j) - gamma' * (xmj-mumj);
        
        alpha(j) = exp(-ej*ej/(2*r1*delta))-d*exp(-ej*ej/(2*r2*delta));
    end
    
    accept = 0;
    while accept == 0
        nSamples = nSamples + 1;
        q = dirrnd(u,1); % row vector, dirrnd() has no error checking
        
        if rand > q(2:end) * alpha % or rand < 1-q(2:end)*alpha
            q=q'; % column vector
            accept = 1;
        end
    end
    
else % z == j > 0
    
    % conjugate update to Dirichlet(lambda)
    u(1+z) = 1 + u(1+z);
    q = dirrnd(u,1)'; % column vector, dirrnd() has no error checking
    
end