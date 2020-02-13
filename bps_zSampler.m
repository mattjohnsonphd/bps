function z = bps_zSampler(q,y,beta,Sigma,r1,r2,d,f0,pi0y,f,s,k,nmc)

J = length(beta);

% Step 1: P(z=j|beta,Sigma)
pz = zeros(1+J,1);
F = repmat(f,1,nmc);
K = repmat(k,1,nmc);
X = F + diag(s) * trnd(K); % nmc samples from h(x)

alpha = zeros(J,1);
for j = 1:J
    phi=Sigma(:,j); phi(j)=[];
    Sigmamj=Sigma; Sigmamj(j,:)=[]; Sigmamj(:,j)=[];
    gamma=Sigmamj\eye(J-1)*phi;
    delta=Sigma(j,j)-phi'*gamma;
    
    betaj=beta(j); betamj=beta; betamj(j)=[];
        
    xj=X(j,:); xmj=X; xmj(j,:)=[];

    ej = xj - f0 - betaj - gamma' * (xmj-f0-repmat(betamj,1,nmc)); 
    
    alpha(j) = mean(exp(-ej.*ej/(2*r1*delta))-d*exp(-ej.*ej/(2*r2*delta)));
end
pz(2:1+J) = q(2:1+J) .* alpha;
pz(1) = 1 - sum(pz);

% Step 2: p(y|z=j,beta,Sigma)
py = zeros(1+J,1);
py(1) = pi0y;
for j = 1:J
    % hj(y+betaj)
    py(1+j) = tpdf((y+beta(j)-f(j))/sqrt(s(j)),k(j))/sqrt(s(j));
end

% Step 3: P(z=j|y,beta,sigma)
p = pz .* py / (pz'*py);
z = mnrnd(1,p) * (0:J)';