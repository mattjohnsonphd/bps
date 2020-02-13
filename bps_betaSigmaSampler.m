function [beta, Sigma, nSamples] = bps_betaSigmaSampler(q,x,y,z,r1,r2,d,f0,b,c,n,S,alphaNorm)

accept = 0;
nSamples = 0;
J = length(b);

% for faster IW samples
nS=n*S; iwdof=n+J-1; [~,CnS]=iwishrnd(nS,iwdof);

if z > 0
    
    xj=x(z); xmj=x; xmj(z)=[];
    betaj = xj - y;
    bj=b(z); bmj=b; bmj(z)=[];
    
    while accept == 0
        
        nSamples = nSamples + 1;
        
        Sigma = iwishrnd(nS,iwdof,CnS);
        phi=Sigma(:,z); phi(z)=[];
        Sigmamj=Sigma; Sigmamj(z,:)=[]; Sigmamj(:,z)=[];
        gamma=Sigmamj\eye(J-1)*phi;
        delta=Sigma(z,z)-phi'*gamma;
        
        betamjMean = bmj + phi/Sigma(z,z) * (betaj-bj);
        betamjCov = c*(Sigmamj-phi*phi'/Sigma(z,z));
        betamj = mvnrnd(betamjMean,betamjCov)';
        
        ej = xj - f0 - betaj - gamma' * (xmj-f0-betamj);
        alpha = exp(-ej*ej/(2*r1*delta))-d*exp(-ej*ej/(2*r2*delta));
        
        if rand < alpha/alphaNorm
            beta = [betamj(1:z-1); betaj; betamj(z:J-1)];
            accept = 1;
        end
        
    end
        
else % z==0
    
    q1J = q(2:1+J); % for faster sampling
    alpha = zeros(J,1);
    
    while accept == 0
        
        nSamples = nSamples + 1;
        
        Sigma = iwishrnd(nS,iwdof,CnS);
        beta = mvnrnd(b,c*Sigma)'; % column vector
        
        for j = 1:J
            phi=Sigma(:,j); phi(j)=[];
            Sigmamj=Sigma; Sigmamj(j,:)=[]; Sigmamj(:,j)=[];
            gamma=Sigmamj\eye(J-1)*phi;
            delta=Sigma(j,j)-phi'*gamma;
        
            xmj=x; xmj(j)=[];
            betamj=beta; betamj(j)=[];
            
            ej = x(j) - f0 - beta(j) - gamma' * (xmj-f0-betamj);
            alpha(j) = exp(-ej*ej/(2*r1*delta))-d*exp(-ej*ej/(2*r2*delta));
        end
        
        if rand > q1J'*alpha % or if rand < 1-q1J'*alpha
            accept = 1;
        end
        
    end
        
end