function [x, nSamples] = bps_xSampler(q,y,z,beta,Sigma,r1,r2,d,f0,f,s,k,alphaNorm)

J = length(beta);
accept = 0;
nSamples = 0;

rs=sqrt(s); % for faster sampling

if z > 0
    
    phi=Sigma(:,z); phi(z)=[];
    Sigmamj=Sigma; Sigmamj(z,:)=[]; Sigmamj(:,z)=[];
    gamma=Sigmamj\eye(J-1)*phi;
    delta=Sigma(z,z)-phi'*gamma;
    
    betaj=beta(z); betamj=beta; betamj(z)=[];
    
    while accept == 0
        
        nSamples = nSamples + 1;
        
        x = f + rs .* trnd(k);
        x(z)=y+betaj; xmj=x; xmj(z)=[];
        
        ej = y - f0 - gamma' * (xmj-f0-betamj); % betaj cancels
        
        alpha = exp(-ej*ej/(2*r1*delta))-d*exp(-ej*ej/(2*r2*delta));
        
        if rand < alpha/alphaNorm
            accept = 1;
        end
            
    end
    
else % z == 0
    
    gamma = zeros(J-1,J);
    delta = zeros(J,1);
    alpha = zeros(J,1);
    
    q1J = q(2:1+J); % for faster sampling
    
    for j = 1:J
        phi=Sigma(:,j); phi(j)=[];
        Sigmamj=Sigma; Sigmamj(j,:)=[]; Sigmamj(:,j)=[];
        gamma(:,j)=Sigmamj\eye(J-1)*phi;
        delta(j)=Sigma(j,j)-phi'*gamma(:,j);
    end
    
    while accept == 0
        
        nSamples = nSamples + 1;
        
        x = f + rs .* trnd(k);
        
        for j = 1:J
            xmj=x; xmj(j)=[];
            betamj=beta; betamj(j)=[];
            ej = x(j) - f0 - beta(j) - gamma(:,j)' * (xmj-f0-betamj);
            alpha(j) = exp(-ej*ej/(2*r1*delta(j)))-d*exp(-ej*ej/(2*r2*delta(j)));
        end
        
        if rand > q1J'*alpha % or if rand < 1-q1J'*alpha
            accept = 1;
        end
        
    end
    
end