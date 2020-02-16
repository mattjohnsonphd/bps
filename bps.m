% Mixture Model BPS for T forecasts

clear; clc;

%====================%
% Forecasts and Data %
%====================%

load('bps_eurForecastsFull.mat');

f=f1; s=s1; k=k1; % one step ahead forecasts
% f=fh; s=sh; k=kh; % h step ahead forecasts
clear 'f1' 'fh' 's1' 'sh' 'k1' 'kh';

% choose which models to keep, including base
% 1: TVAR(1)
% 2: TVAR(1)
% 3: TVAR(2)
% 4: TVAR(5)
% 5: DLM, locally linear, delta=.9, beta=.9
% 6: DLM, locally linear, delta=.925, beta=.925
% 7: DLM, locally linear, delta=.95, beta=.95
% 8: DLM, locally linear, delta=.9, beta=.99
% 9: DLM, locally constant, delta=0.9; beta=0.99;
% 10: DLM, locally constant, delta=0.925; beta=0.99;
% 11: DLM, locally constant, delta=0.95; beta=0.99;

ind = [1 5 6 7];
% ind = [1 3 4 5 6 7 9];
f=f(ind,:); s=s(ind,:); k=k(ind,:);
clear 'ind';

% pull out base density
base = 1; % which row to use as base density (of remaining 1+J)
f0=f(base,:); s0=s(base,:); k0=k(base,:); %pi0y=hy(base,:);
f(base,:)=[]; s(base,:)=[]; k(base,:)=[];
pi0y=tpdf((y'-f0)./sqrt(s0),k0)./sqrt(s0);

J = size(f,1);
lab = 'Base Density';
for j=1:J
    lab = char(lab,['A_' int2str(j)]);
end

%====================%
% Set BPS parameters %
%====================%

% discount factors for dynamic beta_t and Sigma_t
% set to 1 for constant
betaDisc = .975;
SigmaDisc = .99;
qDisc = 0.99;

% alpha parameters
d = 0.5; % in [0,1]
r1 = 72.1348; %173.8030;
r2 = 0.721348; % <d*r1
if r2 >= d*r1
    warning('User set r2 >= d*r1; \alpha_j(x) will be unimodal.')
end
alphaNorm = (d*r1/r2)^(-r2/(r1-r2))*(1-r2/r1);
% alphaNorm = 1; % for extreme r1, r2, above can evaluate to Inf

%================%
% Specify priors %
%================%

% Normal prior for beta_0 at time t=0:
% beta_t ~ N(bt,Bt)
% beta1|y0 ~ N(b0,B0/betaDisc)
bt = zeros(J,1);
r = 0.5; % pairwise correlation
Bt = r * ones(J) + (1-r)*eye(J);
Bt = .0001 * Bt; % actual stds of f-y: .0095, .0098, .0106

% IW prior for Sigma_0 at time t=0:
% Sigma_t ~ IW(n,S) -> E[Sigma_t] = nt/(nt-2) * St
% Sigma1|y0 ~ IW(SigmaDisc*nt,St)
nt = 15;
rho = 0.5;
St = rho * ones(J) + (1-rho)*eye(J);
St = s0(1) * St;

% Dirichlet prior for q_{0:J}
% q_{0:J} ~ Dir(lambdat)
ut = 50*ones(1+J,1);

savename = ['bps_eur1_d=' num2str(d) '_r1=' num2str(r1) '_r2=' num2str(r2) '_betaDisc=' num2str(betaDisc) '_u=' num2str(ut(1)) '.mat'];

%======================%
% Monte Carlo settings %
%======================%

% number of draws to estimate forecasts and log score at each time point
nmcForecast = 10000;

% number of Gibbs samples at each time point
nmcGibbs = 10000;

% number of draws of x vector in zSampler marginalization
nmcz = 10000;

% tolerance for Newton-Raphson in Variational bayes (n and lambdat)
tol = 0.00001;

%========================================%
% Forecast Combination and Gibbs Sampler %
%========================================%

%-----------------------------%
% Initialize storage matrices %
%-----------------------------%

% At each time point, update p(beta,Lambda) and store values
b = zeros(J,T);
B = zeros(J,J,T);
n = zeros(1,T);
S = zeros(J,J,T);
U = zeros(1+J,T);
KL = zeros(2,T); % MC KL divergence

% point predictions, 95% MC intervals, and standard deviations
BPS = zeros(T,4);

% Gibbs samples
Z = zeros(nmcGibbs,T);
X = zeros(J,nmcGibbs,T);
BETA = zeros(J,nmcGibbs,T);
SIGMA = zeros(J,J,nmcGibbs,T);
Q = zeros(1+J,nmcGibbs,T);
% Track number of samples in accept/reject steps
x_nSamples = zeros(nmcGibbs,T);
betaSigma_nSamples = zeros(nmcGibbs,T);
q_nSamples = zeros(nmcGibbs,T);

score = zeros(T,1);

tic
for t = 1:T

    clc; disp(['Forecast Combination and Gibbs sampler: t = ' num2str(t) '/' num2str(T)]);

    yt = y(t);
    ft = f(:,t); f0t = f0(t);
    st = s(:,t);
    kt = k(:,t);

    % Account for dynamics in q, beta, Lambda
    ut = qDisc * ut;
    Bt = Bt / betaDisc;
    nt = SigmaDisc * nt;    

    % initiate alpha_j(x) in mixture weights and alpha*h for log score
    alpha = zeros(J,1);
    alphah = zeros(J,1);

    % for faster samples from iwishrnd
    ntSt = nt*St;
    iwdof = nt+J-1;
%     if iwdof<J; iwdof=J; end
    % Note: MATLAB requires iwdof>=J, but theoretically iwdof just needs to
    % be >J-1. This can cause an error if not accounted for.
    [~,CntSt] = iwishrnd(ntSt,iwdof);

    % sample x0, x, beta, q in one step
    X0f = f0t + sqrt(s0(t)) * trnd(k0(t),1,nmcForecast); % 1 x nmc, samples from pi0(y)
    Xf = repmat(ft,1,nmcForecast) + diag(sqrt(st)) * trnd(repmat(kt,1,nmcForecast)); % J x nmc, samples from h(x)
    BETAf = mvnrnd(bt,Bt,nmcForecast)'; % J x nmcForecast
    Qf = dirrnd(ut,nmcForecast)'; Eq = ut/sum(ut);

    ALPHA = zeros(J,nmcForecast);
    ALPHAH = zeros(J,nmcForecast);
    
    Y=zeros(nmcForecast,T);

    for i = 1:nmcForecast
    
        % forecast distribution
        
        % sample mu, Sigma, x, q
        Sigma = iwishrnd(ntSt,iwdof,CntSt); % iwishrnd only generates one at a time
        beta = BETAf(:,i);
        mu = f0t + beta; % column vector
        x = Xf(:,i); % sample from h_{1:J}(x)
        q = Qf(:,i);
        
        % calculate alpha_j(x)
        for j = 1:J
            mumj = mu; mumj(j) = [];

            phi = Sigma(:,j); phi(j) = [];
            Sigmamj = Sigma; Sigmamj(j,:) = []; Sigmamj(:,j) = [];
            gamma = Sigmamj \ eye(J-1) * phi;
            delta = Sigma(j,j) - phi' * gamma;

            xmj = x; xmj(j) = [];
            ej = x(j) - mu(j) - gamma' * (xmj - mumj);
            alpha(j) = exp(-ej*ej/(2*r1*delta)) - d * exp(-ej*ej/(2*r2*delta));
            
            ejy = yt - f0t - gamma' * (xmj-mumj); % beta_j cancels
            alphah(j) = exp(-ejy*ejy/(2*r1*delta)) - d * exp(-ejy*ejy/(2*r2*delta));
            alphah(j) = alphah(j) * tpdf((yt+beta(j)-ft(j))/sqrt(st(j)),kt(j))/sqrt(st(j));
            
        end
        
        ALPHA(:,i) = alpha;
        ALPHAH(:,i) = alphah;
        
        % get component (multinomial 0:J with weights [1-q'alpha; q.*alpha]
        pz = [1-q(2:end)'*alpha; q(2:end).*alpha];
        pz(pz<0)=0; % possible rounding error
        z = mnrnd(1,pz) * (0:J)'; % index in 0:J
                
        % sample y from full conditional, i.e. alpha(y|...)
        if z == 0
            Y(i,t) = X0f(i);
        else
            Y(i,t) = x(z) - BETAf(z,i);
        end
        
    end
    
    % score
    score(t) = (1-Eq(2:end)'*mean(ALPHA,2))*pi0y(t) + Eq(2:end)'*mean(ALPHAH,2);            
        
    % point estimate and 95% interval
    BPS(t,1) = mean(Y(:,t));
    BPS(t,2:3) = quantile(Y(:,t),[0.025 0.975]);
    BPS(t,4) = std(Y(:,t));

    %---------------%
    % Gibbs Sampler %
    %---------------%

    % set initial values here
    beta = bt;
    Sigma = St;
    q = ut/sum(ut);
    for i = 1:nmcGibbs
        
        z = bps_zSampler(q,yt,beta,Sigma,r1,r2,d,f0t,pi0y(t),ft,st,kt,nmcz);
        [x,nSamples] = bps_xSampler(q,yt,z,beta,Sigma,r1,r2,d,f0t,ft,st,kt,alphaNorm);
        x_nSamples(i,t) = nSamples;
        [beta,Sigma,nSamples] = bps_betaSigmaSampler(q,x,yt,z,r1,r2,d,f0t,bt,Bt,nt,St,alphaNorm);
        betaSigma_nSamples(i,t) = nSamples;
        [q,nSamples] = bps_qSampler(x,z,beta,Sigma,r1,r2,d,f0t,ut);
        q_nSamples(i,t)=nSamples;
        
        X(:,i,t) = x;
        Z(i,t) = z;
        BETA(:,i,t) = beta;
        SIGMA(:,:,i,t) = Sigma;
        Q(:,i,t) = q;
        
    end
        
    %-------------------------------------------------------------%
    % Variational Bayes to approximate p(beta,Sigma|y) and p(q|y) %
    %-------------------------------------------------------------%
    
    [bt,Bt,nt,St,kl] = bps_VB2(BETA(:,:,t),SIGMA(:,:,:,t),nt,tol);
    KL(1,t) = kl;
    [ut,kl] = bps_dirVB(Q(:,:,t),ut,tol);
    KL(2,t) = kl;
    
    b(:,t) = bt;
    B(:,:,t) = Bt;
    n(t) = nt;
    S(:,:,t) = St;
    
    U(:,t) = ut;
    
end
time = toc;

save(savename); exit