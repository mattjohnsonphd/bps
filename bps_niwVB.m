function [mt,ct,nt,St] = bps_niwVB(MU,SIGMA,nt_init,tol)
% Performs Variational Bayes to approximate a NIW distribution to the data
% in MU and SIGMA, and reports the approximate KL Divergence. MU is a 
% J x nGibbs array of J-vectors arranged in columns. SIGMA is a 
% J x J x nGibbs array of JxJ covariance matrices. The  step to optimize nt
% requires an initial value nt_init and a tolerance tol to use in 
% Newton-Raphson.
%
% The distribution is parameterized such that 
% (mu|Sigma) ~ N(mt,ct*Sigma)
% Sigma ~ IW(nt,St) -> E[Lambda] = nt/(nt-2) * St

J = size(MU,1);
nGibbs = size(MU,2);

ESigmaInv = zeros(J);
ESigmaInvmu = zeros(J,1);
EeSigmaInve = 0;
ElogdetSigma = 0;

for i = 1:nGibbs
    SigmaInv = SIGMA(:,:,i)\eye(J);
    ESigmaInv = ESigmaInv + SigmaInv;
    ESigmaInvmu = ESigmaInvmu + SigmaInv*MU(:,i);
    ElogdetSigma = ElogdetSigma + log(det(SIGMA(:,:,i)));
end

ESigmaInv = ESigmaInv/nGibbs;
SigmaHarmonicMean = ESigmaInv\eye(J);
ESigmaInvmu = ESigmaInvmu/nGibbs;
ElogdetSigma = ElogdetSigma/nGibbs;

mt = SigmaHarmonicMean * ESigmaInvmu;

for i = 1:nGibbs
    SigmaInv = SIGMA(:,:,i)\eye(J);
    EeSigmaInve = EeSigmaInve + (MU(:,i)-mt)'*SigmaInv*(MU(:,i)-mt);
end
EeSigmaInve = EeSigmaInve/nGibbs;

ct = EeSigmaInve/J;

% Newton-Raphson to update n
nt = nt_init;
fn = ElogdetSigma - J*log((nt+J-1)/2) + log(det(ESigmaInv)) + sum(psi((nt+J-(1:J))/2));
while abs(fn) > tol
    fprimen = -J/(nt+J-1) + 0.5 * sum(psi(1,(nt+J-(1:J))/2));
    nt = max(nt - fn/fprimen,eps);
    fn = ElogdetSigma - J*log((nt+J-1)/2) + log(det(ESigmaInv)) + sum(psi((nt+J-(1:J))/2));
end

St = (nt+J-1)/nt * SigmaHarmonicMean;