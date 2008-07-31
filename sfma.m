function [bestB, bestP, bestQ] = sfma(POS, NEG, B0, P0, Q0, varargin)
%
% SFMA: Score Fusion by Maximizing the AUC
%
% [B, P, Q] = sfma(POS, NEG, B0, P0, Q0, ...)
%
%   Input:
%     POS                  - Positive scores matrix
%     NEG                  - Negative scores matrix
%     B0                   - Initial score weights
%     P0                   - Initial sigmoid slopes
%     Q0                   - Initial sigmoid displacements
%
%   Input (optional):
%     'beta',BETA          - Sigmoid slope (defaul=10)
%     'rates',RATES        - Set all learning rates (default=0.5)
%     'gamma',GAMMA        - Weights learning rate (default=0.5)
%     'rho',RHO            - Sigmoid slopes learning rate (default=0.5)
%     'sigma',SIGMA        - Sigmoid displacements learning rate (default=0.5)
%     'epsilon',EPSILON    - Convergence criterium (default=1e-7)
%     'minI',MINI          - Minimum number of iterations (default=100)
%     'maxI',MAXI          - Maximum number of iterations (default=1000)
%     'logfile',FID        - Output log file (default=stderr)
%
%   Output:
%     B                    - Final learned score weights
%     P                    - Final learned sigmoid slopes
%     Q                    - Final learned sigmoid displacements
%
%
% Version: 1.00 -- Jul/2008
%

%
% Copyright (C) 2008 Mauricio Villegas (mvillegas AT iti.upv.es)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

beta=10;
rates=0;
gamma=0.5;
rho=0.5;
sigma=0.5;

epsilon=1e-7;
minI=100;
maxI=1000;

logfile=2;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'beta') || strcmp(varargin{n},'gamma') || ...
         strcmp(varargin{n},'rho')  || strcmp(varargin{n},'sigma') || ...
         strcmp(varargin{n},'minI') || strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'epsilon') || strcmp(varargin{n},'rates') || ...
         strcmp(varargin{n},'logfile'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(varargin{n+1}<0),
      argerr=true;
    else
      n=n+2;
    end
  else
    argerr=true;
  end
  if argerr || n>size(varargin,2),
    break;
  end
end

if rates>0,
  gamma=rates;
  rho=rates;
  sigma=rates;
end

B0=B0./sum(B0);

bestB=B0;
bestP=P0;
bestQ=Q0;

D=max(size(B0));

if argerr,
  fprintf(logfile,'sfma: error: incorrect input argument (%d-%d)\n',n+5,n+6);
elseif size(B0,1)~=1 || size(P0,1)~=1 || size(Q0,1)~=1 || size(P0,2)~=D || size(Q0,2)~=D,
  fprintf(logfile,'sfma: error: B0, P0 and Q0 must be row vectors and have the same dimensionality\n');
elseif size(POS,2)~=D || size(NEG,2)~=D,
  fprintf(logfile,'sfma: error: POS and NEG must have the same columns as the dimensionality of B0, P0 and Q0\n');
else

  NP=size(POS,1);
  NN=size(NEG,1);
  overN=1/(NP*NN);

  B=B0;
  P=P0;
  Q=Q0;

  I=0;

  bestI=I;
  bestJ=0;
  bestA=0;

  prevJ=bestJ;

  fprintf(logfile,'sfma: output: iteration | J | delta(J) | AUC\n');

  tic;

  while 1,

    nPOS=1./(1+exp(P(ones(NP,1),:).*(Q(ones(NP,1),:)-POS)));
    fPOS=nPOS*B';

    nNEG=1./(1+exp(P(ones(NN,1),:).*(Q(ones(NN,1),:)-NEG)));
    fNEG=nNEG*B';

    J=0;
    A=0;
    AE=0;

    for p=1:NP,
      J=J+sum(1./(1+exp(beta*(fNEG-fPOS(p)))));
      A=A+sum(fPOS(p)>fNEG);
      AE=AE+sum(fPOS(p)==fNEG);
    end

    J=overN*J;
    A=overN*(A+0.5*AE);

    fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f\n',I,J,J-prevJ,A);

    if J>=bestJ,
      bestB=B;
      bestP=P;
      bestQ=Q;
      bestI=I;
      bestJ=J;
      bestA=A;
    end

    if I>=maxI,
      fprintf(logfile,'sfma: reached maximum number of iterations\n');
      break;
    end

    if I>=minI,
      if abs(J-prevJ)<epsilon,
        fprintf(logfile,'sfma: index has stabilized\n');
        break;
      end
    end

    prevJ=J;
    I=I+1;

    cPOS=zeros(NP,1);
    cNEG=zeros(NN,1);

    for p=1:NP,
      fact=exp(beta*(fNEG-fPOS(p)));
      fact=beta.*fact./power(1+fact,2);
      cPOS(p)=cPOS(p)+sum(fact);
      cNEG=cNEG+fact;
    end

    B0=cPOS'*nPOS-cNEG'*nNEG;
    nPOS=power(nPOS,2).*(1./nPOS-1);
    nPOS(isnan(nPOS))=0;
    nNEG=power(nNEG,2).*(1./nNEG-1);
    nNEG(isnan(nNEG))=0;
    P0=cPOS'*((POS-Q(ones(NP,1),:)).*nPOS)-cNEG'*((NEG-Q(ones(NN,1),:)).*nNEG);
    Q0=cNEG'*nNEG-cPOS'*nPOS;

    B=B+gamma.*overN.*B0;
    P=P+rho.*overN.*B.*P0;
    Q=Q+sigma.*overN.*B.*P.*Q0;

    B=B./sum(B);

  end

  tm=toc;

  fprintf(logfile,'sfma: average iteration time %f\n',tm/I);
  fprintf(logfile,'sfma: best iteration %d, J=%f, AUC=%f\n',bestI,bestJ,bestA);

end
