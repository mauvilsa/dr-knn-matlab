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
%     'devel',AMOUNT       - Use development set (default=false)
%     'approx',AMOUNT      - Use AUC approximation (default=false)
%     'logfile',FID        - Output log file (default=stderr)
%     'algorithm',         - Algorithm (default='ratio')
%       ('ratio'|'diff')
%
%   Output:
%     B                    - Final learned score weights
%     P                    - Final learned sigmoid slopes
%     Q                    - Final learned sigmoid displacements
%
%
% Version: 1.02 -- Sep/2008
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

beta=50;
rates=0;
gamma=0.5;
rho=0.5;
sigma=0.5;

epsilon=1e-7;
minI=100;
maxI=1000;

approx=false;
devel=false;
algorithm='diff';

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
         strcmp(varargin{n},'devel') || strcmp(varargin{n},'approx') || ...
         strcmp(varargin{n},'logfile'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(varargin{n+1}<0),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'algorithm'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~ischar(varargin{n+1}),
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

D=size(POS,2);
NP=size(POS,1);
NN=size(NEG,1);
overN=1/(NP*NN);

if rates>0,
  gamma=rates;
  rho=rates;
  sigma=rates;
end

if strcmp(algorithm,'ratio'),
  algoratio=true;
else
  algoratio=false;
end

sd=std(POS,1,1)+std(NEG,1,1)+std([mean(POS);mean(NEG)],1,1);
if sum(sd==0)>0,
  D=sum(sd~=0);
  POS(:,sd==0)=[];
  NEG(:,sd==0)=[];
  if ~islogical(B0),
    B0(sd==0)=[];
  end
  if ~islogical(P0),
    P0(sd==0)=[];
  end
  if ~islogical(Q0),
    Q0(sd==0)=[];
  end
  fprintf(logfile,'sfma: warning: some dimensions have a standard deviation of zero\n');
end

if islogical(B0) && B0==true,
  B0=power(10,6.*(auc(POS,NEG)-0.5));
end
if islogical(P0) && P0==true,
  P0=3./sd(sd~=0);
  sinv=mean(POS)<mean(NEG);
  P0(sinv)=-P0(sinv);
end
if islogical(Q0) && Q0==true,
  Q0=0.5.*(mean(POS)+mean(NEG));
end

if ~islogical(devel),
  NP=round(NP*(1-devel));
  NN=round(NN*(1-devel));
  devPOS=POS(NP+1:end,:);
  devNEG=NEG(NN+1:end,:);
  devNP=size(devPOS,1);
  devNN=size(devNEG,1);
  POS=POS(1:NP,:);
  NEG=NEG(1:NN,:);
  NP=size(POS,1);
  NN=size(NEG,1);
  overN=1/(NP*NN);
  devel=true;
end

if ~islogical(approx),
  origPOS=POS;
  origNEG=NEG;
  origNP=NP;
  origNN=NN;
  if max(size(approx))==1,
    if approx<=1,
      NP=round(approx*origNP);
      NN=round(approx*origNN);
    else
      NP=min([round(approx),origNP]);
      NN=min([round(approx),origNN]);
    end
  else
    if approx(1)<=1,
      NP=round(approx(1)*origNP);
    else
      NP=min([round(approx(1)),origNP]);
    end
    if approx(2)<=1,
      NN=round(approx(2)*origNN);
    else
      NN=min([round(approx(2)),origNN]);
    end
  end
  overN=1/(NP*NN);
  approx=true;
end

if argerr,
  fprintf(logfile,'sfma: error: incorrect input argument (%s,%d)\n',varargin{n},varargin{n+1});
elseif nargin-size(varargin,2)~=5,
  fprintf(logfile,'sfma: error: not enough input arguments\n');
elseif size(B0,1)~=1 || size(P0,1)~=1 || size(Q0,1)~=1 || size(B0,2)~=D || size(P0,2)~=D || size(Q0,2)~=D,
  fprintf(logfile,'sfma: error: B0, P0 and Q0 must be row vectors and have the same dimensionality\n');
elseif size(POS,2)~=D || size(NEG,2)~=D,
  fprintf(logfile,'sfma: error: POS and NEG must have the same columns\n');
else

  B0(B0<0)=0;
  B0=B0./sum(B0);

  bestB=B0;
  bestP=P0;
  bestQ=Q0;

  B=B0;
  P=P0;
  Q=Q0;

  I=0;

  bestI=I;
  bestJ=0;
  bestA=0;
  bestDevA=0;
  devA=0;

  prevJ=bestJ;

  better=' *';
  worse='';

  if devel,
    fprintf(logfile,'sfma: output: iteration | J | delta(J) | AUC | develAUC\n');
  else
    fprintf(logfile,'sfma: output: iteration | J | delta(J) | AUC\n');
  end

  tic;

  while 1,

    if devel,
      nePOS=1./(1+exp(P(ones(devNP,1),:).*(Q(ones(devNP,1),:)-devPOS)));
      ePOS=nePOS*B';
      neNEG=1./(1+exp(P(ones(devNN,1),:).*(Q(ones(devNN,1),:)-devNEG)));
      eNEG=neNEG*B';

      [scDEV,indDEV]=sort([eNEG;ePOS]);
      labelDEV=[false(devNN,1);true(devNP,1)];
      devA=(sum(find(labelDEV(indDEV)))-0.5*devNP*(devNP+1))/(devNP*devNN);
    end

    if approx,

      nPOS=1./(1+exp(P(ones(origNP,1),:).*(Q(ones(origNP,1),:)-origPOS)));
      fPOS=nPOS*B';
      nNEG=1./(1+exp(P(ones(origNN,1),:).*(Q(ones(origNN,1),:)-origNEG)));
      fNEG=nNEG*B';

      [fPOS,indPOS]=sort(fPOS);
      [fNEG,indNEG]=sort(fNEG);

      [sc,ind]=sort([fNEG;fPOS]);
      label=[false(origNN,1);true(origNP,1)];
      A=(sum(find(label(ind)))-0.5*origNP*(origNP+1))./(origNP*origNN);

      fPOS=fPOS(1:NP);
      iPOS=indPOS(1:NP);
      nPOS=nPOS(iPOS,:);
      POS=origPOS(iPOS,:);

      fNEG=fNEG(origNN-NN+1:origNN);
      iNEG=indNEG(origNN-NN+1:origNN);
      nNEG=nNEG(iNEG,:);
      NEG=origNEG(iNEG,:);

    else

      nPOS=1./(1+exp(P(ones(NP,1),:).*(Q(ones(NP,1),:)-POS)));
      fPOS=nPOS*B';
      nNEG=1./(1+exp(P(ones(NN,1),:).*(Q(ones(NN,1),:)-NEG)));
      fNEG=nNEG*B';

      [sc,ind]=sort([fNEG;fPOS]);
      label=[false(NN,1);true(NP,1)];
      A=(sum(find(label(ind)))-0.5*NP*(NP+1))./(NP*NN);

    end

    J=0;

    if algoratio,
      for p=1:NP,
        J=J+sum(1./(1+exp(-beta*(fPOS(p)./fNEG-1))));
      end
    else
      for p=1:NP,
        J=J+sum(1./(1+exp(beta*(fNEG-fPOS(p)))));
      end
    end

    J=overN*J;

    if ( ~devel && (A>bestA||(A==bestA&&J>=bestJ)) ) || ...
       ( devel && (devA>bestDevA||(devA==bestDevA&&A>bestA)||(devA==bestDevA&&A==bestA&&J>=bestJ)) ),
      bestB=B;
      bestP=P;
      bestQ=Q;
      bestI=I;
      bestJ=J;
      bestA=A;
      bestDevA=devA;
      isbetter=better;
    else
      isbetter=worse;
    end

    if devel,
      fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f\t%.8f%s\n',I,J,J-prevJ,A,devA,isbetter);
    else
      fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-prevJ,A,isbetter);
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

    if algoratio,
      for p=1:NP,
        ratio=fPOS(p)./fNEG;
        fact=exp(-beta*(ratio-1));
        fact=beta.*ratio.*fact./power(1+fact,2);
        cPOS(p)=cPOS(p)+sum(fact./fPOS(p));
        cNEG=cNEG+fact./fNEG;
      end
    else
      for p=1:NP,
        fact=exp(beta*(fNEG-fPOS(p)));
        fact=beta.*fact./power(1+fact,2);
        cPOS(p)=cPOS(p)+sum(fact);
        cNEG=cNEG+fact;
      end
    end

    B0=cPOS'*nPOS-cNEG'*nNEG;
    nPOS=power(nPOS,2).*(1./nPOS-1);
    nPOS(isnan(nPOS))=0;
    nNEG=power(nNEG,2).*(1./nNEG-1);
    nNEG(isnan(nNEG))=0;
    P0=B.*(cPOS'*((POS-Q(ones(NP,1),:)).*nPOS)-cNEG'*((NEG-Q(ones(NN,1),:)).*nNEG));
    Q0=B.*P.*(cNEG'*nNEG-cPOS'*nPOS);

    B=B+gamma.*overN.*B0;
    P=P+rho.*overN.*P0;
    Q=Q+sigma.*overN.*Q0;

    B(B<0)=0;
    B=B./sum(B);

  end

  tm=toc;

  fprintf(logfile,'sfma: average iteration time %f\n',tm/(I+0.5));
  if devel,
    fprintf(logfile,'sfma: best iteration %d, J=%.8f, AUC=%.8f, devAUC=%.8f\n',bestI,bestJ,bestA,bestDevA);
  else
    fprintf(logfile,'sfma: best iteration %d, J=%.8f, AUC=%.8f\n',bestI,bestJ,bestA);
  end

  if exist('sd'),
    if sum(sd==0)>0,
      BPQ=[bestB;bestP;bestQ];
      D=size(sd,2);
      bestB=zeros(1,D);
      bestP=zeros(1,D);
      bestQ=zeros(1,D);
      bestB(sd~=0)=BPQ(1,:);
      bestP(sd~=0)=BPQ(2,:);
      bestQ(sd~=0)=BPQ(3,:);
    end
  end

end
