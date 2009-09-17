function [bestB, bestP, bestPP] = lppkr_tmp(X, XX, B0, P0, PP0, varargin)
%
% LPPKR: Learning Projections and Prototypes for k-NN Regression
%
% [B, P, PP] = lppkr(X, XX, B0, P0, PP0, ...)
%
%   Input:
%     X       - Independent trainig data. Each column vector is a data point.
%     XX      - Dependent training data.
%     B0      - Initial projection base.
%     P0      - Initial independent prototype data.
%     PP0     - Initial dependent prototype data.
%
%   Input (optional):
%     'beta',BETA                - Sigmoid slope (defaul=1)
%     'rateB',RATEB              - Projection base learning rate (default=0.5)
%     'rateP',RATEP              - Ind. Prototypes learning rate (default=0.5)
%     'ratePP',RATEPP            - Dep. Prototypes learning rate (default=0)
%     'rates',RATES              - Set all learning rates to RATES
%     'probe',PROBE              - Probe learning rates (default=false)
%     'probeI',PROBEI            - Iterations for probing learning rates (default=100)
%     'epsilon',EPSILON          - Convergence criterium (default=1e-7)
%     'minI',MINI                - Minimum number of iterations (default=100)
%     'maxI',MAXI                - Maximum number of iterations (default=1000)
%     'seed',SEED                - Random seed (default=system)
%     'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%     'stats',STAT               - Statistics every STAT iterations (default={b:1,s:1000})
%     'orthoit',OIT              - Orthogonalize every OIT iterations (default={b:1,s:1000})
%     'orthonormal',(true|false) - Orthonormal projection base (default=true)
%     'orthogonal',(true|false)  - Orthogonal projection base (default=false)
%     'normalize',(true|false)   - Normalize training data (default=false)
%     'linearnorm',(true|false)  - Linear normalize training data (default=true)
%     'whiten',(true|false)      - Whiten the training data (default=false)
%     'logfile',FID              - Output log file (default=stderr)
%     'verbose',(true|false)     - Verbose (default=true)
%     'distance',('euclidean'|   - Used distance (default='euclidean')
%                 'cosine')
%
%   Output:
%     B   - Final learned projection base
%     P   - Final learned independent prototype data
%     PP  - Final learned dependent prototype data
%
%
% Version: 1.04 -- Sep/2009
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
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
%

beta=1;
rateB=0.5;
rateP=0.5;
ratePP=0;
probeI=100;
probemode=false;

epsilon=1e-7;
minI=100;
maxI=1000;

stochastic=false;

orthonormal=true;
orthogonal=false;
normalize=true;
whiten=false;
linearnorm=false;

distance='euclidean';

logfile=2;
verbose=true;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'beta') || ...
         strcmp(varargin{n},'rateB') || ...
         strcmp(varargin{n},'rateP')  || ...
         strcmp(varargin{n},'ratePP')  || ...
         strcmp(varargin{n},'rates')  || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'probeI') || ...
         strcmp(varargin{n},'probe') || ...
         strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'orthoit') || ...
         strcmp(varargin{n},'seed'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(varargin{n+1}<0),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'normalize') || ...
         strcmp(varargin{n},'whiten') || ...
         strcmp(varargin{n},'linearnorm') || ...
         strcmp(varargin{n},'orthonormal') || ...
         strcmp(varargin{n},'orthogonal') || ...
         strcmp(varargin{n},'balance') || ...
         strcmp(varargin{n},'protoweight') || ...
         strcmp(varargin{n},'verbose') || ...
         strcmp(varargin{n},'probemode') || ...
         strcmp(varargin{n},'stochastic'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      if varargin{n+1}==true,
        if strcmp(varargin{n},'normalize'),
          whiten=false;    linearnorm=false;
        elseif strcmp(varargin{n},'whiten'),
          normalize=false; linearnorm=false;
        elseif strcmp(varargin{n},'linearnorm'),
          normalize=false; whiten=false;
        elseif strcmp(varargin{n},'orthonormal'),
          orthogonal=false;
        elseif strcmp(varargin{n},'orthogonal'),
          orthonormal=false;
        end
      end
      n=n+2;
    end
  elseif strcmp(varargin{n},'distance'),
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

[D,N]=size(X);
DD=size(XX,1);
R=size(B0,2);
M=size(P0,2);

if argerr,
  fprintf(logfile,'lppkr: error: incorrect input argument (%d-%d)\n',varargin{n},varargin{n+1});
elseif nargin-size(varargin,2)~=5,
  fprintf(logfile,'lppkr: error: not enough input arguments\n');
elseif size(B0,1)~=D,
  fprintf(logfile,'lppkr: error: dimensionality of base and data must be the same\n');
elseif size(P0,1)~=D,
  fprintf(logfile,'lppkr: error: dimensionality of prototypes and data must be the same\n');
elseif size(XX,2)~=N || size(PP0,2)~=M,
  fprintf(logfile,'lppkr: error: the number of vectors in the dependent and independent data must be the same\n');
elseif size(PP0,1)~=DD,
  fprintf(logfile,'lppkr: error: the dimensionality of the dependent variables for the data and the prototypes must be the same\n');
elseif ~(strcmp(distance,'euclidean') || strcmp(distance,'cosine')),
  fprintf(logfile,'lppkr: error: invalid distance\n');
else

  if normalize || linearnorm,
    xmu=mean(X,2);
    xsd=std(X,1,2)*sqrt(D);
    if linearnorm,
      xsd=max(xsd)*ones(size(xsd));
    end
    X=(X-xmu(:,ones(N,1)))./xsd(:,ones(N,1));
    P0=(P0-xmu(:,ones(M,1)))./xsd(:,ones(M,1));
    B0=B0.*xsd(:,ones(R,1));
    if sum(xsd==0)>0,
      X(xsd==0,:)=[];
      B0(xsd==0,:)=[];
      P0(xsd==0,:)=[];
      fprintf(logfile,'lppkr: warning: some dimensions have a standard deviation of zero\n');
    end
  elseif whiten,
    [W,V]=pca(X);
    W=W(:,V>1e-9);
    V=V(V>1e-9);
    W=W.*repmat((1./sqrt(V))',D,1);
    W=(1/sqrt(R)).*W;
    xmu=mean(X,2);
    X=W'*(X-xmu(:,ones(N,1)));
    P0=W'*(P0-xmu(:,ones(M,1)));
    IW=pinv(W);
    B0=IW*B0;
  end

  ppmu=mean(PP0,2);
  ppsd=std(PP0,1,2);
  PP0=(PP0-ppmu(:,ones(M,1)))./ppsd(:,ones(M,1));
  XX=(XX-ppmu(:,ones(N,1)))./ppsd(:,ones(N,1));
  ppsd=ppsd.*ppsd;

  if orthonormal,
    B0=orthonorm(B0);
  elseif orthogonal,
    B0=orthounit(B0);
  end

  if stochastic,
    if exist('seed','var'),
      rand('state',seed);
    end
    if ~exist('stats','var'),
      stats=1000;
    end
    if ~exist('orthoit','var'),
      orthoit=1000;
    end
  else
    if ~exist('stats','var'),
      stats=1;
    end
    if ~exist('orthoit','var'),
      orthoit=1;
    end
  end

  if exist('probe','var'),
    unstable=0.2*probeI;
    bestI=0;
    bestJ=1.0;
    rateB=0;
    rateP=0;
    ratePP=0;
    ratesB=unique(probe(1,probe(1,:)>=0));
    ratesP=unique(probe(2,probe(2,:)>=0));
    ratesPP=unique(probe(3,probe(3,:)>=0));
    nB=1;
    tic;
    while nB<=size(ratesB,2),
      rB=ratesB(nB);
      nP=1;
      while nP<=size(ratesP,2),
        rP=ratesP(nP);
        nPP=1;
        while nPP<=size(ratesPP,2),
          rPP=ratesPP(nPP);
          [I, J] = lppkr_tmp(X,XX,B0,P0,PP0, ...
                             'probemode',true, 'verbose',false, 'normalize',false, ...
                             'rateB',rB,'rateP',rP,'ratePP',rPP, 'maxI',probeI );
          if I>bestI || (I==bestI && J<bestJ),
            if verbose,
              fprintf(logfile,'lppkr_probeRates: rates={%.2E %.2E %.2E} => impI=%.2f J=%.4f ++\n',rB,rP,rPP,I/probeI,J);
            end
            bestI=I;
            bestJ=J;
            rateB=rB;
            rateP=rP;
            ratePP=rPP;
          else          
            if verbose,
              fprintf(logfile,'lppkr_probeRates: rates={%.2E %.2E %.2E} => impI=%.2f J=%.4f\n',rB,rP,rPP,I/probeI,J);
            end
            if I<unstable,
              if nPP==1,
                if nP==1,
                  nB=size(ratesB,2)+1;
                end
                nP=size(ratesP,2)+1;
              end
              break;
            end
          end
          nPP=nPP+1;
        end
        nP=nP+1;
      end
      nB=nB+1;
    end
    tm=toc;
    if verbose,
      fprintf(logfile,'lppkr_probeRates: total time (s): %f\n',tm);
      fprintf(logfile,'lppkr_probeRates: selected rates={%.2E %.2E %.2E}\n',rateB,rateP,ratePP);
    end
  end

  euclidean=true;
  if strcmp(distance,'cosine'),
    euclidean=false;
  end

  if exist('rates','var'),
    rateB=rates;
    rateP=rates;
    ratePP=rates;
  end
  if euclidean,
    rateB=2*rateB;
    rateP=2*rateP;
    ratePP=2*ratePP;
  end
  beta=beta/DD;
  rateB=2*rateB*beta/N;
  rateP=2*rateP*beta/N;
  ratePP=2*ratePP*beta/N;
  NDD=N*DD;

  B=B0;
  P=P0;
  PP=PP0;
  bestB=B0;
  bestP=P0;
  bestPP=PP0;
  bestI=0;
  bestJ=1;
  bestE=Inf;
  nimp=-1;

  J0=1;
  I=0;

  ind=1:M;
  ind=ind(ones(N,1),:);
  ind=ind(:);

  ind2=1:DD;
  ind2=ind2(ones(N,1),:);
  ind2=ind2(:);

  mindist=100*sqrt(1/realmax); %%% g(d)=1/d
  %mindist=R/100; %%% g(d)=1/(d+R/100)

  if verbose,
    fprintf(logfile,'lppkr: Dx=%d Dxx=%d R=%d Nx=%d\n',D,DD,R,N);
    fprintf(logfile,'lppkr: output: iteration | J | delta(J) | rmse\n');
  end

  tic;

  if ~stochastic,

    while 1,

      rX=B'*X;
      rP=B'*P;

      if euclidean,

        %dist=reshape(exp(-sum(power(repmat(rX,1,M)-rP(:,ind),2),1)),N,M); %%% g(d)=exp(-d)
        dist=sum(power(repmat(rX,1,M)-rP(:,ind),2),1); dist(dist<mindist)=mindist; dist=reshape(1./dist,N,M); %%% g(d)=1/d
        %dist=reshape(1./(sum(power(repmat(rX,1,M)-rP(:,ind),2),1)+mindist),N,M); %%% g(d)=1/(d+R/100)

      else % cosine

        rpsd=sqrt(sum(rP.*rP,1));
        rP=rP./rpsd(ones(R,1),:);
        rxsd=sqrt(sum(rX.*rX,1));
        rX=rX./rxsd(ones(R,1),:);
        %dist=reshape(exp(-(1-sum(repmat(rX,1,M).*rP(:,ind),1))),N,M); %%% g(d)=exp(-d)
        dist=1-sum(repmat(rX,1,M).*rP(:,ind),1); dist(dist<mindist)=mindist; dist=reshape(1./dist,N,M); %%% g(d)=1/d
        %dist=reshape(1./(1-sum(repmat(rX,1,M).*rP(:,ind),1)+mindist),N,M); %%% g(d)=1/(d+R/100)

      end

      S=sum(dist,2);
      mXX=(reshape(sum(repmat(dist,DD,1).*PP(ind2,:),2),N,DD)./S(:,ones(DD,1)))';
      dist=dist.*dist; %%% g(d)=1/d, g(d)=1/(d+R/100)

      dXX=mXX-XX;
      tanhXX=tanh(beta*sum(dXX.*dXX,1))';
      J=sum(tanhXX)/N;
      E=sqrt(sum(sum(dXX.*dXX,2).*ppsd)/NDD);

      mark='';
      if J<=bestJ,
        bestB=B;
        bestP=P;
        bestPP=PP;
        bestI=I;
        bestJ=J;
        bestE=E;
        mark=' *';
        nimp=nimp+1;
      end

      if verbose && mod(I,stats)==0,
        fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J0,E,mark);
      end

      if I>=maxI,
        if verbose,
          fprintf(logfile,'lppkr: reached maximum number of iterations\n');
        end
        break;
      end

      if I>=minI,
        if abs(J0-J)<epsilon,
          if verbose,
            fprintf(logfile,'lppkr: index has stabilized\n');
          end
          break;
        end
      end

      J0=J;
      I=I+1;

      fact=repmat((1-tanhXX.*tanhXX)./S,M,1).*dist(:);
      fPP=sum(reshape(fact.*sum(repmat(dXX,1,M),1)',N,M),1);
      fact=fact.*sum(repmat(dXX,1,M).*(repmat(mXX,1,M)-PP(:,ind)),1)';

      if euclidean,

        %fact=repmat((1-tanhXX.*tanhXX)./S,M,1).*dist(:).*sum(repmat(dXX,1,M).*(repmat(mXX,1,M)-PP(:,ind)),1)';
        fact=reshape(fact(:,ones(R,1))'.*(repmat(rX,1,M)-rP(:,ind)),[R N M]);
        fP=-permute(sum(fact,2),[1 3 2]);
        fX=sum(fact,3);

      else % cosine

        %fact=repmat((1-tanhXX.*tanhXX)./S,M,1).*dist(:).*sum(repmat(dXX,1,M).*(repmat(mXX,1,M)-PP(:,ind)),1)';
        fP=-permute(sum(reshape(fact(:,ones(R,1))'.*repmat(rX,1,M),[R N M]),2),[1 3 2]);
        fX=-sum(reshape(fact(:,ones(R,1))'.*rP(:,ind),[R N M]),3);

      end

      P0=B*fP;
      B0=X*fX'+P*fP';

      B=B-rateB*B0;
      P=P-rateP*P0;
      PP=PP-ratePP*fPP(ones(DD,1),:);

      if mod(I,orthoit)==0,
        if orthonormal,
          B=orthonorm(B);
        elseif orthogonal,
          B=orthounit(B);
        end
      end

    end

  end

  tm=toc;

  bestPP=bestPP.*sqrt(ppsd(:,ones(M,1)))+ppmu(:,ones(M,1));

  if normalize || linearnorm,
    bestP=bestP.*xsd(xsd~=0,ones(M,1))+xmu(xsd~=0,ones(M,1));
    bestB=bestB./xsd(xsd~=0,ones(R,1));
    if sum(xsd==0)>0,
      P=bestP;
      B=bestB;
      bestP=zeros(D,M);
      bestP(xsd~=0,:)=P;
      bestB=zeros(D,R);
      bestB(xsd~=0,:)=B;
    end
  elseif whiten,
    bestP=IW'*bestP+xmu(:,ones(M,1));
    bestB=W*bestB;
  end

  if verbose,
    fprintf(logfile,'lppkr: average iteration time (ms): %f\n',1000*tm/I);
    fprintf(logfile,'lppkr: total time (s): %f\n',tm);
    fprintf(logfile,'lppkr: amount of improvement iterations: %f\n',nimp/I);
    fprintf(logfile,'lppkr: best iteration: I=%d, J=%f, RMSE=%f\n',bestI,bestJ,bestE);
  end

  if probemode,
    bestB=nimp;
    bestP=bestJ;
    bestPP=bestE;
  end

end
