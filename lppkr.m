function [bestB, bestP, bestPP] = lppkr(X, XX, B0, P0, PP0, varargin)
%
% LPPKR: Learning Projections and Prototypes for Regression
%
% [B, P, PP] = lppkr(X, XX, B0, P0, PP0, ...)
%
% Input:
%   X       - Independent training data. Each column vector is a data point.
%   XX      - Dependent training data.
%   B0      - Initial projection base.
%   P0      - Initial independent prototype data.
%   PP0     - Initial dependent prototype data.
%
% Input (optional):
%   'slope',SLOPE              - Tanh slope (default=1)
%   'rateB',RATEB              - Projection base learning rate (default=0.1)
%   'rateP',RATEP              - Ind. Prototypes learning rate (default=0.1)
%   'ratePP',RATEPP            - Dep. Prototypes learning rate (default=0)
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'seed',SEED                - Random seed (default=system)
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stats',STAT               - Statistics every STAT (default={b:1,s:1000})
%   'orthoit',OIT              - Orthogonalize every OIT (default=1)
%   'orthonormal',(true|false) - Orthonormal projection base (default=true)
%   'orthogonal',(true|false)  - Orthogonal projection base (default=false)
%   'euclidean',(true|false)   - Euclidean distance (default=true)
%   'cosine',(true|false)      - Cosine distance (default=false)
%   'normalize',(true|false)   - Normalize training data (default=true)
%   'linearnorm',(true|false)  - Linear normalize training data (default=false)
%   'whiten',(true|false)      - Whiten training data (default=false)
%   'logfile',FID              - Output log file (default=stderr)
%   'verbose',(true|false)     - Verbose (default=true)
%
% Output:
%   B       - Final learned projection base
%   P       - Final learned independent prototype data
%   PP      - Final learned dependent prototype data
%
% $Revision$
% $Date$
%

%
% Copyright (C) 2008-2009 Mauricio Villegas (mvillegas AT iti.upv.es)
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

if strncmp(X,'-v',2),
  unix('echo "$Revision$* $Date$*" | sed "s/^:/lppkr: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='lppkr:';
minargs=5;

bestB=[];
bestP=[];
bestPP=[];

slope=1;
rateB=0.1;
rateP=0.1;
ratePP=0;

probeI=100;
probeunstable=0.2;
autoprobe=false;

epsilon=1e-7;
minI=100;
maxI=1000;
orthoit=1;

stochastic=false;
orthonormal=true;
orthogonal=false;
euclidean=true;
cosine=false;
normalize=true;
linearnorm=false;
whiten=false;
indepPP=false;

logfile=2;
verbose=true;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'probemode'),
    eval([varargin{n},'=varargin{n+1};']);
    n=n+2;
  elseif strcmp(varargin{n},'slope') || ...
         strcmp(varargin{n},'rateB') || ...
         strcmp(varargin{n},'rateP')  || ...
         strcmp(varargin{n},'ratePP')  || ...
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
    if ~isnumeric(varargin{n+1}) || sum(sum(varargin{n+1}<0))~=0,
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'orthonormal') || ...
         strcmp(varargin{n},'orthogonal') || ...
         strcmp(varargin{n},'euclidean') || ...
         strcmp(varargin{n},'cosine') || ...
         strcmp(varargin{n},'normalize') || ...
         strcmp(varargin{n},'linearnorm') || ...
         strcmp(varargin{n},'whiten') || ...
         strcmp(varargin{n},'stochastic') || ...
         strcmp(varargin{n},'autoprobe') || ...
         strcmp(varargin{n},'indepPP') || ...
         strcmp(varargin{n},'verbose'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      if varargin{n+1}==true,
        if strcmp(varargin{n},'orthonormal'),
          orthogonal=false;
        elseif strcmp(varargin{n},'orthogonal'),
          orthonormal=false;
        elseif strcmp(varargin{n},'euclidean'),
          cosine=false;
        elseif strcmp(varargin{n},'cosine'),
          euclidean=false;
        elseif strcmp(varargin{n},'normalize'),
          whiten=false;    linearnorm=false;
        elseif strcmp(varargin{n},'linearnorm'),
          normalize=false; whiten=false;
        elseif strcmp(varargin{n},'whiten'),
          normalize=false; linearnorm=false;
        end
      end
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

if exist('probemode','var'),
  rateB=probemode.rateB;
  rateP=probemode.rateP;
  ratePP=probemode.ratePP;
  minI=probemode.minI;
  maxI=probemode.maxI;
  probeunstable=minI;
  slope=probemode.slope;
  stats=probemode.stats;
  orthoit=probemode.orthoit;
  orthonormal=probemode.orthonormal;
  orthogonal=probemode.orthogonal;
  euclidean=probemode.euclidean;
  cosine=probemode.cosine;
  stochastic=probemode.stochastic;
  ind=probemode.ind;
  ind2=probemode.ind2;
  normalize=false;
  verbose=false;
  epsilon=0;
  probemode=true;
  xxsd=ones(DD,1);
else
  probemode=false;
end

if probemode,
elseif argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif size(B0,1)~=D || size(P0,1)~=D,
  fprintf(logfile,'%s error: dimensionality of base, prototypes and data must be the same\n',fn);
  return;
elseif size(XX,2)~=N || size(PP0,2)~=M,
  fprintf(logfile,'%s error: the number of vectors in the dependent and independent data must be the same\n',fn);
  return;
elseif size(PP0,1)~=DD,
  fprintf(logfile,'%s error: the dimensionality of the dependent variables for the data and the prototypes must be the same\n',fn);
  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

if ~probemode,
  tic;

  if normalize || linearnorm,
    xmu=mean(X,2);
    xsd=std(X,1,2);
    if euclidean,
      xsd=R*xsd;
    end
    if linearnorm,
      xsd=max(xsd)*ones(size(xsd));
    end
    if issparse(X) && ~cosine,
      xmu=full(xmu);
      xsd=full(xsd);
      X=X./xsd(:,ones(N,1));
      P0=P0./xsd(:,ones(M,1));
    else
      X=(X-xmu(:,ones(N,1)))./xsd(:,ones(N,1));
      P0=(P0-xmu(:,ones(M,1)))./xsd(:,ones(M,1));
    end
    B0=B0.*xsd(:,ones(R,1));
    if sum(xsd==0)>0,
      X(xsd==0,:)=[];
      B0(xsd==0,:)=[];
      P0(xsd==0,:)=[];
      fprintf(logfile,'%s warning: some dimensions have a standard deviation of zero\n',fn);
    end
  elseif whiten,
    [W,V]=pca(X);
    W=W(:,V>eps);
    V=V(V>eps);
    W=W.*repmat((1./sqrt(V))',D,1);
    if euclidean,
      W=(1/R).*W;
    end
    xmu=mean(X,2);
    X=W'*(X-xmu(:,ones(N,1)));
    P0=W'*(P0-xmu(:,ones(M,1)));
    IW=pinv(W);
    B0=IW*B0;
  end

  xxmu=mean(XX,2);
  xxsd=std(XX,1,2);
  XX=(XX-xxmu(:,ones(N,1)))./xxsd(:,ones(N,1));
  PP0=(PP0-xxmu(:,ones(M,1)))./xxsd(:,ones(M,1));
  xxsd=xxsd.*xxsd;

  if stochastic,
    if exist('seed','var'),
      rand('state',seed);
    end
    if ~exist('stats','var'),
      stats=1000;
    end
    minI=minI*stats;
    maxI=maxI*stats;
    orthoit=orthoit*stats;
  else
    if ~exist('stats','var'),
      stats=1;
    end
  end

  ind=1:M;
  ind=ind(ones(N,1),:);
  ind=ind(:);

  ind2=1:DD;
  ind2=ind2(ones(N,1),:);
  ind2=ind2(:);

  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,toc);
end

if autoprobe,
  probe=[zeros(3,1),10.^[-4:4;-4:4;-4:4]];
end
if exist('probe','var'),
  tic;
  probecfg.minI=round(probeunstable*probeI);
  probecfg.maxI=probeI;
  probecfg.slope=slope;
  probecfg.stats=stats;
  probecfg.orthoit=orthoit;
  probecfg.orthonormal=orthonormal;
  probecfg.orthogonal=orthogonal;
  probecfg.euclidean=euclidean;
  probecfg.cosine=cosine;
  probecfg.stochastic=stochastic;
  probecfg.ind=ind;
  probecfg.ind2=ind2;
  bestIJE=[0,1];
  ratesB=unique(probe(1,probe(1,:)>=0));
  ratesP=unique(probe(2,probe(2,:)>=0));
  ratesPP=unique(probe(3,probe(3,:)>=0));
  nB=1;
  while nB<=size(ratesB,2),
    nP=1;
    while nP<=size(ratesP,2),
      nPP=1;
      while nPP<=size(ratesPP,2),
        if ~(ratesB(nB)==0 && ratesP(nP)==0 && ratesPP(nPP)==0),
          probecfg.rateB=ratesB(nB);
          probecfg.rateP=ratesP(nP);
          probecfg.ratePP=ratesPP(nPP);
          [I,J]=lppkr(X,XX,B0,P0,PP0,'probemode',probecfg);
          mark='';
          if I>bestIJE(1) || (I==bestIJE(1) && J<bestIJE(2)),
            bestIJE=[I,J];
            rateB=ratesB(nB);
            rateP=ratesP(nP);
            ratePP=ratesPP(nPP);
            mark=' +';
          end
          if I<probeunstable*probeI,
            if nPP==1,
              if nP==1,
                nB=size(ratesB,2)+1;
              end
              nP=size(ratesP,2)+1;
            end
            break;
          end
          fprintf(logfile,'%s rates={%.2E %.2E %.2E} => impI=%.2f J=%.4f%s\n',fn,ratesB(nB),ratesP(nP),ratesPP(nPP),I/probeI,J,mark);
        end
        nPP=nPP+1;
      end
      nP=nP+1;
    end
    nB=nB+1;
  end
  fprintf(logfile,'%s total probe time (s): %f\n',fn,toc);
  fprintf(logfile,'%s selected rates={%.2E %.2E %.2E} impI=%.2f J=%.4f\n',fn,rateB,rateP,ratePP,bestIJE(1)/probeI,bestIJE(2));
end

if euclidean,
  rateB=2*rateB;
  rateP=2*rateP;
  ratePP=2*ratePP;
end
slope=slope/DD;
rateB=2*rateB*slope/N;
rateP=2*rateP*slope/N;
ratePP=2*ratePP*slope/N;
NDD=N*DD;

if orthonormal,
  B0=orthonorm(B0);
elseif orthogonal,
  B0=orthounit(B0);
end

Bi=B0;
Pi=P0;
PPi=PP0;
bestB=B0;
bestP=P0;
bestPP=PP0;
bestIJE=[0 1 Inf -1];

J0=1;
I=0;

mindist=100*sqrt(1/realmax);

fprintf(logfile,'%s Dx=%d Dxx=%d R=%d Nx=%d\n',fn,D,DD,R,N);
fprintf(logfile,'%s output: iteration | J | delta(J) | E\n',fn);

if ~probemode,
  tic;
end

if ~stochastic,

  while 1,

    rX=Bi'*X;
    rP=Bi'*Pi;

    if euclidean,

      dist=sum(power(repmat(rX,1,M)-rP(:,ind),2),1); dist(dist<mindist)=mindist; dist=reshape(1./dist,N,M);

    elseif cosine,

      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(ones(R,1),:);
      rxsd=sqrt(sum(rX.*rX,1));
      rX=rX./rxsd(ones(R,1),:);
      dist=1-sum(repmat(rX,1,M).*rP(:,ind),1); dist(dist<mindist)=mindist; dist=reshape(1./dist,N,M);

    end

    S=sum(dist,2);
    mXX=(reshape(sum(repmat(dist,DD,1).*PPi(ind2,:),2),N,DD)./S(:,ones(DD,1)))';
    dist=dist.*dist;

    dXX=mXX-XX;
    tanhXX=tanh(slope*sum(dXX.*dXX,1))';
    J=sum(tanhXX)/N;
    E=sqrt(sum(sum(dXX.*dXX,2).*xxsd)/NDD);

    mark='';
    if J<=bestIJE(2),
      bestB=Bi;
      bestP=Pi;
      bestPP=PPi;
      bestIJE=[I J E bestIJE(4)+1];
      mark=' *';
    end

    if probemode,
      if bestIJE(4)+(maxI-I)<probeunstable,
        break;
      end
    end

    if mod(I,stats)==0,
      fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J0,E,mark);
    end

    if I>=maxI,
      fprintf(logfile,'%s reached maximum number of iterations\n',fn);
      break;
    end

    if I>=minI,
      if abs(J0-J)<epsilon,
        fprintf(logfile,'%s index has stabilized\n',fn);
        break;
      end
    end

    J0=J;
    I=I+1;

    fact=repmat((1-tanhXX.*tanhXX)./S,M,1).*dist(:);
    if indepPP,
      fPP=permute(sum(reshape(fact(:,ones(DD,1)).*repmat(dXX,1,M)',N,M,DD)),[3 2 1]);
    else
      fPP=sum(reshape(fact.*sum(repmat(dXX,1,M),1)',N,M),1);
    end
    fact=fact.*sum(repmat(dXX,1,M).*(repmat(mXX,1,M)-PPi(:,ind)),1)';

    if euclidean,

      fact=reshape(fact(:,ones(R,1))'.*(repmat(rX,1,M)-rP(:,ind)),[R N M]);
      fP=-permute(sum(fact,2),[1 3 2]);
      fX=sum(fact,3);

    elseif cosine,

      fP=-permute(sum(reshape(fact(:,ones(R,1))'.*repmat(rX,1,M),[R N M]),2),[1 3 2]);
      fX=-sum(reshape(fact(:,ones(R,1))'.*rP(:,ind),[R N M]),3);

    end

    P0=Bi*fP;
    B0=X*fX'+Pi*fP';

    Bi=Bi-rateB*B0;
    Pi=Pi-rateP*P0;
    if indepPP,
      PPi=PPi-ratePP*fPP;
    else
      PPi=PPi-ratePP*fPP(ones(DD,1),:);
    end

    if mod(I,orthoit)==0,
      if orthonormal,
        Bi=orthonorm(Bi);
      elseif orthogonal,
        Bi=orthounit(Bi);
      end
    end

  end

end

if ~probemode,
  tm=toc;
  fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
  fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);
  fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJE(4)/max(I,1));
  fprintf(logfile,'%s best iteration: I=%d J=%f E=%f\n',fn,bestIJE(1),bestIJE(2),bestIJE(3));
end

if ~verbose,
  fclose(logfile);
end

if probemode,
  bestB=bestIJE(4);
  bestP=bestIJE(2);
  return;
end

bestPP=bestPP.*sqrt(xxsd(:,ones(M,1)))+xxmu(:,ones(M,1));
if normalize || linearnorm,
  if issparse(X) && ~cosine,
    bestP=bestP.*xsd(xsd~=0,ones(M,1));
  else
    bestP=bestP.*xsd(xsd~=0,ones(M,1))+xmu(xsd~=0,ones(M,1));
  end
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
