function [bestB, bestP] = ldpp(X, Xlabels, B0, P0, Plabels, varargin)
%
% LDPP: Learning Discriminative Projections and Prototypes
%
% [B, P] = ldpp(X, Xlabels, B0, P0, Plabels, ...)
%
%   Input:
%     X       - Data matrix. Each column vector is a data point.
%     Xlabels - Data class labels.
%     B0      - Initial projection base.
%     P0      - Initial prototypes.
%     Plabels - Prototype class labels.
%
%   Input (optional):
%     'beta',BETA                - Sigmoid slope (defaul=10)
%     'gamma',GAMMA              - Projection base learning rate (default=0.5)
%     'eta',ETA                  - Prototypes learning rate (default=100)
%     'epsilon',EPSILON          - Convergence criterium (default=1e-7)
%     'minI',MINI                - Minimum number of iterations (default=100)
%     'maxI',MAXI                - Maximum number of iterations (default=1000)
%     'prior',PRIOR              - A priori probabilities (default=Nc/N)
%     'balance',(true|false)     - A priori probabilities = 1/C (default=false)
%     'stats',STAT               - Statistics every STAT iterations (default=1000)
%     'seed',SEED                - Random seed (default=system)
%     'stochastic',(true|false)  - Stochastic gradient ascend (default=true)
%     'orthoit',OIT              - Orthogonalize every OIT iterations (default=1)
%     'orthonormal',(true|false) - Orthonormal projection base (default=true)
%     'orthogonal',(true|false)  - Orthogonal projection base (default=false)
%     'normalize',(true|false)   - Normalize training data (default=true)
%     'whiten',(true|false)      - Whiten the training data (default=false)
%     'protoweight',(true|false) - Weigth prototype update (default=false)
%     'logfile',FID              - Output log file (default=stderr)
%     'distance',('euclidean'|   - NN distance (default='euclidean')
%                 'cosine')
%
%   Output:
%     B   - Final learned projection base
%     P   - Final learned prototypes
%
%
% Reference:
%
%   M. Villegas and R. Paredes. "Simultaneous Learning of a Discriminative
%   Projection and Prototypes for Nearest-Neighbor Classification."
%   CVPR'2008.
%
%
% Version: 1.05 -- Mar/2009
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

beta=10;
gamma=0.5;
eta=100;

epsilon=1e-7;
minI=100;
maxI=1000;

stochastic=false;
seed=rand('seed');
stats=1000;
orthoit=100;

orthonormal=true;
orthogonal=false;
normalize=true;
whiten=false;
balance=false;
protoweight=false;
distance='euclidean';

logfile=2;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'beta') || strcmp(varargin{n},'gamma') || ...
         strcmp(varargin{n},'eta')  || strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'prior') || strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'stats') || strcmp(varargin{n},'orthoit') || ...
         strcmp(varargin{n},'seed'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(varargin{n+1}<0),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'normalize') || strcmp(varargin{n},'whiten') || ...
         strcmp(varargin{n},'orthonormal') || strcmp(varargin{n},'orthogonal') || ...
         strcmp(varargin{n},'balance') || strcmp(varargin{n},'protoweight') || ...
         strcmp(varargin{n},'stochastic'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      if varargin{n+1}==true,
        if strcmp(varargin{n},'normalize'),
          whiten=false;
        elseif strcmp(varargin{n},'whiten'),
          normalize=false;
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

N=size(X,2);
D=size(X,1);
R=size(B0,2);
M=size(P0,2);
C=max(size(unique(Plabels)));

if argerr,
  fprintf(logfile,'ldpp: error: incorrect input argument (%d-%d)\n',varargin{n},varargin{n+1});
elseif nargin-size(varargin,2)~=5,
  fprintf(logfile,'ldpp: error: not enough input arguments\n');
elseif max(size(Xlabels))~=N || min(size(Xlabels))~=1,
  fprintf(logfile,'ldpp: error: Xlabels must be a vector with size the same as the number of data points\n');
elseif max(size(Plabels))~=M || min(size(Plabels))~=1,
  fprintf(logfile,'ldpp: error: Plabels must be a vector with size the same as the number of prototypes\n');
elseif max(size(unique(Xlabels)))~=C || sum(unique(Xlabels)~=unique(Plabels))~=0,
  fprintf(logfile,'ldpp: error: there must be the same classes in Xlabels and Plabels, and there must be at least one prototype per class\n');
elseif size(B0,1)~=D,
  fprintf(logfile,'ldpp: error: dimensionality of base and data must be the same\n');
elseif size(P0,1)~=D,
  fprintf(logfile,'ldpp: error: dimensionality of prototypes and data must be the same\n');
elseif exist('prior','var') && balance,
  fprintf(logfile,'ldpp: error: either specify the priors or set balance to true, but not both\n');
elseif exist('prior','var') && max(size(prior))~=C,
  fprintf(logfile,'ldpp: error: the size of prior must be the same as the number of classes\n');
elseif ~(strcmp(distance,'euclidean') || strcmp(distance,'cosine')),
  fprintf(logfile,'ldpp: error: invalid distance\n');
else

  if normalize,
    xmu=mean(X,2);
    xsd=std(X,1,2);
    X=(X-xmu(:,ones(N,1)))./xsd(:,ones(N,1));
    P0=(P0-xmu(:,ones(M,1)))./xsd(:,ones(M,1));
    B0=B0.*xsd(:,ones(R,1));
    if sum(xsd==0)>0,
      X(xsd==0,:)=[];
      B0(xsd==0,:)=[];
      P0(xsd==0,:)=[];
      fprintf(logfile,'ldpp: warning: some dimensions have a standard deviation of zero\n');
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

  if orthonormal,
    B0=orthonorm(B0);
  elseif orthogonal,
    B0=orthounit(B0);
  end

  clab=unique(Plabels);
  if clab(1)~=1 || clab(end)~=C || max(size(clab))~=C,
    nPlabels=ones(size(Plabels));
    nXlabels=ones(size(Xlabels));
    for c=2:C,
      nPlabels(Plabels==clab(c))=c;
      nXlabels(Xlabels==clab(c))=c;
    end
    Plabels=nPlabels;
    Xlabels=nXlabels;
  end

  if balance,
    prior=(1/C).*ones(C,1);
  end

  if ~exist('prior','var'),
    prior=ones(C,1);
    for c=1:C,
      prior(c)=sum(Xlabels==c)/N;
    end
  end

  if stochastic,
    rand('seed',seed);
    [Xlabels,srt]=sort(Xlabels);
    X=X(:,srt);
    cumprior=cumsum(prior);
    nc=zeros(C,1);
    cnc=zeros(C,1);
    nc(1)=sum(Xlabels==1);
    for c=2:C,
      nc(c)=sum(Xlabels==c);
      cnc(c)=cnc(c-1)+nc(c-1);
    end
  end

  cfact=zeros(N,1);
  for c=1:C,
    cfact(Xlabels==c)=prior(c)/sum(Xlabels==c);
  end

  euclidean=true;
  if strcmp(distance,'cosine'),
    euclidean=false;
  end

  if euclidean,
    gamma=2*gamma;
    eta=2*eta;
  end

  dist=zeros(M,1);
  ds=zeros(N,1);
  dd=zeros(N,1);
  is=zeros(N,1);
  id=zeros(N,1);

  B=B0;
  P=P0;
  bestB=B0;
  bestP=P0;
  bestI=0;
  bestJ=1;
  bestE=1;

  J0=1;
  I=0;

  fprintf(logfile,'ldpp: output: iteration | J | delta(J) | error\n');

  tic;

  if ~stochastic,

  while 1,

    rX=B'*X;
    rP=B'*P;

    if euclidean,

      for n=1:N,
        dist=sum(power(rX(:,n*ones(M,1))-rP,2),1);
        ds(n)=min(dist(Plabels==Xlabels(n)));
        dd(n)=min(dist(Plabels~=Xlabels(n)));
        is(n)=find(dist==ds(n),1);
        id(n)=find(dist==dd(n),1);
      end

    else
      
      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(ones(R,1),:);
      for n=1:N,
        rX(:,n)=rX(:,n)./sqrt(rX(:,n)'*rX(:,n));
        dist=1-sum(rX(:,n*ones(M,1)).*rP,1);
        ds(n)=min(dist(Plabels==Xlabels(n)));
        dd(n)=min(dist(Plabels~=Xlabels(n)));
        is(n)=find(dist==ds(n),1);
        id(n)=find(dist==dd(n),1);
      end

    end

    ds(ds==0)=realmin;
    dd(dd==0)=realmin;
    ratio=ds./dd;
    expon=exp(beta*(1-ratio));
    J=sum(cfact./(1+expon));
    E=0;
    for c=1:C,
      E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
    end

    mark='';
    if J<=bestJ,
      bestB=B;
      bestP=P;
      bestI=I;
      bestJ=J;
      bestE=E;
      mark=' *';
    end

    fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J0,E,mark);

    if I>=maxI,
      fprintf(logfile,'ldpp: reached maximum number of iterations\n');
      break;
    end

    if I>=minI,
      if abs(J0-J)<epsilon,
        fprintf(logfile,'ldpp: index has stabilized\n');
        break;
      end
    end

    J0=J;
    I=I+1;

    ratio=cfact.*beta.*ratio.*expon./((1+expon).*(1+expon));
    ds=ratio./ds;
    dd=ratio./dd;

    if euclidean,

      rXs=(rX-rP(:,is)).*ds(:,ones(R,1))';
      rXd=(rX-rP(:,id)).*dd(:,ones(R,1))';
      for m=1:M,
        rP0(:,m)=sum(rXd(:,id==m),2)-sum(rXs(:,is==m),2);
      end
      P0=B*rP0;
      Xs=X-P(:,is);
      Xd=X-P(:,id);
      B0=Xs*rXs'-Xd*rXd';

    else

      rXs=rX.*ds(:,ones(R,1))';
      rXd=rX.*dd(:,ones(R,1))';
      for m=1:M,
        rP0(:,m)=sum(rXd(:,id==m),2)-sum(rXs(:,is==m),2);
      end
      P0=B*rP0;
      rX=rP(:,id).*dd(:,ones(R,1))'-rP(:,is).*ds(:,ones(R,1))';
      B0=X*rX'+P(:,id)*rXd'-P(:,is)*rXs';

    end

    if protoweight,
      for m=1:M,
        Nm=sum(id==m)+sum(is==m);
        if Nm>0,
          P0(:,m)=P0(:,m)*(N/Nm);
        end
      end
    end

    B=B-gamma*B0;
    P=P-eta*P0;

    if orthonormal,
      B=orthonorm(B);
    elseif orthogonal,
      B=orthounit(B);
    end

  end

  else % stochastic

  while 1,

    if mod(I,stats)==0,

      rX=B'*X;
      rP=B'*P;

      if euclidean,

        for n=1:N,
          dist=sum(power(rX(:,n*ones(M,1))-rP,2),1);
          ds(n)=min(dist(Plabels==Xlabels(n)));
          dd(n)=min(dist(Plabels~=Xlabels(n)));
        end

      else

        rpsd=sqrt(sum(rP.*rP,1));
        rP=rP./rpsd(ones(R,1),:);
        for n=1:N,
          rX(:,n)=rX(:,n)./sqrt(rX(:,n)'*rX(:,n));
          dist=1-sum(rX(:,n*ones(M,1)).*rP,1);
          ds(n)=min(dist(Plabels==Xlabels(n)));
          dd(n)=min(dist(Plabels~=Xlabels(n)));
        end

      end

      ds(ds==0)=realmin;
      dd(dd==0)=realmin;
      ratio=ds./dd;
      expon=exp(beta*(1-ratio));
      J=sum(cfact./(1+expon));
      E=0;
      for c=1:C,
        E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
      end

      mark='';
      if J<=bestJ,
        bestB=B;
        bestP=P;
        bestI=I;
        bestJ=J;
        bestE=E;
        mark=' *';
      end

      fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J0,E,mark);

      if I>=maxI,
        fprintf(logfile,'ldpp: reached maximum number of iterations\n');
        break;
      end

      if I>=minI,
        if abs(J0-J)<epsilon,
          fprintf(logfile,'ldpp: index has stabilized\n');
          break;
        end
      end

      J0=J;

    end

    I=I+1;

    c=sum(rand>cumprior)+1;
    n=cnc(c)+round((nc(c)-1)*rand)+1;

    rX=B'*X(:,n);
    rP=B'*P;

    if euclidean,
      dist=sum(power(rX(:,ones(M,1))-rP,2),1);
      dsn=min(dist(Plabels==Xlabels(n)));
      ddn=min(dist(Plabels~=Xlabels(n)));
    else
      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(ones(R,1),:);
      rX=rX./sqrt(rX'*rX);
      dist=1-sum(rX(:,n*ones(M,1)).*rP,1);
      dsn=min(dist(Plabels==Xlabels(n)));
      ddn=min(dist(Plabels~=Xlabels(n)));
    end

    is=find(dist==dsn,1);
    id=find(dist==ddn,1);

    ratio=dsn./ddn;
    expon=exp(beta*(1-ratio));
    sigm=(cfact(n)./prior(c)).*beta.*ratio.*expon./((1+expon).*(1+expon));
    dsn=sigm./dsn;
    ddn=sigm./ddn;

    if euclidean,
      B0=dsn.*(X(:,n)-P(:,is))*(rX-rP(:,is))'-ddn.*(X(:,n)-P(:,id))*(rX-rP(:,id))';
      P0=zeros(size(P));
      P0(:,is)=-dsn.*B*(rX-rP(:,is));
      P0(:,id)= ddn.*B*(rX-rP(:,id));
    else
      B0=-dsn.*(X(:,n)*rP(:,is)'+P(:,is)*rX')+ddn.*(X(:,n)*rP(:,id)'+P(:,id)*rX');
      P0=zeros(size(P));
      P0(:,is)=-dsn.*B*rX;
      P0(:,id)= ddn.*B*rX;
    end

    B=B-gamma*B0;
    P=P-eta*P0;

    if orthonormal && mod(I,orthoit)==0,
      B=orthonorm(B);
    elseif orthogonal,
      B=orthounit(B);
    end

  end

  if orthonormal,
    bestB=orthonorm(bestB);
  elseif orthogonal,
    bestB=orthounit(bestB);
  end

  end

  tm=toc;

  if normalize,
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

  fprintf(logfile,'ldpp: average iteration time %f\n',tm/I);
  fprintf(logfile,'ldpp: best iteration %d, J=%f, E=%f\n',bestI,bestJ,bestE);

end
