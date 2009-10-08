function [bestB, bestP] = ldpp(X, Xlabels, B0, P0, Plabels, varargin)
%
% LDPP: Learning Discriminative Projections and Prototypes for NN Classification
%
% [B, P] = ldpp(X, Xlabels, B0, P0, Plabels, ...)
%
% Input:
%   X       - Data matrix. Each column vector is a data point.
%   Xlabels - Data class labels.
%   B0      - Initial projection base.
%   P0      - Initial prototypes.
%   Plabels - Prototype class labels.
%
% Input (optional):
%   'slope',SLOPE              - Sigmoid slope (defaul=10)
%   'rateB',RATEB              - Projection base learning rate (default=0.5)
%   'rateP',RATEP              - Prototypes learning rate (default=100)
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'prior',PRIOR              - A priori probabilities (default=Nc/N)
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
%   P       - Final learned prototypes
%
%
% Reference:
%
%   M. Villegas and R. Paredes. "Simultaneous Learning of a Discriminative
%   Projection and Prototypes for Nearest-Neighbor Classification."
%   CVPR'2008.
%
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
  unix('echo "$Revision$* $Date$*" | sed "s/^:/ldpp: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='ldpp:';
minargs=5;

bestB=[];
bestP=[];

slope=10;
rateB=0.5;
rateP=100;

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
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'prior') || ...
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
C=max(size(unique(Plabels)));
R=size(B0,2);
M=size(P0,2);

if exist('probemode','var'),
  rateB=probemode.rateB;
  rateP=probemode.rateP;
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
  prior=probemode.prior;
  cfact=probemode.cfact;
  ind=probemode.ind;
  sel=probemode.sel;
  if stochastic,
    cumprior=probemode.cumprior;
    nc=probemode.nc;
    cnc=probemode.cnc;
  end
  normalize=false;
  verbose=false;
  epsilon=0;
  probemode=true;
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
elseif max(size(Xlabels))~=N || min(size(Xlabels))~=1 || ...
       max(size(Plabels))~=M || min(size(Plabels))~=1,
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return
elseif max(size(unique(Xlabels)))~=C || ...
       sum(unique(Xlabels)~=unique(Plabels))~=0,
  fprintf(logfile,'%s error: there must be the same classes in labels and at least one prototype per class\n',fn);
  return;
elseif exist('prior','var') && max(size(prior))~=C,
  fprintf(logfile,'%s error: the size of prior must be the same as the number of classes\n',fn);
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
    X=(X-xmu(:,ones(N,1)))./xsd(:,ones(N,1));
    P0=(P0-xmu(:,ones(M,1)))./xsd(:,ones(M,1));
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

  if ~exist('prior','var'),
    prior=ones(C,1);
    for c=1:C,
      prior(c)=sum(Xlabels==c)/N;
    end
  end

  cfact=zeros(N,1);
  for c=1:C,
    cfact(Xlabels==c)=prior(c)/sum(Xlabels==c);
  end

  if stochastic,
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

  sel=Plabels(:,ones(N,1))'==Xlabels(:,ones(M,1));

  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,toc);
end

if autoprobe,
  probe=[zeros(2,1),10.^[-4:4;-4:4]];
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
  probecfg.prior=prior;
  probecfg.cfact=cfact;
  probecfg.ind=ind;
  probecfg.sel=sel;
  if stochastic,
    probecfg.cumprior=cumprior;
    probecfg.nc=nc;
    probecfg.cnc=cnc;
  end
  bestIJE=[0,1];
  ratesB=unique(probe(1,probe(1,:)>=0));
  ratesP=unique(probe(2,probe(2,:)>=0));
  nB=1;
  while nB<=size(ratesB,2),
    nP=1;
    while nP<=size(ratesP,2),
      if ~(ratesB(nB)==0 && ratesP(nP)==0),
        probecfg.rateB=ratesB(nB);
        probecfg.rateP=ratesP(nP);
        [I,J]=ldpp(X,Xlabels,B0,P0,Plabels,'probemode',probecfg);
        mark='';
        if I>bestIJE(1) || (I==bestIJE(1) && J<bestIJE(2)),
          bestIJE=[I,J];
          rateB=ratesB(nB);
          rateP=ratesP(nP);
          mark=' +';
        end
        if I<probeunstable*probeI,
          if nP==1,
            nB=size(ratesB,2)+1;
          end
          break;
        end
        fprintf(logfile,'%s rates={%.2E %.2E} => impI=%.2f J=%.4f%s\n',fn,ratesB(nB),ratesP(nP),I/probeI,J,mark);
      end
      nP=nP+1;
    end
    nB=nB+1;
  end
  fprintf(logfile,'%s total probe time (s): %f\n',fn,toc);
  fprintf(logfile,'%s selected rates={%.2E %.2E} impI=%.2f J=%.4f\n',fn,rateB,rateP,bestIJE(1)/probeI,bestIJE(2));
end

if euclidean,
  rateB=2*rateB;
  rateP=2*rateP;
end

if orthonormal,
  B0=orthonorm(B0);
elseif orthogonal,
  B0=orthounit(B0);
end

Bi=B0;
Pi=P0;
bestB=B0;
bestP=P0;
bestIJE=[0 1 1 -1];

J0=1;
I=0;

fprintf(logfile,'%s D=%d C=%d R=%d N=%d\n',fn,D,C,R,N);
fprintf(logfile,'%s output: iteration | J | delta(J) | E\n',fn);

if ~probemode,
  tic;
end

if ~stochastic,

  while 1,

    rX=Bi'*X;
    rP=Bi'*Pi;

    if euclidean,

      ds=reshape(sum(power(repmat(rX,1,M)-rP(:,ind),2),1),N,M);

    elseif cosine,

      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(ones(R,1),:);
      rxsd=sqrt(sum(rX.*rX,1));
      rX=rX./rxsd(ones(R,1),:);
      ds=reshape(1-sum(repmat(rX,1,M).*rP(:,ind),1),N,M);

    end

    dd=ds;
    ds(~sel)=inf;
    dd(sel)=inf;
    [ds,is]=min(ds,[],2);
    [dd,id]=min(dd,[],2);
    ds(ds==0)=realmin;
    dd(dd==0)=realmin;
    ratio=ds./dd;
    expon=exp(slope*(1-ratio));
    J=sum(cfact./(1+expon));
    E=0;
    for c=1:C,
      E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
    end

    mark='';
    if J<=bestIJE(2),
      bestB=Bi;
      bestP=Pi;
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

    ratio=cfact.*slope.*ratio.*expon./((1+expon).*(1+expon));
    ds=ratio./ds;
    dd=ratio./dd;

    if euclidean,

      rXs=(rX-rP(:,is)).*ds(:,ones(R,1))';
      rXd=(rX-rP(:,id)).*dd(:,ones(R,1))';
      fX=rXs-rXd;
      for m=1:M,
        fP(:,m)=sum(rXd(:,id==m),2)-sum(rXs(:,is==m),2);
      end

    elseif cosine,

      rXs=rX.*ds(:,ones(R,1))';
      rXd=rX.*dd(:,ones(R,1))';
      fX=rP(:,id).*dd(:,ones(R,1))'-rP(:,is).*ds(:,ones(R,1))';
      for m=1:M,
        fP(:,m)=sum(rXd(:,id==m),2)-sum(rXs(:,is==m),2);
      end

    end

    P0=Bi*fP;
    B0=X*fX'+Pi*fP';

    Bi=Bi-rateB*B0;
    Pi=Pi-rateP*P0;

    if mod(I,orthoit)==0,
      if orthonormal,
        Bi=orthonorm(Bi);
      elseif orthogonal,
        Bi=orthounit(Bi);
      end
    end

  end

else % stochastic

  while 1,

    if mod(I,stats)==0,

      rX=Bi'*X;
      rP=Bi'*Pi;

      if euclidean,

        ds=reshape(sum(power(repmat(rX,1,M)-rP(:,ind),2),1),N,M);

      elseif cosine,

        rpsd=sqrt(sum(rP.*rP,1));
        rP=rP./rpsd(ones(R,1),:);
        rxsd=sqrt(sum(rX.*rX,1));
        rX=rX./rxsd(ones(R,1),:);
        ds=reshape(1-sum(repmat(rX,1,M).*rP(:,ind),1),N,M);

      end

      dd=ds;
      ds(~sel)=inf;
      dd(sel)=inf;
      ds=min(ds,[],2);
      dd=min(dd,[],2);
      ds(ds==0)=realmin;
      dd(dd==0)=realmin;
      ratio=ds./dd;
      expon=exp(slope*(1-ratio));
      J=sum(cfact./(1+expon));
      E=0;
      for c=1:C,
        E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
      end

      mark='';
      if J<=bestIJE(2),
        bestB=Bi;
        bestP=Pi;
        bestIJE=[I J E bestIJE(4)+1];
        mark=' *';
      end

      if probemode,
        if bestIJE(4)+(maxI-I)<probeunstable,
          break;
        end
      end

      fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J0,E,mark);

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

    end

    I=I+1;

    c=sum(rand>cumprior)+1;
    n=cnc(c)+round((nc(c)-1)*rand)+1;

    rX=Bi'*X(:,n);
    rP=Bi'*Pi;

    if euclidean,
      dist=sum(power(rX(:,ones(M,1))-rP,2),1);
      dsn=min(dist(Plabels==Xlabels(n)));
      ddn=min(dist(Plabels~=Xlabels(n)));
    elseif cosine,
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
    expon=exp(slope*(1-ratio));
    sigm=(cfact(n)./prior(c)).*slope.*ratio.*expon./((1+expon).*(1+expon));
    dsn=sigm./dsn;
    ddn=sigm./ddn;

    if euclidean,
      B0=dsn.*(X(:,n)-Pi(:,is))*(rX-rP(:,is))'-ddn.*(X(:,n)-Pi(:,id))*(rX-rP(:,id))';
      P0=zeros(size(Pi));
      P0(:,is)=-dsn.*Bi*(rX-rP(:,is));
      P0(:,id)= ddn.*Bi*(rX-rP(:,id));
    else
      B0=-dsn.*(X(:,n)*rP(:,is)'+Pi(:,is)*rX')+ddn.*(X(:,n)*rP(:,id)'+Pi(:,id)*rX');
      P0=zeros(size(Pi));
      P0(:,is)=-dsn.*Bi*rX;
      P0(:,id)= ddn.*Bi*rX;
    end

    Bi=Bi-rateB*B0;
    Pi=Pi-rateP*P0;

    if orthonormal && mod(I,orthoit)==0,
      Bi=orthonorm(Bi);
    elseif orthogonal,
      Bi=orthounit(Bi);
    end

  end

  if orthonormal,
    bestB=orthonorm(bestB);
  elseif orthogonal,
    bestB=orthounit(bestB);
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
