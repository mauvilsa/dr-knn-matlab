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
%   'rateB',RATEB              - Projection base learning rate (default=0.1)
%   'rateP',RATEP              - Prototypes learning rate (default=0.1)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'stats',STAT               - Statistics every STAT (default=1)
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
%   'prior',PRIOR              - A priori probabilities (default=Nc/N)
%   'devel',Y,Ylabels          - Set the development set (default=false)
%   'seed',SEED                - Random seed (default=system)
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=1)
%   'stocheck',SIT             - Stats every SIT stoch. iterations (default=100)
%   'stocheckfull',(true|f...  - Stats for whole data set (default=false)
%   'stochfinalexact',(tru...  - Final stats for whole data set (default=true)
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

%%% Default values %%%
bestB=[];
bestP=[];

slope=10;
rateB=0.1;
rateP=0.1;

probeI=100;
probeunstable=0.2;
autoprobe=false;

epsilon=1e-7;
minI=100;
maxI=1000;
stats=1;
orthoit=1;

devel=false;
stochastic=false;
stochsamples=1;
stocheck=100;
stocheckfull=false;
stochfinalexact=true;
orthonormal=true;
orthogonal=false;
euclidean=true;
cosine=false;
normalize=true;
linearnorm=false;
whiten=false;
testJ=false;

logfile=2;
verbose=true;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'probemode'),
    eval([varargin{n},'=varargin{n+1};']);
    n=n+2;
  elseif strcmp(varargin{n},'slope') || ...
         strcmp(varargin{n},'rates') || ...
         strcmp(varargin{n},'rateB') || ...
         strcmp(varargin{n},'rateP')  || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'probe') || ...
         strcmp(varargin{n},'probeI') || ...
         strcmp(varargin{n},'prior') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'stochsamples') || ...
         strcmp(varargin{n},'stocheck') || ...
         strcmp(varargin{n},'orthoit') || ...
         strcmp(varargin{n},'logfile'),
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
         strcmp(varargin{n},'stocheckfull') || ...
         strcmp(varargin{n},'stochfinalexact') || ...
         strcmp(varargin{n},'autoprobe') || ...
         strcmp(varargin{n},'testJ') || ...
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
  elseif strcmp(varargin{n},'devel'),
    devel=true;
    Y=varargin{n+1};
    Ylabels=varargin{n+2};
    n=n+3;
  else
    argerr=true;
  end
  if argerr || n>size(varargin,2),
    break;
  end
end

[D,Nx]=size(X);
C=max(size(unique(Plabels)));
R=size(B0,2);
Np=size(P0,2);
if devel,
  Ny=size(Y,2);
end

if exist('probemode','var'),
  rateB=probemode.rateB;
  rateP=probemode.rateP;
  minI=probemode.minI;
  maxI=probemode.maxI;
  probeunstable=minI;
  stats=probemode.stats;
  orthoit=probemode.orthoit;
  orthonormal=probemode.orthonormal;
  orthogonal=probemode.orthogonal;
  stochastic=probemode.stochastic;
  devel=probemode.devel;
  work=probemode.work;  
  if devel,
    dwork=probemode.dwork;
    Y=probemode.Y;
    Ylabels=probemode.Ylabels;
  end
  if stochastic,
    swork=probemode.swork;
    onesC=probemode.onesC;
    onesS=probemode.onesS;
    cumprior=probemode.cumprior;
    nc=probemode.nc;
    cnc=probemode.cnc;
    stochsamples=probemode.stochsamples;
    stocheck=probemode.stocheck;
    stocheckfull=probemode.stocheckfull;
    stochfinalexact=probemode.stochfinalexact;
  end
  normalize=false;
  verbose=false;
  epsilon=0;
  probemode=true;
else
  probemode=false;
end

%%% Error detection %%%
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
elseif max(size(Xlabels))~=Nx || min(size(Xlabels))~=1 || ...
       max(size(Plabels))~=Np || min(size(Plabels))~=1,
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
elseif max(size(unique(Xlabels)))~=C || ...
       sum(unique(Xlabels)~=unique(Plabels))~=0,
  fprintf(logfile,'%s error: there must be the same classes in labels and at least one prototype per class\n',fn);
  return;
elseif devel,
  if max(size(Ylabels))~=Ny || min(size(Ylabels))~=1,
    fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
    return;
  elseif max(size(unique(Ylabels)))~=C || ...
         sum(unique(Ylabels)~=unique(Plabels))~=0,
    fprintf(logfile,'%s error: there must be the same classes in labels and at least one prototype per class\n',fn);
    return;
  end
elseif exist('prior','var') && max(size(prior))~=C,
  fprintf(logfile,'%s error: the size of prior must be the same as the number of classes\n',fn);
  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

%%% Preprocessing %%%
if ~probemode,
  tic;

  if exist('rates','var'),
    rateB=rates;
    rateP=rates;
  end

  onesNx=ones(Nx,1);
  onesNp=ones(Np,1);
  onesR=ones(R,1);
  if devel,
    onesNy=ones(Ny,1);
  end

  %%% Normalization %%%
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
      X=X./xsd(:,onesNx);
      if devel,
        Y=Y./xsd(:,onesNy);
      end
      P0=P0./xsd(:,onesNp);
    else
      X=(X-xmu(:,onesNx))./xsd(:,onesNx);
      if devel,
        Y=(Y-xmu(:,onesNy))./xsd(:,onesNy);
      end
      P0=(P0-xmu(:,onesNp))./xsd(:,onesNp);
    end
    B0=B0.*xsd(:,onesR);
    if sum(xsd==0)>0,
      X(xsd==0,:)=[];
      if devel,
        Y(xsd==0,:)=[];
      end
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
    X=W'*(X-xmu(:,onesNx));
    if devel,
      Y=W'*(Y-xmu(:,onesNy));
    end
    P0=W'*(P0-xmu(:,onesNp));
    IW=pinv(W);
    B0=IW*B0;
  end

  %%% Adjusting the labels to be between 1 and C %%%
  clab=unique(Plabels);
  if clab(1)~=1 || clab(end)~=C || max(size(clab))~=C,
    nPlabels=ones(size(Plabels));
    nXlabels=ones(size(Xlabels));
    if devel,
      nYlabels=ones(size(Ylabels));
    end
    for c=2:C,
      nPlabels(Plabels==clab(c))=c;
      nXlabels(Xlabels==clab(c))=c;
      if devel,
        nYlabels(Ylabels==clab(c))=c;
      end
    end
    Plabels=nPlabels;
    Xlabels=nXlabels;
    if devel,
      Ylabels=nYlabels;
    end
  end
  clear clab nPlabels nXlabels nYlabels;

  if ~exist('prior','var'),
    prior=ones(C,1);
    for c=1:C,
      prior(c)=sum(Xlabels==c)/Nx;
    end
  end

  jfact=1;
  cfact=zeros(Nx,1);
  for c=1:C,
    cfact(Xlabels==c)=prior(c)/sum(Xlabels==c);
  end
  if euclidean,
    cfact=2*cfact;
    jfact=0.5;
  end

  %%% Stochastic preprocessing %%%
  if stochastic,
    [Xlabels,srt]=sort(Xlabels);
    X=X(:,srt);
    clear srt;
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
    orthoit=orthoit*stocheck;
    minI=minI*stocheck;
    maxI=maxI*stocheck;
    stats=stats*stocheck;
    onesC=ones(C,1);
  end

  %%% Initial parameter constraints %%%
  if orthonormal,
    B0=orthonorm(B0);
  elseif orthogonal,
    B0=orthounit(B0);
  end

  %%% Constant data structures %%%
  ind1=[1:Nx]';
  ind1=ind1(:,onesNp);
  ind1=ind1(:);

  ind2=1:Np;
  ind2=ind2(onesNx,:);
  ind2=ind2(:);

  sel=Plabels(:,onesNx)'==Xlabels(:,onesNp);

  work.slope=slope;
  work.ind1=ind1;
  work.ind2=ind2;
  work.sel=sel;
  work.onesR=onesR;
  work.Np=Np;
  work.Nx=Nx;
  work.C=C;
  work.R=R;
  work.euclidean=euclidean;
  work.cfact=cfact;
  work.jfact=jfact;
  work.prior=prior;

  if stochastic,
    onesS=ones(stochsamples,1);
    overS=1/stochsamples;

    ind1=[1:stochsamples]';
    ind1=ind1(:,onesNp);
    ind1=ind1(:);

    ind2=1:Np;
    ind2=ind2(onesS,:);
    ind2=ind2(:);

    swork.slope=slope;
    swork.ind1=ind1;
    swork.ind2=ind2;
    swork.onesR=onesR;
    swork.onesS=onesS;
    swork.overS=overS;
    swork.onesNp=onesNp;
    swork.Np=Np;
    swork.Nx=stochsamples;
    swork.C=C;
    swork.R=R;
    swork.euclidean=euclidean;
  end

  if devel,
    ind1=[1:Ny]';
    ind1=ind1(:,onesNp);
    ind1=ind1(:);

    ind2=1:Np;
    ind2=ind2(onesNy,:);
    ind2=ind2(:);

    sel=Plabels(:,onesNy)'==Ylabels(:,onesNp);

    dwork.slope=slope;
    dwork.ind1=ind1;
    dwork.ind2=ind2;
    dwork.sel=sel;
    dwork.onesR=onesR;
    dwork.Np=Np;
    dwork.Nx=Ny;
    dwork.C=C;
    dwork.R=R;
    dwork.euclidean=euclidean;
    dwork.prior=prior;
  end

  clear onesNx onesNy;
  clear ind1 ind2 sel;

  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,toc);
end

if autoprobe,
  probe=[zeros(2,1),10.^[-4:4;-4:4]];
end
if exist('probe','var'),
  tic;
  probecfg.onesR=onesR;
  probecfg.minI=round(probeunstable*probeI);
  probecfg.maxI=probeI;
  probecfg.stats=stats;
  probecfg.orthoit=orthoit;
  probecfg.orthonormal=orthonormal;
  probecfg.orthogonal=orthogonal;
  probecfg.stochastic=stochastic;
  probecfg.devel=devel;
  probecfg.work=work;
  if devel,
    probecfg.dwork=dwork;
    probecfg.Y=Y;
    probecfg.Ylabels=Ylabels;
  end
  if stochastic,
    probecfg.swork=swork;
    probecfg.onesC=onesC;
    probecfg.onesS=onesS;
    probecfg.cumprior=cumprior;
    probecfg.nc=nc;
    probecfg.cnc=cnc;
    probecfg.stochsamples=stochsamples;
    probecfg.stocheck=stocheck;
    probecfg.stocheckfull=stocheckfull;
    probecfg.stochfinalexact=stochfinalexact;
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

Bi=B0;
Pi=P0;
bestB=B0;
bestP=P0;
bestIJE=[0 1 1 -1];

J00=1;
J0=1;
I=0;

fprintf(logfile,'%s D=%d C=%d R=%d Nx=%d\n',fn,D,C,R,Nx);
fprintf(logfile,'%s output: iteration | J | delta(J) | E\n',fn);

if ~probemode,
  tic;
end

%%% Batch gradient descent %%%
if ~stochastic,

  while true,

    %%% Compute statistics %%%
    [E,J,fX,fP]=ldpp_index(Bi'*Pi,Plabels,Bi'*X,Xlabels,work);
    if devel,
      E=ldpp_index(Bi'*Pi,Plabels,Bi'*Y,Ylabels,dwork);
    end

    %%% Determine if there was improvement %%%
    mark='';
    if (  testJ && (J<bestIJE(2)||(J==bestIJE(2)&&E<=bestIJE(3))) ) || ...
       ( ~testJ && (E<bestIJE(3)||(E==bestIJE(3)&&J<=bestIJE(2))) ),
      bestB=Bi;
      bestP=Pi;
      bestIJE=[I J E bestIJE(4)+1];
      mark=' *';
    end

    %%% Print statistics %%%
    if mod(I,stats)==0,
      fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J00,E,mark);
      J00=J;
    end

    %%% Determine if algorithm has to be stopped %%%
    if probemode,
      if bestIJE(4)+(maxI-I)<probeunstable,
        break;
      end
    end
    if I>=maxI || ~isfinite(J) || ~isfinite(E) || (I>=minI && abs(J-J0)<epsilon),
      fprintf(logfile,'%s stopped iterating, ',fn);
      if I>=maxI,
        fprintf(logfile,'reached maximum number of iterations\n');
      elseif ~isfinite(J) || ~isfinite(E),
        fprintf(logfile,'reached unstable state\n');
      else
        fprintf(logfile,'index has stabilized\n');
      end
      break;
    end

    J0=J;
    I=I+1;

    %%% Update parameters %%%
    Bi=Bi-rateB.*(X*fX'+Pi*fP');
    Pi=Pi-rateP.*(Bi*fP);

    %%% Parameter constraints %%%
    if mod(I,orthoit)==0,
      if orthonormal,
        Bi=orthonorm(Bi);
      elseif orthogonal,
        Bi=orthounit(Bi);
      end
    end

  end % while true

%%% Stochasitc gradient descent %%%
else

  prevJ=1;
  prevE=1;

  while true,

    %%% Compute statistics %%%
    if mod(I,stocheck)==0 && stocheckfull,
      [E,J]=ldpp_index(Bi'*Pi,Plabels,Bi'*X,Xlabels,work);
      if devel,
        E=ldpp_index(Bi'*Pi,Plabels,Bi'*Y,Ylabels,dwork);
      end
    end

    %%% Select random samples %%%
    rands=rand(1,stochsamples);
    randc=(sum(rands(onesC,:)>cumprior(:,onesS))+1)';
    randn=cnc(randc)+round((nc(randc)-1).*rand(stochsamples,1))+1;
    sX=X(:,randn);

    %%% Compute statistics %%%
    [Ei,Ji,fX,fP]=ldpp_sindex(Bi'*Pi,Plabels,Bi'*sX,randc,swork);
    if ~stocheckfull,
      J=0.5*(prevJ+Ji);
      prevJ=J;
      if ~devel,
        E=0.5*(prevE+Ei);
        prevE=E;
      end
    end

    if mod(I,stocheck)==0,
      if ~stocheckfull && devel,
        E=ldpp_index(Bi'*Pi,Plabels,Bi'*Y,Ylabels,dwork);
      end

      %%% Determine if there was improvement %%%
      mark='';
      if (  testJ && (J<bestIJE(2)||(J==bestIJE(2)&&E<=bestIJE(3))) ) || ...
         ( ~testJ && (E<bestIJE(3)||(E==bestIJE(3)&&J<=bestIJE(2))) ),
        bestB=Bi;
        bestP=Pi;
        bestIJE=[I J E bestIJE(4)+1];
        mark=' *';
      end

      %%% Print statistics %%%
      if mod(I,stats)==0,
        fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J00,E,mark);
        J00=J;
      end

      %%% Determine if algorithm has to be stopped %%%
      if probemode,
        if bestIJE(4)+(maxI-I)<probeunstable,
          break;
        end
      end
      if I>=maxI || ~isfinite(J) || ~isfinite(E) || (I>=minI && abs(J-J0)<epsilon),
        fprintf(logfile,'%s stopped iterating, ',fn);
        if I>=maxI,
          fprintf(logfile,'reached maximum number of iterations\n');
        elseif ~isfinite(J) || ~isfinite(E),
          fprintf(logfile,'reached unstable state\n');
        else
          fprintf(logfile,'index has stabilized\n');
        end
        break;
      end

      J0=J;
    end % if mod(I,stocheck)==0

    I=I+1;

    %%% Update parameters %%%
    Bi=Bi-rateB.*(sX*fX'+Pi*fP');
    Pi=Pi-rateP.*(Bi*fP);

    %%% Parameter constraints %%%
    if mod(I,orthoit)==0,
      if orthonormal,
        Bi=orthonorm(Bi);
      elseif orthogonal,
        Bi=orthounit(Bi);
      end
    end

  end % while true

  %%% Parameter constraints %%%
  if orthonormal,
    bestB=orthonorm(bestB);
  elseif orthogonal,
    bestB=orthounit(bestB);
  end

  %%% Compute final statistics %%%
  if stochfinalexact && ~stocheckfull,
    [E,J]=ldpp_index(bestB'*bestP,Plabels,bestB'*X,Xlabels,work);
    if devel,
      E=ldpp_index(bestB'*bestP,Plabels,bestB'*Y,Ylabels,dwork);
    end

    fprintf(logfile,'%s best iteration approx: I=%d J=%f E=%f\n',fn,bestIJE(1),bestIJE(2),bestIJE(3));
    bestIJE(2)=J;
    bestIJE(3)=E;
  end % if stochfinalexact

end

if ~probemode,
  tm=toc;
  fprintf(logfile,'%s best iteration: I=%d J=%f E=%f\n',fn,bestIJE(1),bestIJE(2),bestIJE(3));
  fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJE(4)/max(I,1));
  fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
  fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);
end

if ~verbose,
  fclose(logfile);
end

if probemode,
  bestB=bestIJE(4);
  bestP=bestIJE(2);
  return;
end

%%% Compensate for normalization in the final parameters %%%
if normalize || linearnorm,
  if issparse(X) && ~cosine,
    bestP=bestP.*xsd(xsd~=0,onesNp);
  else
    bestP=bestP.*xsd(xsd~=0,onesNp)+xmu(xsd~=0,onesNp);
  end
  bestB=bestB./xsd(xsd~=0,onesR);
  if sum(xsd==0)>0,
    P=bestP;
    B=bestB;
    bestP=zeros(D,Np);
    bestP(xsd~=0,:)=P;
    bestB=zeros(D,R);
    bestB(xsd~=0,:)=B;
  end
elseif whiten,
  bestP=IW'*bestP+xmu(:,onesNp);
  bestB=W*bestB;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                            Helper functions                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [E, J, fX, fP] = ldpp_index(P, Plabels, X, Xlabels, work)

  R=work.R;
  Np=work.Np;
  Nx=work.Nx;
  onesR=work.onesR;
  sel=work.sel;
  prior=work.prior;

  %%% Compute distances %%%
  if work.euclidean,
    ds=reshape(sum((X(:,work.ind1)-P(:,work.ind2)).^2,1),Nx,Np);
  else %elseif cosine,
    psd=sqrt(sum(P.*P,1));
    P=P./psd(onesR,:);
    xsd=sqrt(sum(X.*X,1));
    X=X./xsd(onesR,:);
    ds=reshape(1-sum(X(:,work.ind1).*P(:,work.ind2),1),Nx,Np);
  end

  dd=ds;
  ds(~sel)=inf;
  dd(sel)=inf;
  [ds,is]=min(ds,[],2);
  [dd,id]=min(dd,[],2);
  ds(ds==0)=realmin;
  dd(dd==0)=realmin;

  %%% Compute statistics %%%
  E=0;
  for c=1:work.C,
    E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
  end
  if nargout>1,
    ratio=ds./dd;
    expon=exp(work.slope*(1-ratio));
    sigm=1./(1+expon);
    J=work.jfact*sum(work.cfact.*sigm);
  end

  %%% Compute gradient %%%
  if nargout>2,
    dsigm=work.slope.*expon./((1+expon).*(1+expon));
    ratio=work.cfact.*ratio;
    dfact=ratio.*dsigm;
    sfact=dfact./ds;
    dfact=dfact./dd;

    fP=zeros(R,Np);

    if work.euclidean,
      Xs=(X-P(:,is)).*sfact(:,onesR)';
      Xd=(X-P(:,id)).*dfact(:,onesR)';
      fX=Xs-Xd;
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    else %elseif cosine,
      Xs=X.*sfact(:,onesR)';
      Xd=X.*dfact(:,onesR)';
      fX=P(:,id).*dfact(:,onesR)'-P(:,is).*sfact(:,onesR)';
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [E, J, fX, fP] = ldpp_sindex(P, Plabels, X, Xlabels, work)

  R=work.R;
  Np=work.Np;
  Nx=work.Nx;
  onesR=work.onesR;
  slope=work.slope;
  overS=work.overS;

  %%% Compute distances %%%
  if work.euclidean,
    ds=reshape(sum((X(:,work.ind1)-P(:,work.ind2)).^2,1),Nx,Np);
  else %elseif cosine,
    psd=sqrt(sum(P.*P,1));
    P=P./psd(onesR,:);
    xsd=sqrt(sum(X.*X,1));
    X=X./xsd(onesR,:);
    ds=reshape(1-sum(X(:,work.ind1).*P(:,work.ind2),1),Nx,Np);
  end

  dd=ds;
  ssel=Plabels(:,work.onesS)'==Xlabels(:,work.onesNp);
  ds(~ssel)=inf;
  dd(ssel)=inf;
  [ds,is]=min(ds,[],2);
  [dd,id]=min(dd,[],2);
  ds(ds==0)=realmin;
  dd(dd==0)=realmin;
  ratio=ds./dd;
  expon=exp(slope*(1-ratio));
  sigm=1./(1+expon);

  %%% Compute statistics %%%
  J=overS*sum(sigm);
  E=overS*sum(dd<ds);

  %%% Compute gradient %%%
  dsigm=slope.*expon./((1+expon).*(1+expon));
  ratio=overS.*ratio;
  dfact=ratio.*dsigm;
  sfact=dfact./ds;
  dfact=dfact./dd;

  fP=zeros(R,Np);

  if work.euclidean,
    Xs=(X-P(:,is)).*sfact(:,onesR)';
    Xd=(X-P(:,id)).*dfact(:,onesR)';
    fX=Xs-Xd;
    for m=1:Np,
      fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
    end
  else %elseif cosine,
    Xs=X.*sfact(:,onesR)';
    Xd=X.*dfact(:,onesR)';
    fX=P(:,id).*dfact(:,onesR)'-P(:,is).*sfact(:,onesR)';
    for m=1:Np,
      fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
    end
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = orthonorm(X)

  for n=1:size(X,2),
    for m=1:n-1,
      X(:,n)=X(:,n)-(X(:,n)'*X(:,m))*X(:,m);
    end
    X(:,n)=(1/sqrt(X(:,n)'*X(:,n)))*X(:,n);
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = orthounit(X)

  onX=orthonorm(X);
  X=onX.*repmat(sum(onX'*X,1),size(X,1),1);
  X=sqrt(size(X,2)).*X./sqrt(sum(diag(X'*X)));
