function [bestB, bestP] = ppma_new(X, Xlabels, B0, P0, Plabels, varargin)
%
% PPMA: Projections and Prototypes by Maximizing AUC
%
% [B, P] = ppma(X, Xlabels, B0, P0, Plabels, ...)
%
% Input:
%   X       - Data matrix. Each column vector is a data point.
%   Xlabels - Data class labels.
%   B0      - Initial projection base.
%   P0      - Initial prototypes.
%   Plabels - Prototype class labels.
%
% Input (optional):
%   'slope',SLOPE              - Sigmoid slope (defaul=30)
%   'rateB',RATEB              - Projection base learning rate (default=0.1)
%   'rateP',RATEP              - Prototypes learning rate (default=0.1)
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%%   'prior',PRIOR              - A priori probabilities (default=Nc/N)
%   'devel',Y,Ylabels          - Set the development set (default=false)
%   'seed',SEED                - Random seed (default=system)
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=1)
%   'stocheck',SIT             - Check every SIT stochastic iterations (default=100)
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
% $Revision: 70 $
% $Date: 2009-10-19 09:34:45 +0200 (Mon, 19 Oct 2009) $
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
  unix('echo "$Revision: 70 $* $Date: 2009-10-19 09:34:45 +0200 (Mon, 19 Oct 2009) $*" | sed "s/^:/ppma: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='ppma:';
minargs=5;

%%% Default values %%%
bestB=[];
bestP=[];

c2faster=false;
c2faster=true;
auctypeA=false;
auctypeA=true;

slope=30;
rateB=0.1;
rateP=0.1;

probeI=100;
probeunstable=0.2;
autoprobe=false;

adaptrates=false;
adaptrate=0.1;
adaptdecay=0.9;

epsilon=1e-7;
minI=100;
maxI=1000;
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
testA=true;

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
         strcmp(varargin{n},'rateB') || ...
         strcmp(varargin{n},'rateP')  || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'probeI') || ...
         strcmp(varargin{n},'probe') || ...
         strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'orthoit') || ...
         strcmp(varargin{n},'adaptrate') || ...
         strcmp(varargin{n},'adaptdecay') || ...
         strcmp(varargin{n},'stochsamples') || ...
         strcmp(varargin{n},'stocheck') || ...
         strcmp(varargin{n},'seed'),
         %strcmp(varargin{n},'prior') || ...
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
         strcmp(varargin{n},'autoprobe') || ...
         strcmp(varargin{n},'adaptrates') || ...
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

if exist('probemode','var'),
  onesR=probemode.onesR;
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
  %prior=probemode.prior;
  %cfact=probemode.cfact;
  %jfact=probemode.jfact;
  %ind1=probemode.ind1;
  %ind2=probemode.ind2;
  %sel=probemode.sel;
  devel=probemode.devel;
  if devel,
    Y=probemode.Y;
    Ylabels=probemode.Ylabels;
  end
  if stochastic,
    onesNp=probemode.onesNp;
    %cumprior=probemode.cumprior;
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

[D,Nx]=size(X);
C=max(size(unique(Plabels)));
R=size(B0,2);
Np=size(P0,2);
if devel,
  Ny=size(Y,2);
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
%elseif exist('prior','var') && max(size(prior))~=C,
%  fprintf(logfile,'%s error: the size of prior must be the same as the number of classes\n',fn);
%  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

%%% Preprocessing %%%
if ~probemode,
  tic;

  onesNx=ones(Nx,1);
  onesNp=ones(Np,1);
  onesR=ones(R,1);
  onesC=ones(C,1);
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
      P0=P0./xsd(:,onesNp);
      if devel,
        Y=Y./xsd(:,onesNy);
      end
    else
      X=(X-xmu(:,onesNx))./xsd(:,onesNx);
      P0=(P0-xmu(:,onesNp))./xsd(:,onesNp);
      if devel,
        Y=(Y-xmu(:,onesNy))./xsd(:,onesNy);
      end
    end
    B0=B0.*xsd(:,onesR);
    if sum(xsd==0)>0,
      X(xsd==0,:)=[];
      B0(xsd==0,:)=[];
      P0(xsd==0,:)=[];
      if devel,
        Y(xsd==0,:)=[];
      end
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
    P0=W'*(P0-xmu(:,onesNp));
    if devel,
      Y=W'*(Y-xmu(:,onesNy));
    end
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

  %jfact=1;
  %cfact=zeros(Nx,1);
  %for c=1:C,
  %  cfact(Xlabels==c)=prior(c)/sum(Xlabels==c);
  %end
  %if euclidean,
  %  cfact=2*cfact;
  %  jfact=0.5;
  %end

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
    if ~exist('stats','var'),
      stats=1000;
    end
    orthoit=orthoit*stocheck;
    minI=minI*stocheck;
    maxI=maxI*stocheck;
    %onesS=ones(stochsamples,1);
    %overS=1/stochsamples;
    stats=stats*stocheck;
    %ind3=[1:stochsamples]';
    %ind3=ind3(:,onesNp);
    %ind3=ind3(:);
    %ind4=1:Np;
    %ind4=ind4(onesS,:);
    %ind4=ind4(:);
  else
    if ~exist('stats','var'),
      stats=1;
    end
  end

  ind1=[1:Nx]';
  ind1=ind1(:,onesNp);
  ind1=ind1(:);

  ind2=1:Np;
  ind2=ind2(onesNx,:);
  ind2=ind2(:);

  work.slope=slope;
  work.ind1=ind1;
  work.ind2=ind2;
  work.c2faster=c2faster;
  work.auctypeA=auctypeA;
  
  %sel=Plabels(:,onesNx)'==Xlabels(:,onesNp);

  if devel,
    ind1=[1:Ny]';
    ind1=ind1(:,onesNp);
    ind1=ind1(:);

    ind2=1:Np;
    ind2=ind2(onesNy,:);
    ind2=ind2(:);

    dwork.slope=slope;
    dwork.ind1=ind1;
    dwork.ind2=ind2;
    dwork.c2faster=c2faster;
    dwork.auctypeA=auctypeA;
  end

  clear onesNx onesNy;
  clear ind1 ind2;

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
  probecfg.slope=slope;
  probecfg.stats=stats;
  probecfg.orthoit=orthoit;
  probecfg.orthonormal=orthonormal;
  probecfg.orthogonal=orthogonal;
  probecfg.euclidean=euclidean;
  probecfg.cosine=cosine;
  probecfg.stochastic=stochastic;
  %probecfg.prior=prior;
  probecfg.cfact=cfact;
  probecfg.ind1=ind1;
  probecfg.ind2=ind2;
  %probecfg.sel=sel;
  if stochastic,
    probecfg.onesNp=onesNp;
    probecfg.cumprior=cumprior;
    probecfg.nc=nc;
    probecfg.cnc=cnc;
  end
  bestIJA=[0,1];
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
        if I>bestIJA(1) || (I==bestIJA(1) && J<bestIJA(2)),
          bestIJA=[I,J];
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
  fprintf(logfile,'%s selected rates={%.2E %.2E} impI=%.2f J=%.4f\n',fn,rateB,rateP,bestIJA(1)/probeI,bestIJA(2));
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
bestIJA=[0 0 0 -1];

if adaptrates,
  rateB=rateB*ones(D,R);
  rateP=rateP*ones(D,Np);
  Bv=zeros(D,R);
  Pv=zeros(D,Np);
  if orthonormal || orthogonal,
    prevB=Bi;
  end
end

J00=0;
J0=0;
I=0;

fprintf(logfile,'%s D=%d C=%d R=%d Nx=%d\n',fn,D,C,R,Nx);
fprintf(logfile,'%s output: iteration | J | delta(J) | AUC\n',fn);

if ~probemode,
  tic;
end

%%% Batch gradient descent %%%
if ~stochastic,

  while true,

    %%% Compute statistics %%%
    [A, J, fX, fP] = ppma_index(Bi'*Pi, Plabels, Bi'*X, Xlabels, work);
    if devel,
      A = ppma_index(Bi'*Pi, Plabels, Bi'*Y, Ylabels, dwork);
    end

    %%% Determine if there was improvement %%%
    mark='';
    if (~testA && J>=bestIJA(2)) || ...
       ( testA && A>=bestIJA(3)),
      bestB=Bi;
      bestP=Pi;
      bestIJA=[I J A bestIJA(4)+1];
      mark=' *';
    end

    %%% Print statistics %%%
    if mod(I,stats)==0,
      fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J00,A,mark);
      J00=J;
    end

    %%% Determine if algorithm has to be stopped %%%
    if probemode,
      if bestIJA(4)+(maxI-I)<probeunstable,
        break;
      end
    end
    if I>=maxI || ~isfinite(J) || ~isfinite(A) || (I>=minI && abs(J-J0)<epsilon),
      fprintf(logfile,'%s stopped iterating, ',fn);
      if I>=maxI,
        fprintf(logfile,'reached maximum number of iterations\n');
      elseif ~isfinite(J) || ~isfinite(A),
        fprintf(logfile,'reached unstable state\n');
      else
        fprintf(logfile,'index has stabilized\n');
      end
      break;
    end

    J0=J;
    I=I+1;

    %%% Update parameters %%%
    Bi=Bi+rateB.*(X*fX'+Pi*fP');
    Pi=Pi+rateP.*(Bi*fP);

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

    if mod(I,stocheck)==0 && stocheckfull,
      [A, J] = ppma_index(Bi'*P, Plabels, Bi'*X, Xlabels, work);
    end

    if auctypeA,
      c=round((C-1)*rand(stochsamples,1))+1;
      n1=cnc(c)+round((nc(c)-1).*rand(stochsamples,1))+1;
      n2=mod(cnc(c)+nc(c)+round((Nx-nc(c)-1).*rand(stochsamples,1)),Nx)+1;
    else
      c1=round((C-1)*rand(stochsamples,1))+1;
      c2=mod(c1+round((C-2).*rand(stochsamples,1)),C)+1;
      n1=cnc(c1)+round((nc(c1)-1).*rand(stochsamples,1))+1;
      n2=cnc(c2)+round((nc(c2)-1).*rand(stochsamples,1))+1;
    end
    
    %??????????????????    

    %%% Compute statistics %%%
    if ~stocheckfull,
      [A, J, fX, fP] = ppma_index(Bi'*P, Plabels, Bi'*X, Xlabels, work);
      J=0.5*(0.9*prevJ+J);
      prevJ=J;
      if ~devel,
        E=0;
        for c=1:C,
          csel=randc==c;
          E=E+prior(c)*sum(dd(csel)<ds(csel))/max(sum(csel),1);
        end
        E=0.5*(0.9*prevE+E);
        prevE=E;
      end
    end

    if mod(I,stocheck)==0,
      if ~stocheckfull && devel,
        E=classify_nn(rP,Plabels,Bi'*Y,Ylabels,'prior',prior);
      end

      %%% Determine if there was improvement %%%
      mark='';
      if (~testA && J>=bestIJA(2)) || ...
         ( testA && A>=bestIJA(3)),
        bestB=Bi;
        bestP=Pi;
        bestIJA=[I J E bestIJA(4)+1];
        mark=' *';
      end

      %%% Print statistics %%%
      if mod(I,stats)==0,
        fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J00,E,mark);
        J00=J;
      end

      %%% Determine if algorithm has to be stopped %%%
      if probemode,
        if bestIJA(4)+(maxI-I)<probeunstable,
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
    Bi=Bi+rateB.*(X*fX'+Pi*fP');
    Pi=Pi+rateP.*(Bi*fP);

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
    [A, J] = ppma_index(bestB'*bestP, Plabels, bestB'*X, Xlabels, work);

    fprintf(logfile,'%s best iteration approx: I=%d J=%f A=%f\n',fn,bestIJA(1),bestIJA(2),bestIJA(3));
    bestIJA(2)=J;
    bestIJA(3)=A;
  end % if stochfinalexact
end

if ~probemode,
  tm=toc;
  fprintf(logfile,'%s best iteration: I=%d J=%f A=%f\n',fn,bestIJA(1),bestIJA(2),bestIJA(3));
  fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJA(4)/max(I,1));
  fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
  fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);
end

if ~verbose,
  fclose(logfile);
end

if probemode,
  bestB=bestIJA(4);
  bestP=bestIJA(2);
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                   Helper functions                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, J, fX, fP] = ppma_index(P, Plabels, X, Xlabels, work)

  [R,Np]=size(P);
  Nx=size(X,2);
  C=max(size(unique(Plabels)));

  ind1=work.ind1;
  ind2=work.ind2;
  slope=work.slope;
  auctypeA=work.auctypeA;
  c2faster=work.c2faster;

  d=reshape(sum((X(:,ind1)-P(:,ind2)).^2,1),Nx,Np);

  dist=zeros(Nx,C);
  indx=zeros(Nx,C);
  for c=1:C,
    csel=Plabels==c;
    [dist(:,c),indx(:,c)]=min(d(:,csel),[],2);
    asel=find(csel);
    indx(:,c)=asel(indx(:,c));
  end

  J=0;
  A=0;

  grad=false;
  if nargin>2,
    fX=zeros(R,Nx);
    fP=zeros(R,Np);
    grad=true;
  end

  %%% c against the rest %%%
  if auctypeA, % opA

    CC=C;
    if C==2 && c2faster,
      CC=1;
    end
    CC

    %for c=1:C,
    for c=1:CC,
      ds=dist(:,c);
      csel=([1:C]~=c)';
      [dd,id]=min(dist(:,csel),[],2);
      ns=Xlabels==c;
      nd=Xlabels~=c;
      POS=dd(ns)./(dd(ns)+ds(ns));
      NEG=dd(nd)./(dd(nd)+ds(nd));
      ePOS=exp(-slope*POS);
      eNEG=exp(slope*NEG);
      A=A+auc(POS,NEG);
      Jc=0;
      NP=size(POS,1);
      if grad,
        fact=1/(sum(ns)*sum(nd)*C);
        ns=find(ns);
        nd=find(nd);
        is=indx(:,c);
        csel=Nx*(find(csel)-1);
        id=indx([1:Nx]'+csel(id));
        %id=min(indx([1:C]~=c,:)); % this is wrong ?
        uidnd=unique(id(nd))';
        uisnd=unique(is(nd))';
      end
      for n=1:NP,
        expon=eNEG.*ePOS(n);
        Jc=Jc+sum(1./(1+expon));
        if grad,
          dsigm=slope.*expon./((1+expon).^2);
          factPOS=fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)));
          factNEG=repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)';
          ONEmPOS=1-POS(n);
          rONEmNEG=repmat(1-NEG,1,R)';
          rNEG=repmat(NEG,1,R)';
          dXPidnsn=X(:,ns(n))-P(:,id(ns(n)));
          dXPisnsn=X(:,ns(n))-P(:,is(ns(n)));
          dXPidnd=X(:,nd)-P(:,id(nd));
          dXPisnd=X(:,nd)-P(:,is(nd));
          fX(:,ns(n))=fX(:,ns(n))+factPOS.*(ONEmPOS .*dXPidnsn-POS(n).*dXPisnsn);
          fX(:,nd)   =fX(:,nd)   -factNEG.*(rONEmNEG.*dXPidnd -rNEG  .*dXPisnd );
          fP(:,id(ns(n)))=fP(:,id(ns(n)))-factPOS.*ONEmPOS.*dXPidnsn;
          fP(:,is(ns(n)))=fP(:,is(ns(n)))+factPOS.*POS(n) .*dXPisnsn;
          for idnd=uidnd,
            fP(:,idnd)=fP(:,idnd)+sum(factNEG.*rONEmNEG.*dXPidnd,2);
          end
          for isnd=uisnd,
            fP(:,isnd)=fP(:,isnd)-sum(factNEG.*rNEG    .*dXPisnd,2);
          end
        end
      end
      %for n=1:NP,
      %  expon=eNEG.*ePOS(n);
      %  Jc=Jc+sum(1./(1+expon));
      %  dsigm=slope.*expon./((1+expon).^2);
      %  fX(:,ns(n))=fX(:,ns(n))+(fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)))).*((1-POS(n))        .*(X(:,ns(n))-P(:,id(ns(n))))-POS(n)          .*(X(:,ns(n))-P(:,is(ns(n)))));
      %  fX(:,nd)   =fX(:,nd)   -repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)'.*(repmat(1-NEG,1,R)'.*(X(:,nd)   -P(:,id(nd)))   -repmat(NEG,1,R)'.*(X(:,nd)   -P(:,is(nd))));   
      %  fP(:,id(ns(n)))=fP(:,id(ns(n)))-(fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)))).*(1-POS(n)) .*(X(:,ns(n))-P(:,id(ns(n))));
      %  fP(:,is(ns(n)))=fP(:,is(ns(n)))+(fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)))).*POS(n)     .*(X(:,ns(n))-P(:,is(ns(n))));
      %  for idnd=unique(id(nd))',
      %    fP(:,idnd)=fP(:,idnd)+sum(repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)'.*repmat(1-NEG,1,R)'.*(X(:,nd)-P(:,id(nd))),2);
      %  end
      %  for isnd=unique(is(nd))',
      %    fP(:,isnd)=fP(:,isnd)-sum(repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)'.*repmat(NEG,1,R)'  .*(X(:,nd)-P(:,is(nd))),2);
      %  end
      %end
      J=J+Jc/(NP*(Nx-NP));
    end
    if ~(C==2 && c2faster),
      A=A/C;
      J=J/C;
    elseif grad,
      fX=fX+fX;
      fP=fP+fP;
    end

  %%% c against cc %%% dhand
  else % opB

    for c=1:C,
      for cc=1:C,
        if c~=cc,
          ds=dist(:,c);
          dd=dist(:,cc);
          ns=Xlabels==c;
          nd=Xlabels==cc;
          POS=dd(ns)./(dd(ns)+ds(ns));
          NEG=dd(nd)./(dd(nd)+ds(nd));
          ePOS=exp(-slope*POS);
          eNEG=exp(slope*NEG);
          A=A+auc(POS,NEG);
          Jc=0;
          NP=size(POS,1);
          NN=size(NEG,1);
          if grad,
            fact=1/(sum(ns)*sum(nd)*C*(C-1));
            ns=find(ns);
            nd=find(nd);
            is=indx(:,c);
            id=indx(:,cc);
            uidnd=unique(id(nd))';
            uisnd=unique(is(nd))';
          end
          for n=1:NP,
            expon=eNEG.*ePOS(n);
            Jc=Jc+sum(1./(1+expon));
            if grad,
              dsigm=slope.*expon./((1+expon).^2);
              factPOS=fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)));
              factNEG=repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)';
              ONEmPOS=1-POS(n);
              rONEmNEG=repmat(1-NEG,1,R)';
              rNEG=repmat(NEG,1,R)';
              dXPidnsn=X(:,ns(n))-P(:,id(ns(n)));
              dXPisnsn=X(:,ns(n))-P(:,is(ns(n)));
              dXPidnd=X(:,nd)-P(:,id(nd));
              dXPisnd=X(:,nd)-P(:,is(nd));
              fX(:,ns(n))=fX(:,ns(n))+factPOS.*(ONEmPOS .*dXPidnsn-POS(n).*dXPisnsn);
              fX(:,nd)   =fX(:,nd)   -factNEG.*(rONEmNEG.*dXPidnd -rNEG  .*dXPisnd );
              fP(:,id(ns(n)))=fP(:,id(ns(n)))-factPOS.*ONEmPOS.*dXPidnsn;
              fP(:,is(ns(n)))=fP(:,is(ns(n)))+factPOS.*POS(n) .*dXPisnsn;
              for idnd=uidnd,
                fP(:,idnd)=fP(:,idnd)+sum(factNEG.*rONEmNEG.*dXPidnd,2);
              end
              for isnd=uisnd,
                fP(:,isnd)=fP(:,isnd)-sum(factNEG.*rNEG    .*dXPisnd,2);
              end
            end
          end
          %for n=1:NP,
          %  expon=eNEG.*ePOS(n);
          %  Jc=Jc+sum(1./(1+expon));
          %  dsigm=slope.*expon./((1+expon).^2);
          %  fX(:,ns(n))=fX(:,ns(n))+(fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)))).*((1-POS(n))        .*(X(:,ns(n))-P(:,id(ns(n))))-POS(n)          .*(X(:,ns(n))-P(:,is(ns(n)))));
          %  fX(:,nd)   =fX(:,nd)   -repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)'.*(repmat(1-NEG,1,R)'.*(X(:,nd)   -P(:,id(nd)))   -repmat(NEG,1,R)'.*(X(:,nd)   -P(:,is(nd))));   
          %  fP(:,id(ns(n)))=fP(:,id(ns(n)))-(fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)))).*(1-POS(n)) .*(X(:,ns(n))-P(:,id(ns(n))));
          %  fP(:,is(ns(n)))=fP(:,is(ns(n)))+(fact.*sum(dsigm)./(ds(ns(n))+dd(ns(n)))).*POS(n)     .*(X(:,ns(n))-P(:,is(ns(n))));
          %  for idnd=unique(id(nd))',
          %    fP(:,idnd)=fP(:,idnd)+sum(repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)'.*repmat(1-NEG,1,R)'.*(X(:,nd)-P(:,id(nd))),2);
          %  end
          %  for isnd=unique(is(nd))',
          %    fP(:,isnd)=fP(:,isnd)-sum(repmat(fact.*dsigm./(ds(nd)+dd(nd)),1,R)'.*repmat(NEG,1,R)'  .*(X(:,nd)-P(:,is(nd))),2);
          %  end
          %end
          J=J+Jc/(NP*NN);
        end
      end
    end
    A=A/(C*(C-1));
    J=J/(C*(C-1));

  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, J, fX, fP] = ppma_sindex(P, Plabels, X, Xlabels, work)

  [R,Np]=size(P);
  Nx=size(X,2);
  C=max(size(unique(Plabels)));
