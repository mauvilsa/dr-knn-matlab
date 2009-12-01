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
%   'devel',Y,Ylabels          - Set the development set (default=false)
%   'seed',SEED                - Random seed (default=system)
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=1)
%   'stocheck',SIT             - Check every SIT stochastic iterations (default=100)
%   'stocheckfull',(true|f...  - Stats for complete data set (default=false)
%   'stochfinalexact',(tru...  - Final stats for complete data set (default=true)
%   'stats',STAT               - Statistics every STAT (default={b:1,s:1000})
%   'orthoit',OIT              - Orthogonalize every OIT (default=1)
%   'orthonormal',(true|false) - Orthonormal projection base (default=true)
%   'orthogonal',(true|false)  - Orthogonal projection base (default=false)
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
%euclidean=true;
%cosine=false;
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
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(sum(varargin{n+1}<0))~=0,
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'orthonormal') || ...
         strcmp(varargin{n},'orthogonal') || ...
         strcmp(varargin{n},'normalize') || ...
         strcmp(varargin{n},'linearnorm') || ...
         strcmp(varargin{n},'whiten') || ...
         strcmp(varargin{n},'stochastic') || ...
         strcmp(varargin{n},'stocheckfull') || ...
         strcmp(varargin{n},'stochfinalexact') || ...
         strcmp(varargin{n},'autoprobe') || ...
         strcmp(varargin{n},'adaptrates') || ...
         strcmp(varargin{n},'c2faster') || ...
         strcmp(varargin{n},'auctypeA') || ...
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
        %elseif strcmp(varargin{n},'euclidean'),
        %  cosine=false;
        %elseif strcmp(varargin{n},'cosine'),
        %  euclidean=false;
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
  rateB=probemode.rateB;
  rateP=probemode.rateP;
  minI=probemode.minI;
  maxI=probemode.maxI;
  probeunstable=minI;
  stats=probemode.stats;
  orthoit=probemode.orthoit;
  orthonormal=probemode.orthonormal;
  orthogonal=probemode.orthogonal;
  %euclidean=probemode.euclidean;
  %cosine=probemode.cosine;
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
    %if euclidean,
      xsd=R*xsd;
    %end
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
    %if euclidean,
      W=(1/R).*W;
    %end
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

  rfact=1;
  %if euclidean,
    rfact=2;
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
    stats=stats*stocheck;
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
  work.C=C;

  if stochastic,
    c1=1:C;
    c1=c1(ones(stochsamples,1),:);
    c1=c1(:);
    stochsamples=C*stochsamples;

    ind1=[1:2*stochsamples]';
    ind1=ind1(:,onesNp);
    ind1=ind1(:);

    ind2=1:Np;
    ind2=ind2(ones(2*stochsamples,1),:);
    ind2=ind2(:);

    swork.slope=slope;
    swork.ind1=ind1;
    swork.ind2=ind2;
    swork.auctypeA=auctypeA;
    swork.C=C;
    swork.onesR=onesR;
  end

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
    dwork.C=C;
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
  probecfg.minI=round(probeunstable*probeI);
  probecfg.maxI=probeI;
  probecfg.stats=stats;
  probecfg.orthoit=orthoit;
  probecfg.orthonormal=orthonormal;
  probecfg.orthogonal=orthogonal;
  %probecfg.euclidean=euclidean;
  %probecfg.cosine=cosine;
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
    probecfg.cumprior=cumprior;
    probecfg.nc=nc;
    probecfg.cnc=cnc;
  end
  bestIJA=[0,0];
  ratesB=unique(probe(1,probe(1,:)>=0));
  ratesP=unique(probe(2,probe(2,:)>=0));
  nB=1;
  while nB<=size(ratesB,2),
    nP=1;
    while nP<=size(ratesP,2),
      if ~(ratesB(nB)==0 && ratesP(nP)==0),
        probecfg.rateB=ratesB(nB);
        probecfg.rateP=ratesP(nP);
        [I,J]=ppma(X,Xlabels,B0,P0,Plabels,'probemode',probecfg);
        mark='';
        if I>bestIJA(1) || (I==bestIJA(1) && J>bestIJA(2)),
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
    [A,J,fX,fP]=ppma_index(Bi'*Pi,Plabels,Bi'*X,Xlabels,work);
    if devel,
      A=ppma_index(Bi'*Pi,Plabels,Bi'*Y,Ylabels,dwork);
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
    Bi=Bi+rfact.*rateB.*(X*fX'+Pi*fP');
    Pi=Pi+rfact.*rateP.*(Bi*fP);

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
  prevA=1;

  while true,

    if mod(I,stocheck)==0 && stocheckfull,
      [A,J]=ppma_index(Bi'*Pi,Plabels,Bi'*X,Xlabels,work);
      if devel,
        A=ppma_index(Bi'*Pi,Plabels,Bi'*Y,Ylabels,dwork);
      end
    end

    %%% Select random samples %%%
    if auctypeA,
      c2=[];
      n1=cnc(c1)+round((nc(c1)-1).*rand(stochsamples,1))+1;
      n2=mod(cnc(c1)+nc(c1)+round((Nx-nc(c1)-1).*rand(stochsamples,1)),Nx)+1;
    %else
    %  c2=mod(c1+round((C-2).*rand(stochsamples,1)),C)+1;
    %  n1=cnc(c1)+round((nc(c1)-1).*rand(stochsamples,1))+1;
    %  n2=cnc(c2)+round((nc(c2)-1).*rand(stochsamples,1))+1;
    end

    %%% Compute statistics %%%
    sX=[X(:,n1),X(:,n2)];
    [Ai,Ji,fX,fP]=ppma_sindex(Bi'*Pi,Plabels,Bi'*sX,c1,c2,swork);
    if ~stocheckfull,
      J=0.5*(0.9*prevJ+Ji);
      prevJ=J;
      if ~devel,
        A=0.5*(0.9*prevA+Ai);
        prevA=A;
      end
    end

    if mod(I,stocheck)==0,
      if ~stocheckfull && devel,
        A=ppma_index(Bi'*Pi,Plabels,Bi'*Y,Ylabels,dwork);
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
    end % if mod(I,stocheck)==0

    I=I+1;

    %%% Update parameters %%%
    Bi=Bi+rfact.*rateB.*(sX*fX'+Pi*fP');
    Pi=Pi+rfact.*rateP.*(Bi*fP);

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
    [A,J]=ppma_index(bestB'*bestP,Plabels,bestB'*X,Xlabels,work);
    if devel,
      A=ppma_index(bestB'*bestP,Plabels,bestB'*Y,Ylabels,dwork);
    end

    fprintf(logfile,'%s best iteration approx: I=%d J=%f A=%f\n',fn,bestIJA(1),bestIJA(2),bestIJA(3));
    bestIJA(2)=J;
    bestIJA(3)=A;
  end % if stochfinalexact

  bestIJA(4)=bestIJA(4)*stocheck;
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

  ind1=work.ind1;
  ind2=work.ind2;
  slope=work.slope;
  auctypeA=work.auctypeA;
  c2faster=work.c2faster;
  C=work.C;

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
    if C==2 && c2faster, % problem with fP
      CC=1;
    else
      c2faster=false;
    end

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
        fact=1/(sum(ns)*sum(nd)*CC);
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
      J=J+Jc/(NP*(Nx-NP));
    end
    if ~c2faster,
      A=A/C;
      J=J/C;
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
          J=J+Jc/(NP*NN);
        end
      end
    end
    A=A/(C*(C-1));
    J=J/(C*(C-1));

  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, J, fXY, fP] = ppma_sindex(P, Plabels, XY, Xlabels, Ylabels, work)

  [R,Np]=size(P);
  Nx=size(Xlabels,1);
  Nxy=size(XY,2);

  ind1=work.ind1;
  ind2=work.ind2;
  onesR=work.onesR;
  slope=work.slope;
  auctypeA=work.auctypeA;
  C=work.C;

  d=reshape(sum((XY(:,ind1)-P(:,ind2)).^2,1),Nxy,Np);

  %%% c against the rest %%%
  if auctypeA, % opA

    X=XY(:,1:Nx);
    Y=XY(:,Nx+1:end);
    dX=d(1:Nx,:);
    dY=d(Nx+1:end,:);

    dsX=zeros(Nx,1);
    ddX=zeros(Nx,1);
    isX=zeros(Nx,1);
    idX=zeros(Nx,1);
    dsY=zeros(Nx,1);
    ddY=zeros(Nx,1);
    isY=zeros(Nx,1);
    idY=zeros(Nx,1);
    for c=1:C,
      sel=Xlabels==c;
      psel=Plabels==c;
      asel=find(psel);
      [dsX(sel),isX(sel)]=min(dX(sel,psel),[],2);
      [dsY(sel),isY(sel)]=min(dY(sel,psel),[],2);
      isX(sel)=asel(isX(sel));
      isY(sel)=asel(isY(sel));
      psel=~psel;
      asel=find(psel);
      [ddX(sel),idX(sel)]=min(dX(sel,psel),[],2);
      [ddY(sel),idY(sel)]=min(dY(sel,psel),[],2);
      idX(sel)=asel(idX(sel));
      idY(sel)=asel(idY(sel));
    end

    POS=ddX./(ddX+dsX);
    NEG=ddY./(ddY+dsY);
    expon=exp(slope*(NEG-POS));

    A=(sum(POS>NEG)+0.5*sum(POS==NEG))/Nx;
    J=(sum(1./(1+expon)))/Nx;

    dsigm=slope.*expon./(Nx*(1+expon).^2);
    factPOS=dsigm./(dsX+ddX);
    factNEG=dsigm./(dsY+ddY);
    fPOS=factPOS.*POS;
    fNEG=factNEG.*NEG;
    fONEmPOS=factPOS-fPOS;
    fONEmNEG=factNEG-fNEG;
    fddXP=fONEmPOS(:,onesR)'.*(X-P(:,idX));
    fdsXP=    fPOS(:,onesR)'.*(X-P(:,isX));
    fddYP=fONEmNEG(:,onesR)'.*(Y-P(:,idY));
    fdsYP=    fNEG(:,onesR)'.*(Y-P(:,isY));
    fXY=[fddXP-fdsXP,fdsYP-fddYP];
    fP=zeros(R,Np);
    for n=1:Np
      fP(:,n)=-sum(fddXP(:,idX==n),2) ...
              +sum(fdsXP(:,isX==n),2) ...
              +sum(fddYP(:,idY==n),2) ...
              -sum(fdsYP(:,isY==n),2);
    end

  %%% c against cc %%% dhand
  %else % opB
  % not implemented
  
  end
