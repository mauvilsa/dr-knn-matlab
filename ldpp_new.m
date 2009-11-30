function [bestB, bestP] = ldpp_new(X, Xlabels, B0, P0, Plabels, varargin)
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
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'prior',PRIOR              - A priori probabilities (default=Nc/N)
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
% Reference:
%
%   M. Villegas and R. Paredes. "Simultaneous Learning of a Discriminative
%   Projection and Prototypes for Nearest-Neighbor Classification."
%   CVPR'2008.
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
  unix('echo "$Revision: 70 $* $Date: 2009-10-19 09:34:45 +0200 (Mon, 19 Oct 2009) $*" | sed "s/^:/ldpp: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
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
testE=true;

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
         strcmp(varargin{n},'prior') || ...
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
  prior=probemode.prior;
  cfact=probemode.cfact;
  jfact=probemode.jfact;
  ind1=probemode.ind1;
  ind2=probemode.ind2;
  sel=probemode.sel;
  devel=probemode.devel;
  if devel,
    Y=probemode.Y;
    Ylabels=probemode.Ylabels;
  end
  if stochastic,
    onesNp=probemode.onesNp;
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

  jfact=1;
  cfact=zeros(Nx,1);
  for c=1:C,
    cfact(Xlabels==c)=prior(c)/sum(Xlabels==c);
  end
  if euclidean,
    cfact=2*cfact;
    jfact=0.5;
  end

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
    onesS=ones(stochsamples,1);
    overS=1/stochsamples;
    stats=stats*stocheck;
    ind3=[1:stochsamples]';
    ind3=ind3(:,onesNp);
    ind3=ind3(:);
    ind4=1:Np;
    ind4=ind4(onesS,:);
    ind4=ind4(:);
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

  sel=Plabels(:,onesNx)'==Xlabels(:,onesNp);

  clear onesNx onesNy;

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
  probecfg.prior=prior;
  probecfg.cfact=cfact;
  probecfg.ind1=ind1;
  probecfg.ind2=ind2;
  probecfg.sel=sel;
  if stochastic,
    probecfg.onesNp=onesNp;
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

if adaptrates,
  rateB=rateB*ones(D,R);
  rateP=rateP*ones(D,Np);
  Bv=zeros(D,R);
  Pv=zeros(D,Np);
  if orthonormal || orthogonal,
    prevB=Bi;
  end
end

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

    %%% Project data %%%
    rX=Bi'*X;
    rP=Bi'*Pi;

    %%% Compute distances %%%
    if euclidean,
      ds=reshape(sum((rX(:,ind1)-rP(:,ind2)).^2,1),Nx,Np);
    elseif cosine,
      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(onesR,:);
      rxsd=sqrt(sum(rX.*rX,1));
      rX=rX./rxsd(onesR,:);
      ds=reshape(1-sum(rX(:,ind1).*rP(:,ind2),1),Nx,Np);
    end
      ds=reshape(sum((rX(:,ind1)-rP(:,ind2)).^2,1),Nx,Np);
    dd=ds;
    ds(~sel)=inf;
    dd(sel)=inf;
    [ds,is]=min(ds,[],2);
    [dd,id]=min(dd,[],2);
    ds(ds==0)=realmin;
    dd(dd==0)=realmin;
    ratio=ds./dd;
    expon=exp(slope*(1-ratio));
    sigm=1./(1+expon);

    %%% Compute statistics %%%
    J=jfact*sum(cfact.*sigm);
    if ~devel,
      E=0;
      for c=1:C,
        E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
      end
    else
      E=classify_nn(rP,Plabels,Bi'*Y,Ylabels,'prior',prior);
    end

    %%% Determine if there was improvement %%%
    mark='';
    if (~testE && J<=bestIJE(2)) || ...
       ( testE && E<=bestIJE(3)),
      bestB=Bi;
      bestP=Pi;
      bestIJE=[I J E bestIJE(4)+1];
      mark=' *';
    end

    %%% Print statistics %%%
    if mod(I,stats)==0,
      fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J00,E,mark);
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

    %%% Compute gradient %%%
    dsigm=slope.*expon./((1+expon).*(1+expon));
    ratio=cfact.*ratio;
    dfact=ratio.*dsigm;
    sfact=dfact./ds;
    dfact=dfact./dd;

    if euclidean,
      rXs=(rX-rP(:,is)).*sfact(:,onesR)';
      rXd=(rX-rP(:,id)).*dfact(:,onesR)';
      fX=rXs-rXd;
      for m=1:Np,
        fP(:,m)=-sum(rXs(:,is==m),2)+sum(rXd(:,id==m),2);
      end
    elseif cosine,
      rXs=rX.*sfact(:,onesR)';
      rXd=rX.*dfact(:,onesR)';
      fX=rP(:,id).*dfact(:,onesR)'-rP(:,is).*sfact(:,onesR)';
      for m=1:Np,
        fP(:,m)=-sum(rXs(:,is==m),2)+sum(rXd(:,id==m),2);
      end
    end

    P0=Bi*fP;
    B0=X*fX'+Pi*fP';

    %%% Update parameters %%%
    if adaptrates,
      rateB=rateB.*max(0.5,1+adaptrate*Bv.*B0);
      rateP=rateP.*max(0.5,1+adaptrate*Pv.*P0);
    end
    Bi=Bi-rateB.*B0;
    Pi=Pi-rateP.*P0;

    %%% Parameter constraints %%%
    if mod(I,orthoit)==0,
      if orthonormal,
        Bi=orthonorm(Bi);
      elseif orthogonal,
        Bi=orthounit(Bi);
      end
    end

    %%% Adapt learning rates %%%
    if adaptrates,
      rXhb=Bv'*X;
      rPhb=Bv'*Pi;
      rPhp=Bi'*Pv;
      if euclidean,
        dshb=2*sum((rXhb-rPhb(:,is)).^2,1)';
        ddhb=2*sum((rXhb-rPhb(:,id)).^2,1)';
        dshp=-2*sum((rX-rP(:,is)).*rPhp(:,is))';
        ddhp=-2*sum((rX-rP(:,id)).*rPhp(:,id))';
      end
      dshb=dshb./ds;
      ddhb=ddhb./dd;
      dshp=dshp./ds;
      ddhp=ddhp./dd;

      ddsigm=2*dsigm.*dsigm./sigm-slope.*dsigm;

      if euclidean,
        sfact=ratio./ds;
        dfact=ratio./dd;
        sfact1=sfact.*dsigm;
        dfact1=dfact.*dsigm;
        ratio=(dsigm+ddsigm).*(dshb-ddhb);
        sfact2=sfact.*(ratio-dsigm.*dshb);
        dfact2=dfact.*(ratio-dsigm.*ddhb);
        rXs=sfact1(:,onesR)'.*(rXhb-rPhb(:,is))+sfact2(:,onesR)'.*(rX-rP(:,is));
        rXd=dfact1(:,onesR)'.*(rXhb-rPhb(:,id))+dfact2(:,onesR)'.*(rX-rP(:,id));

        fX=rXs-rXd;
        for m=1:Np,
          fP(:,m)=-sum(rXs(:,is==m),2)+sum(rXd(:,id==m),2);
        end

        HBv=X*fX'+Pi*fP';

        ratio=(dsigm+ddsigm).*(dshp-ddhp);
        sfact2=sfact.*(ratio-dsigm.*dshp);
        dfact2=dfact.*(ratio-dsigm.*ddhp);
        rXs=sfact1(:,onesR)'.*rPhp(:,is)-sfact2(:,onesR)'.*(rX-rP(:,is));
        rXd=dfact1(:,onesR)'.*rPhp(:,id)-dfact2(:,onesR)'.*(rX-rP(:,id));

        for m=1:Np,
          fP(:,m)=sum(rXs(:,is==m),2)-sum(rXd(:,id==m),2);
        end

        HPv=Bi*fP;
      end % if euclidean

      if (orthonormal || orthogonal) && mod(I,orthoit)==0,
        B0=(Bi-prevB)./rateB;
      end

      Bv=adaptdecay*Bv+rateB.*(B0-adaptdecay.*HBv);
      Pv=adaptdecay*Pv+rateP.*(P0-adaptdecay.*HPv);

      if (orthonormal || orthogonal) && mod(I,orthoit)==0,
        prevB=Bi;
      end
    end % if adaptrates
  end % while true

%%% Stochasitc gradient descent %%%
else

  prevJ=1;
  prevE=1;

  while true,

    if mod(I,stocheck)==0 && stocheckfull,
      rX=Bi'*X;
      rP=Bi'*Pi;

      if euclidean,
        ds=reshape(sum((rX(:,ind1)-rP(:,ind2)).^2,1),Nx,Np);
      elseif cosine,
        rpsd=sqrt(sum(rP.*rP,1));
        rP=rP./rpsd(onesR,:);
        rxsd=sqrt(sum(rX.*rX,1));
        rX=rX./rxsd(onesR,:);
        ds=reshape(1-sum(rX(:,ind1).*rP(:,ind2),1),Nx,Np);
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
      sigm=1./(1+expon);

      J=jfact*sum(cfact.*sigm);
      if ~devel,
        E=0;
        for c=1:C,
          csel=Xlabels==c;
          E=E+prior(c)*sum(dd(csel)<ds(csel))/sum(csel);
        end
      else
        E=classify_nn(rP,Plabels,Bi'*Y,Ylabels,'prior',prior);
      end
    end % if mod(I,stocheck)==0 && stocheckfull

    %%% Select random samples %%%
    rands=rand(1,stochsamples);
    randc=(sum(rands(onesC,:)>cumprior(:,onesS))+1)';
    randn=cnc(randc)+round((nc(randc)-1).*rand(stochsamples,1))+1;

    %%% Project data %%%
    sX=X(:,randn);
    rX=Bi'*sX;
    rP=Bi'*Pi;

    %%% Compute distances %%%
    if euclidean,
      ds=reshape(sum((rX(:,ind3)-rP(:,ind4)).^2,1),stochsamples,Np);
    elseif cosine,
      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(onesR,:);
      rxsd=sqrt(sum(rX.*rX,1));
      rX=rX./rxsd(onesR,:);
      ds=reshape(1-sum(rX(:,ind3).*rP(:,ind4),1),stochsamples,Np);
    end
    dd=ds;
    ssel=Plabels(:,onesS)'==randc(:,onesNp);
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
    if ~stocheckfull,
      J=overS*sum(sigm);
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
      if (~testE && J<=bestIJE(2)) || ...
         ( testE && E<=bestIJE(3)),
        bestB=Bi;
        bestP=Pi;
        bestIJE=[I J E bestIJE(4)+1];
        mark=' *';
      end

      %%% Print statistics %%%
      if mod(I,stats)==0,
        fprintf(logfile,'%d\t%.9f\t%.9f\t%f%s\n',I,J,J-J00,E,mark);
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

    %%% Compute gradient %%%
    dsigm=slope.*expon./((1+expon).*(1+expon));
    ratio=overS.*ratio;
    dfact=ratio.*dsigm;
    sfact=dfact./ds;
    dfact=dfact./dd;

    if euclidean,
      rXs=(rX-rP(:,is)).*sfact(:,onesR)';
      rXd=(rX-rP(:,id)).*dfact(:,onesR)';
      fX=rXs-rXd;
      for m=1:Np,
        fP(:,m)=-sum(rXs(:,is==m),2)+sum(rXd(:,id==m),2);
      end
    elseif cosine,
      rXs=rX.*sfact(:,onesR)';
      rXd=rX.*dfact(:,onesR)';
      fX=rP(:,id).*dfact(:,onesR)'-rP(:,is).*sfact(:,onesR)';
      for m=1:Np,
        fP(:,m)=-sum(rXs(:,is==m),2)+sum(rXd(:,id==m),2);
      end
    end

    P0=Bi*fP;
    B0=sX*fX'+Pi*fP';

    %%% Update parameters %%%
    if adaptrates,
      rateB=rateB.*max(0.5,1+adaptrate*Bv.*B0);
      rateP=rateP.*max(0.5,1+adaptrate*Pv.*P0);
    end
    Bi=Bi-rateB.*B0;
    Pi=Pi-rateP.*P0;

    %%% Parameter constraints %%%
    if mod(I,orthoit)==0,
      if orthonormal,
        Bi=orthonorm(Bi);
      elseif orthogonal,
        Bi=orthounit(Bi);
      end
    end

    %%% Adapt learning rates %%%
    if adaptrates,
      rXhb=Bv'*sX;
      rPhb=Bv'*Pi;
      rPhp=Bi'*Pv;
      if euclidean,
        dshb=2*sum((rXhb-rPhb(:,is)).^2,1)';
        ddhb=2*sum((rXhb-rPhb(:,id)).^2,1)';
        dshp=-2*sum((rX-rP(:,is)).*rPhp(:,is))';
        ddhp=-2*sum((rX-rP(:,id)).*rPhp(:,id))';
      end
      dshb=dshb./ds;
      ddhb=ddhb./dd;
      dshp=dshp./ds;
      ddhp=ddhp./dd;

      ddsigm=2*dsigm.*dsigm./sigm-slope.*dsigm;

      if euclidean,
        sfact=ratio./ds;
        dfact=ratio./dd;
        sfact1=sfact.*dsigm;
        dfact1=dfact.*dsigm;
        ratio=(dsigm+ddsigm).*(dshb-ddhb);
        sfact2=sfact.*(ratio-dsigm.*dshb);
        dfact2=dfact.*(ratio-dsigm.*ddhb);
        rXs=sfact1(:,onesR)'.*(rXhb-rPhb(:,is))+sfact2(:,onesR)'.*(rX-rP(:,is));
        rXd=dfact1(:,onesR)'.*(rXhb-rPhb(:,id))+dfact2(:,onesR)'.*(rX-rP(:,id));

        fX=rXs-rXd;
        for m=1:Np,
          fP(:,m)=-sum(rXs(:,is==m),2)+sum(rXd(:,id==m),2);
        end

        HBv=sX*fX'+Pi*fP';

        ratio=(dsigm+ddsigm).*(dshp-ddhp);
        sfact2=sfact.*(ratio-dsigm.*dshp);
        dfact2=dfact.*(ratio-dsigm.*ddhp);
        rXs=sfact1(:,onesR)'.*rPhp(:,is)-sfact2(:,onesR)'.*(rX-rP(:,is));
        rXd=dfact1(:,onesR)'.*rPhp(:,id)-dfact2(:,onesR)'.*(rX-rP(:,id));

        for m=1:Np,
          fP(:,m)=sum(rXs(:,is==m),2)-sum(rXd(:,id==m),2);
        end

        HPv=Bi*fP;
      end % if euclidean

      if (orthonormal || orthogonal) && mod(I,orthoit)==0,
        B0=(Bi-prevB)./rateB;
      end

      Bv=adaptdecay*Bv+rateB.*(B0-adaptdecay.*HBv);
      Pv=adaptdecay*Pv+rateP.*(P0-adaptdecay.*HPv);

      if (orthonormal || orthogonal) && mod(I,orthoit)==0,
        prevB=Bi;
      end
    end % if adaptrates    
  end % while true

  %%% Parameter constraints %%%
  if orthonormal,
    bestB=orthonorm(bestB);
  elseif orthogonal,
    bestB=orthounit(bestB);
  end

  %%% Compute final statistics %%%
  if stochfinalexact && ~stocheckfull,
    rX=bestB'*X;
    rP=bestB'*bestP;

    if euclidean,
      ds=reshape(sum((rX(:,ind1)-rP(:,ind2)).^2,1),Nx,Np);
    elseif cosine,
      rpsd=sqrt(sum(rP.*rP,1));
      rP=rP./rpsd(onesR,:);
      rxsd=sqrt(sum(rX.*rX,1));
      rX=rX./rxsd(onesR,:);
      ds=reshape(1-sum(rX(:,ind1).*rP(:,ind2),1),Nx,Np);
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
    sigm=1./(1+expon);

    J=jfact*sum(cfact.*sigm);
    if ~devel,
      E=0;
      for c=1:C,
        E=E+prior(c)*sum(dd(Xlabels==c)<ds(Xlabels==c))/sum(Xlabels==c);
      end
    else
      E=classify_nn(rP,Plabels,Bi'*Y,Ylabels,'prior',prior);
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
