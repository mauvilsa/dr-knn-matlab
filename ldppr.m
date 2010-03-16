function [bestB, bestP, bestPP] = ldppr(X, XX, B0, P0, PP0, varargin)
%
% LDPPR: Learning Discriminative Projections and Prototypes for Regression
%
% [B, P, PP] = ldppr(X, XX, B0, P0, PP0, ...)
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
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'stats',STAT               - Statistics every STAT (default=1)
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
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
  unix('echo "$Revision$* $Date$*" | sed "s/^:/ldppr: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

if strncmp(X,'initP',4),
  if ~exist('PP0','var'),
    [bestB, bestP] = ldppr_initP(XX, B0, P0);
  else
    [bestB, bestP] = ldppr_initP(XX, B0, P0, PP0, varargin{:});
  end
  return;
end

fn='ldppr:';
minargs=5;

%%% Default values %%%
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
indepPP=false;
testJ=false;
MAD=false;

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
         strcmp(varargin{n},'ratePP')  || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'probe') || ...
         strcmp(varargin{n},'probeI') || ...
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
         strcmp(varargin{n},'indepPP') || ...
         strcmp(varargin{n},'testJ') || ...
         strcmp(varargin{n},'MAD') || ...
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
    YY=varargin{n+2};
    n=n+3;
  else
    argerr=true;
  end
  if argerr || n>size(varargin,2),
    break;
  end
end

[D,Nx]=size(X);
DD=size(XX,1);
R=size(B0,2);
Np=size(P0,2);
if devel,
  Ny=size(Y,2);
end

if exist('probemode','var'),
  onesDD=probemode.onesDD;
  rateB=probemode.rateB;
  rateP=probemode.rateP;
  ratePP=probemode.ratePP;
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
    YY=probemode.YY;
  end
  if stochastic,
    swork=probemode.swork;
    stochsamples=probemode.stochsamples;
    stocheck=probemode.stocheck;
    stocheckfull=probemode.stocheckfull;
    stochfinalexact=probemode.stochfinalexact;
  end
  normalize=false;
  verbose=false;
  epsilon=0;
  probemode=true;
  xxsd=ones(DD,1);
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
elseif size(XX,2)~=Nx || size(PP0,2)~=Np,
  fprintf(logfile,'%s error: the number of vectors in the dependent and independent data must be the same\n',fn);
  return;
elseif size(PP0,1)~=DD,
  fprintf(logfile,'%s error: the dimensionality of the dependent variables for the data and the prototypes must be the same\n',fn);
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
  onesDD=ones(DD,1);
  if devel,
    onesNy=ones(Ny,1);
  end
  mindist=100*sqrt(1/realmax);

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

  xxmu=mean(XX,2);
  xxsd=std(XX,1,2);
  XX=(XX-xxmu(:,onesNx))./xxsd(:,onesNx);
  if devel,
    YY=(YY-xxmu(:,onesNy))./xxsd(:,onesNy);
  end
  PP0=(PP0-xxmu(:,onesNp))./xxsd(:,onesNp);
  if ~MAD,
    xxsd=xxsd.*xxsd;
  end

  %%% Stochastic preprocessing %%%
  if stochastic,
    if exist('seed','var'),
      rand('state',seed);
    end
    orthoit=orthoit*stocheck;
    minI=minI*stocheck;
    maxI=maxI*stocheck;
    stats=stats*stocheck;
  end

  %%% Initial parameter constraints %%%
  if orthonormal,
    B0=orthonorm(B0);
  elseif orthogonal,
    B0=orthounit(B0);
  end

  %%% Constant data structures %%%
  ind1=1:Np;
  ind1=ind1(onesNx,:);
  ind1=ind1(:);

  ind2=1:DD;
  ind2=ind2(onesNx,:);
  ind2=ind2(:);

  work.slope=slope;
  work.ind1=ind1;
  work.ind2=ind2;
  work.onesR=onesR;
  work.onesDD=onesDD;
  work.xxsd=xxsd;
  work.mindist=mindist;
  work.Np=Np;
  work.Nx=Nx;
  work.R=R;
  work.DD=DD;
  work.NxDD=Nx*DD;
  work.euclidean=euclidean;
  work.indepPP=indepPP;
  work.MAD=MAD;

  if stochastic,
    onesS=ones(stochsamples,1);

    ind1=1:Np;
    ind1=ind1(onesS,:);
    ind1=ind1(:);

    ind2=1:DD;
    ind2=ind2(onesS,:);
    ind2=ind2(:);

    swork.slope=slope;
    swork.ind1=ind1;
    swork.ind2=ind2;
    swork.onesR=onesR;
    swork.onesNp=onesNp;
    swork.onesDD=onesDD;
    swork.xxsd=xxsd;
    swork.mindist=mindist;
    swork.Np=Np;
    swork.Nx=stochsamples;
    swork.R=R;
    swork.DD=DD;
    swork.NxDD=stochsamples*DD;
    swork.euclidean=euclidean;
    swork.indepPP=indepPP;
    swork.MAD=MAD;
  end

  if devel,
    ind1=1:Np;
    ind1=ind1(onesNy,:);
    ind1=ind1(:);

    ind2=1:DD;
    ind2=ind2(onesNy,:);
    ind2=ind2(:);

    dwork.slope=slope;
    dwork.ind1=ind1;
    dwork.ind2=ind2;
    dwork.onesR=onesR;
    dwork.onesDD=onesDD;
    dwork.xxsd=xxsd;
    dwork.mindist=mindist;
    dwork.Np=Np;
    dwork.Nx=Ny;
    dwork.R=R;
    dwork.DD=DD;
    dwork.NxDD=Ny*DD;
    dwork.euclidean=euclidean;
    dwork.indepPP=indepPP;
    dwork.MAD=MAD;
  end

  clear onesNx onesNy;
  clear ind1 ind2;

  etype='RMSE';
  if MAD,
    etype='MAD';
  end

  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,toc);
end

if autoprobe,
  probe=[zeros(3,1),10.^[-4:4;-4:4;-4:4]];
end
if exist('probe','var'),
  tic;
  probecfg.onesDD=onesDD;
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
    probecfg.YY=YY;
  end
  if stochastic,
    probecfg.swork=swork;
    probecfg.stochsamples=stochsamples;
    probecfg.stocheck=stocheck;
    probecfg.stocheckfull=stocheckfull;
    probecfg.stochfinalexact=stochfinalexact;
  end
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
          [I,J]=ldppr(X,XX,B0,P0,PP0,'probemode',probecfg);
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
rateB=2*rateB*slope/Nx;
rateP=2*rateP*slope/Nx;
ratePP=2*ratePP*slope/Nx;

Bi=B0;
Pi=P0;
PPi=PP0;
bestB=B0;
bestP=P0;
bestPP=PP0;
bestIJE=[0 1 Inf -1];

J00=1;
J0=1;
I=0;

if ~probemode,
  fprintf(logfile,'%s Dx=%d Dxx=%d R=%d Nx=%d\n',fn,D,DD,R,Nx);
  fprintf(logfile,'%s output: iteration | J | delta(J) | %s\n',fn,etype);
  tic;
end

%%% Batch gradient descent %%%
if ~stochastic,

  while true,

    %%% Compute statistics %%%
    rP=Bi'*Pi;
    [E,J,fX,fP,fPP]=ldppr_index(rP,PPi,Bi'*X,XX,work);
    if devel,
      E=ldppr_index(rP,PPi,Bi'*Y,YY,dwork);
    end

    %%% Determine if there was improvement %%%
    mark='';
    if (  testJ && (J<bestIJE(2)||(J==bestIJE(2)&&E<=bestIJE(3))) ) || ...
       ( ~testJ && (E<bestIJE(3)||(E==bestIJE(3)&&J<=bestIJE(2))) ),
      bestB=Bi;
      bestP=Pi;
      bestPP=PPi;
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
    if indepPP,
      PPi=PPi-ratePP.*fPP;
    else
      PPi=PPi-ratePP.*fPP(onesDD,:);
    end

    %if mod(I,3)==0,
    %  Bi=Bi-rateB.*(X*fX'+Pi*fP');
    %elseif mod(I+1,3)==0,
    %  Pi=Pi-rateP.*(Bi*fP);
    %else
    %  if indepPP,
    %    PPi=PPi-ratePP.*fPP;
    %  else
    %    PPi=PPi-ratePP.*fPP(onesDD,:);
    %  end
    %end

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
  if devel,
    prevE=ldppr_index(Bi'*Pi,PPi,Bi'*Y,YY,dwork);
  else
    prevE=ldppr_index(Bi'*Pi,PPi,Bi'*X,XX,work);
  end

  while true,

    %%% Compute statistics %%%
    if mod(I,stocheck)==0 && stocheckfull,
      rP=Bi'*Pi;
      [E,J]=ldppr_index(rP,PPi,Bi'*X,XX,work);
      if devel,
        E=ldppr_index(rP,PPi,Bi'*Y,YY,dwork);
      end
    end

    %%% Select random samples %%%
    randn=round((Nx-1).*rand(stochsamples,1))+1; % modify for no repetitions
    sX=X(:,randn);

    %%% Compute statistics %%%
    [Ei,Ji,fX,fP,fPP]=ldppr_sindex(Bi'*Pi,PPi,Bi'*sX,XX(:,randn),swork);
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
        E=ldppr_index(Bi'*Pi,PPi,Bi'*Y,YY,dwork);
      end

      %%% Determine if there was improvement %%%
      mark='';
      if (  testJ && (J<bestIJE(2)||(J==bestIJE(2)&&E<=bestIJE(3))) ) || ...
         ( ~testJ && (E<bestIJE(3)||(E==bestIJE(3)&&J<=bestIJE(2))) ),
        bestB=Bi;
        bestP=Pi;
        bestPP=PPi;
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
    if indepPP,
      PPi=PPi-ratePP.*fPP;
    else
      PPi=PPi-ratePP.*fPP(onesDD,:);
    end

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
    [E,J]=ldppr_index(bestB'*bestP,bestPP,bestB'*X,XX,work);
    if devel,
      E=ldpp_index(bestB'*bestP,bestPP,bestB'*Y,YY,dwork);
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
if ~MAD,
  xxsd=sqrt(xxsd);
end
bestPP=bestPP.*xxsd(:,onesNp)+xxmu(:,onesNp);
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
function [E, J, fX, fP, fPP] = ldppr_index(rP, PP, rX, XX, work)

  R=work.R;
  DD=work.DD;
  Np=work.Np;
  Nx=work.Nx;
  onesR=work.onesR;
  onesDD=work.onesDD;
  mindist=work.mindist;
  ind1=work.ind1;
  ind2=work.ind2;

  %%% Compute distances %%%
  if work.euclidean,
    dist=sum(power(repmat(rX,1,Np)-rP(:,ind1),2),1);
  else %elseif cosine,
    rpsd=sqrt(sum(rP.*rP,1));
    rP=rP./rpsd(onesR,:);
    rxsd=sqrt(sum(rX.*rX,1));
    rX=rX./rxsd(onesR,:);
    dist=1-sum(repmat(rX,1,Np).*rP(:,ind1),1);
  end
  dist(dist<mindist)=mindist;
  dist=reshape(1./dist,Nx,Np);

  S=sum(dist,2);
  mXX=(reshape(sum(repmat(dist,DD,1).*PP(ind2,:),2),Nx,DD)./S(:,onesDD))';
  dXX=mXX-XX;

  %%% Compute statistics %%%
  if work.MAD;
    E=sum(sum(abs(dXX),2).*work.xxsd)/work.NxDD;
  else
    E=sqrt(sum(sum(dXX.*dXX,2).*work.xxsd)/work.NxDD);
  end
  if nargout>1,
    tanhXX=tanh(work.slope*sum(dXX.*dXX,1))';
    J=sum(tanhXX)/Nx;
  end

  %%% Compute gradient %%%
  if nargout>2,
    dist=dist.*dist;
    fact=repmat((1-tanhXX.*tanhXX)./S,Np,1).*dist(:);
    if work.indepPP,
      fPP=permute(sum(reshape(fact(:,onesDD).*repmat(dXX,1,Np)',Nx,Np,DD)),[3 2 1]);
    else
      fPP=sum(reshape(fact.*sum(repmat(dXX,1,Np),1)',Nx,Np),1);
    end
    fact=fact.*sum(repmat(dXX,1,Np).*(repmat(mXX,1,Np)-PP(:,ind1)),1)';

    if work.euclidean,
      fact=reshape(fact(:,onesR)'.*(repmat(rX,1,Np)-rP(:,ind1)),[R Nx Np]);
      fP=-permute(sum(fact,2),[1 3 2]);
      fX=sum(fact,3);
    else %elseif cosine,
      fP=-permute(sum(reshape(fact(:,onesR)'.*repmat(rX,1,Np),[R Nx Np]),2),[1 3 2]);
      fX=-sum(reshape(fact(:,onesR)'.*rP(:,ind1),[R Nx Np]),3);
    end
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [E, J, fX, fP, fPP] = ldppr_sindex(rP, PP, rX, XX, work)

  R=work.R;
  DD=work.DD;
  Np=work.Np;
  Nx=work.Nx;
  onesR=work.onesR;
  onesDD=work.onesDD;
  mindist=work.mindist;
  ind1=work.ind1;
  ind2=work.ind2;

  %%% Compute distances %%%
  if work.euclidean,
    dist=sum(power(repmat(rX,1,Np)-rP(:,ind1),2),1);
  else %elseif cosine,
    rpsd=sqrt(sum(rP.*rP,1));
    rP=rP./rpsd(onesR,:);
    rxsd=sqrt(sum(rX.*rX,1));
    rX=rX./rxsd(onesR,:);
    dist=1-sum(repmat(rX,1,Np).*rP(:,ind1),1);
  end
  dist(dist<mindist)=mindist;
  dist=reshape(1./dist,Nx,Np);

  S=sum(dist,2);
  mXX=(reshape(sum(repmat(dist,DD,1).*PP(ind2,:),2),Nx,DD)./S(:,onesDD))';
  dXX=mXX-XX;
  tanhXX=tanh(work.slope*sum(dXX.*dXX,1))';

  %%% Compute statistics %%%
  if work.MAD;
    E=sum(sum(abs(dXX),2).*work.xxsd)/work.NxDD;
  else
    E=sqrt(sum(sum(dXX.*dXX,2).*work.xxsd)/work.NxDD);
  end
  J=sum(tanhXX)/Nx;

  %%% Compute gradient %%%
  dist=dist.*dist;
  fact=repmat((1-tanhXX.*tanhXX)./S,Np,1).*dist(:);
  if work.indepPP,
    fPP=permute(sum(reshape(fact(:,onesDD).*repmat(dXX,1,Np)',Nx,Np,DD)),[3 2 1]);
  else
    fPP=sum(reshape(fact.*sum(repmat(dXX,1,Np),1)',Nx,Np),1);
  end
  fact=fact.*sum(repmat(dXX,1,Np).*(repmat(mXX,1,Np)-PP(:,ind1)),1)';

  if work.euclidean,
    fact=reshape(fact(:,onesR)'.*(repmat(rX,1,Np)-rP(:,ind1)),[R Nx Np]);
    fP=-permute(sum(fact,2),[1 3 2]);
    fX=sum(fact,3);
  else %elseif cosine,
    fP=-permute(sum(reshape(fact(:,onesR)'.*repmat(rX,1,Np),[R Nx Np]),2),[1 3 2]);
    fX=-sum(reshape(fact(:,onesR)'.*rP(:,ind1),[R Nx Np]),3);
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P0, PP0] = ldppr_initP(X, XX, M, varargin)
%
% LDPPR_INITP: Initialize Prototypes for LDPPR
%
% [P0, PP0] = ldppr_initP(X, XX, M)
%
%   Input:
%     X       - Independent data matrix. Each column vector is a data point.
%     XX      - Dependent data matrix.
%     M       - Number of prototypes.
%
%   Input (optional):
%     'extra',EXTRA              - Extrapolate EXTRA from extreme values (defaul=false)
%     'multi',MULT               - Multimodal prototypes, MULT-means (defaul=false)
%     'seed',SEED                - Random seed (default=system)
%
%   Output:
%     P0      - Initialized prototypes. Independent data.
%     PP0     - Initialized prototypes. Dependent data.
%

fn='ldppr_initP:';
minargs=3;

P0=[];
PP0=[];

seed=rand('state');

logfile=2;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'extra') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'multi'),
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

if argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif exist('multi','var') && mod(M,multi)~=0,
  fprintf(logfile,'%s error: the number of prototypes should be a multiple of MULT\n',fn);
  return;
end

DD=size(XX,1);

mx=max(XX');
mn=min(XX');

if exist('extra','var'),
  omx=mx;
  omn=mn;
  mx=omx+extra.*(omx-omn);
  mn=omn-extra.*(omx-omn);
end

if ~exist('multi','var'),
  multi=1;
else
  M=M/multi;
end

rand('state',seed);

if DD==1,

  d=(mx-mn)/(M-1);

  for m=mn:d:mx,
    s=XX>=m-d/2 & XX<m+d/2;
    if sum(s)>multi,
      P0=[P0,kmeans(X(:,s),multi,'seed',seed)];
      seed=rand('state');
    else
      [mdist,idx]=sort(abs(XX-m));
      P0=[P0,X(:,idx(1:multi))];
    end
    PP0=[PP0,m*ones(1,multi)];
  end

elseif DD==2,

  M=round(sqrt(M));
  d=(mx-mn)/(M-1);

  Nx=size(XX,2);
  onesNx=ones(Nx,1);

  for m=mn(1):d(1):mx(1),
    sm=XX(1,:)>=m-d(1)/2 & XX(1,:)<m+d(1)/2;
    for n=mn(2):d(2):mx(2),
      sn=XX(2,:)>=n-d(2)/2 & XX(2,:)<n+d(2)/2;
      s=sm&sn;
      if sum(s)>multi,
        P0=[P0,kmeans(X(:,s),multi,'seed',seed)];
        seed=rand('state');
      else
        mu=[m;n];
        mdist=sum((XX-mu(:,onesNx)).^2,1);
        [mdist,idx]=sort(mdist);
        P0=[P0,X(:,idx(1:multi))];
      end
      PP0=[PP0,[m;n]*ones(1,multi)];
    end
  end
  
else
  fprintf(logfile,'%s error: dimensionality of dependent data higher than two not supported\n',fn);
end
