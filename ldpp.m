function [bestB, bestP, Plabels, info, other] = ldpp(X, Xlabels, B0, P0, Plabels, varargin)
%
% LDPP: Learning Discriminative Projections and Prototypes for NN Classification
%
% Usage:
%   [B, P] = ldpp(X, Xlabels, B0, P0, Plabels, ...)
%
% Usage initialize prototypes:
%   [P0, Plabels] = ldpp('initP', X, Xlabels, Npc, B)
%
% Usage cross-validation (PCA & kmeans initialization):
%   [B, P, Plabels] = ldpp(X, Xlabels, maxDr, maxNpc, [], ...)
%
% Input:
%   X       - Data matrix. Each column vector is a data point.
%   Xlabels - Data class labels.
%   B0      - Initial projection base.
%   P0      - Initial prototypes.
%   Plabels - Prototype class labels.
%   maxDr   - Maximum reduced dimensionality.
%   maxNpc  - Maximum number of prototypes per class.
%
% Output:
%   B       - Final learned projection base.
%   P       - Final learned prototypes.
%
% Learning options:
%   'slope',SLOPE              - Sigmoid slope (defaul=10)
%   'rateB',RATEB              - Projection base learning rate (default=0.1)
%   'rateP',RATEP              - Prototypes learning rate (default=0.1)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'prior',[PRIOR]            - A priori probabilities (default=Nc/N)
%   'orthoit',OIT              - Orthogonalize every OIT (default=1)
%   'orthonormal',(true|false) - Orthonormal projection base (default=true)
%   'orthogonal',(true|false)  - Orthogonal projection base (default=false)
%   'euclidean'                - Euclidean distance (default=true)
%   'cosine'                   - Cosine distance (default=false)
%   'rtangent'                 - Ref. tangent distance (default=false)
%   'otangent'                 - Obs. tangent distance (default=false)
%   'atangent'                 - Avg. tangent distance (default=false)
%
% Data normalization options:
%   'normalize',(true|false)   - Normalize training data (default=true)
%   'linearnorm',(true|false)  - Linear normalize training data (default=false)
%   'whiten',(true|false)      - Whiten training data (default=false)
%
% Stochastic options:
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=10)
%   'stocheck',SIT             - Stats every SIT stoch. iterations (default=100)
%   'stocheckfull',(true|f...  - Stats for whole data set (default=false)
%   'stochfinalexact',(tru...  - Final stats for whole data set (default=true)
%
% Verbosity options:
%   'verbose',(true|false)     - Verbose (default=true)
%   'stats',STAT               - Statistics every STAT (default=10)
%   'logfile',FID              - Output log file (default=stderr)
%
% Tangent distances options:
%   'tangtypes'                - Tangent types [hvrspdtHV]+[k]K (default='k2')
%                                h: image horizontal translation
%                                v: image vertical translation
%                                r: image rotation
%                                s: image scaling
%                                p: image parallel hyperbolic transformation
%                                d: image diagonal hyperbolic transformation
%                                t: image trace thickening
%                                H: image horizontal illumination
%                                V: image vertical illumination
%                                k: K nearest neighbors
%   'imSize',[W H]             - Image size (default=square)
%   'bw',BW                    - Tangent derivative gaussian bandwidth (default=0.5)
%   'krh',KRH                  - Supply tangent derivative kernel, horizontal
%   'krv',KRV                  - Supply tangent derivative kernel, vertical
%
% Other options:
%   'devel',Y,Ylabels          - Set the development set (default=false)
%   'seed',SEED                - Random seed (default=system)
%
% Cross-validation options:
%   'crossvalidate',K          - Do K-fold cross-validation (default=2)
%   'cv_slope',[SLOPES]        - Slopes to try (default=[10])
%   'cv_Npc',[NPCs]            - Prototypes per class to try (default=[2.^[0:3]])
%   'cv_Dr',[DRs]              - Reduced dimensions to try (default=[2.^[2:5]])
%   'cv_rateB',[RATEBs]        - B learning rates to try (default=[10.^[-2:0]])
%   'cv_rateP',[RATEPs]        - P learning rates to try (default=[10.^[-2:0]])
%   'cv_save',(true|false)     - Save cross-validation results (default=false)
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
% Copyright (C) 2008-2010 Mauricio Villegas (mvillegas AT iti.upv.es)
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

if strncmp(X,'initP',5),
  [bestB, bestP] = ldpp_initP(Xlabels, B0, unique(B0), P0, Plabels);
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

epsilon=1e-7;
minI=100;
maxI=1000;
stats=10;
orthoit=1;

devel=false;
stochastic=false;
stochsamples=10;
stocheck=100;
stocheckfull=false;
stochfinalexact=true;
orthonormal=true;
orthogonal=false;
dtype.euclidean=true;
dtype.cosine=false;
dtype.tangent=false;
dtype.rtangent=false;
dtype.otangent=false;
dtype.atangent=false;
tangtypes='k2';
normalize=true;
linearnorm=false;
whiten=false;
testJ=false;
crossvalidate=false;
cv_save=false;
initP=false;

logfile=2;
verbose=true;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}),% || size(varargin,2)<n+1,
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
         strcmp(varargin{n},'prior') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'stochsamples') || ...
         strcmp(varargin{n},'stocheck') || ...
         strcmp(varargin{n},'orthoit') || ...
         strcmp(varargin{n},'crossvalidate') || ...
         strcmp(varargin{n},'cv_slope') || ...
         strcmp(varargin{n},'cv_Npc') || ...
         strcmp(varargin{n},'cv_Dr') || ...
         strcmp(varargin{n},'cv_rateB') || ...
         strcmp(varargin{n},'cv_rateP') || ...
         strcmp(varargin{n},'imSize') || ...
         strcmp(varargin{n},'bw') || ...
         strcmp(varargin{n},'krh') || ...
         strcmp(varargin{n},'krv') || ...
         strcmp(varargin{n},'logfile'),
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
         strcmp(varargin{n},'testJ') || ...
         strcmp(varargin{n},'cv_save') || ...
         strcmp(varargin{n},'initP') || ...
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
  elseif strcmp(varargin{n},'tangtypes'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~ischar(varargin{n+1}),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'euclidean') || ...
         strcmp(varargin{n},'rtangent') || ...
         strcmp(varargin{n},'otangent') || ...
         strcmp(varargin{n},'atangent') || ...
         strcmp(varargin{n},'cosine'),
    dtype.euclidean=false;
    dtype.cosine=false;
    dtype.rtangent=false;
    dtype.otangent=false;
    dtype.atangent=false;
    eval(['dtype.',varargin{n},'=true;']);
    n=n+1;
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

if max(size(Plabels))==0,
  Plabels=unique(Xlabels);
end
C=max(size(unique(Plabels)));
[D,Nx]=size(X);
if devel,
  Ny=size(Y,2);
end

%%% Automatic initial parameters %%%
if ~crossvalidate && max(size(B0))==1 && max(size(P0))==1,
  crossvalidate=2;
end
if max(size(B0))==1,
  if ~crossvalidate,
    Bi=pca(X);
    B0=Bi(:,1:min(B0,D));
  else
    B0=rand(D,min(B0,D));
  end
end
if max(size(P0))==1,
  if ~crossvalidate,
    [P0,Plabels]=ldpp_initP(X,Xlabels,unique(Xlabels),P0,B0);
  else
    Plabels=repmat(unique(Xlabels),P0,1);
    P0=rand(D,C*P0);
  end
end
if initP,
  bestB=B0;
  bestP=P0;
  return;
end

Dr=size(B0,2);
Np=size(P0,2);

%%% Probe mode %%%
if exist('probemode','var'),
  probevars=fieldnames(probemode);
  for n=1:size(probevars,1),
    eval([probevars{n} '=probemode.' probevars{n} ';']);
  end
  normalize=false;
  verbose=true;
  probemode=true;
else
  probemode=false;
  if crossvalidate,
    if ~exist('cv_slope','var'),
      cv_slope=slope;
    end
    if ~exist('cv_Npc','var'),
      cv_Npc=[2.^[0:3]];
      cv_Npc(cv_Npc>=Np/C)=[];
      cv_Npc=[cv_Npc,Np/C];
    end
    if ~exist('cv_Dr','var'),
      cv_Dr=[2.^[2:5]];
      cv_Dr(cv_Dr>=Dr)=[];
      cv_Dr=[cv_Dr,Dr];
    end
    if ~exist('cv_rateB','var'),
      cv_rateB=[10.^[-2:0]];
    end
    if ~exist('cv_rateP','var'),
      cv_rateP=[10.^[-2:0]];
    end
  end
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
       max(size(Plabels))~=Np || min(size(Plabels))~=1 || ...
       (devel && (max(size(Ylabels))~=Ny || min(size(Ylabels))~=1) ),
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
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

%%% Preprocessing %%%
if ~probemode,
  tic;

  if exist('seed','var'),
    rand('state',seed);
  end

  if exist('rates','var'),
    rateB=rates;
    rateP=rates;
  end

  onesNx=ones(Nx,1);
  onesNp=ones(Np,1);
  onesDr=ones(Dr,1);
  if devel,
    onesNy=ones(Ny,1);
  end

  %%% Normalization %%%
  oX=X;
  if devel,
    oY=Y;
  end
  if normalize || linearnorm,
    xmu=mean(X,2);
    xsd=std(X,1,2);
    if dtype.euclidean || dtype.rtangent || dtype.otangent || dtype.atangent,
      xsd=Dr*xsd;
    end
    if linearnorm,
      xsd=max(xsd)*ones(size(xsd));
    end
    if issparse(X) && ~dtype.cosine,
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
    B0=B0.*xsd(:,onesDr);
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
    if dtype.euclidean || dtype.rtangent || dtype.otangent || dtype.atangent,
      W=(1/Dr).*W;
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
    clear nPlabels nXlabels nYlabels;
  end

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
  if dtype.euclidean || dtype.rtangent || dtype.otangent || dtype.atangent,
    cfact=2*cfact;
    jfact=0.5;
  end

  %%% Tangent vectors %%%
  if dtype.rtangent || dtype.otangent || dtype.atangent,
    dtype.tangent=true;
    tangcfg.imgtangcfg=struct;
    if exist('imSize','var'),
      tangcfg.imgtangcfg.imSize=imSize;
    end
    if exist('bw','var'),
      tangcfg.imgtangcfg.bw=bw;
    end
    if exist('krh','var'),
      tangcfg.imgtangcfg.krh=krh;
    end
    if exist('krv','var'),
      tangcfg.imgtangcfg.krv=krv;
    end
    tangcfg.imgtangs=false;
    tangcfg.knntangs=false;
    tangcfg.devel=devel;
    tangcfg.dtype=dtype;
    tangcfg.onesNp=onesNp;
    tangcfg.Np=Np;
    tangcfg.D=D;
    tangcfg.Vx=[];
    tangcfg.Vp=[];
    tangcfg.Vy=[];
    %%% k-NN tangents %%%
    if sum(tangtypes=='k')>0,
      idx=find(tangtypes=='k');
      tangcfg.knntangs=str2num(tangtypes(idx+1:end));
      tangcfg.knntypes=tangtypes(idx:end);
      tangtypes=tangtypes(1:idx-1);
      tangcfg.Xlabels=Xlabels;
      tangcfg.Plabels=Plabels;
      if devel,
        tangcfg.Ylabels=Ylabels;
      end
    end
    %%% Image tangents %%%
    if numel(tangtypes)>0,
      tangcfg.imgtangs=numel(tangtypes);
      tangcfg.imgtypes=tangtypes;
      tangcfg.normalize=normalize;
      tangcfg.linearnorm=linearnorm;
      tangcfg.whiten=whiten;
      if dtype.rtangent || dtype.atangent,
        if normalize || linearnorm,
          tangcfg.xmu=xmu;
          tangcfg.xsd=xsd;
        elseif whiten,
          tangcfg.W=W;
          tangcfg.IW=IW;
          tangcfg.xmu=xmu;
        end
        if tangcfg.knntangs,
          tangcfg.knntangsp=repmat([false(tangcfg.imgtangs,1);true(tangcfg.knntangs,1)],1,Np);
          tangcfg.knntangsp=tangcfg.knntangsp(:);
          tangcfg.Vp=zeros(Dr,numel(tangcfg.knntangsp));
        end
      end
      if dtype.otangent || dtype.atangent,
        tangcfg.oVx=tangVects(oX,tangcfg.imgtypes,tangcfg.imgtangcfg);
        if normalize || linearnorm,
          tangcfg.oVx=(tangcfg.oVx-xmu(:,ones(size(tangcfg.oVx,2),1)))./xsd(:,ones(size(tangcfg.oVx,2),1));
          if sum(xsd==0)>0,
            tangcfg.oVx(xsd==0,:)=[];
          end
        elseif whiten,
          tangcfg.oVx=W'*(tangcfg.oVx-cfg.xmu(:,ones(size(tangcfg.oVx,2),1)));
        end
        if tangcfg.knntangs,
          tangcfg.knntangsx=repmat([false(tangcfg.imgtangs,1);true(tangcfg.knntangs,1)],1,Nx);
          tangcfg.knntangsx=tangcfg.knntangsx(:);
          tangcfg.Vx=zeros(Dr,numel(tangcfg.knntangsx));
        end
        if devel,
          tangcfg.oVy=tangVects(oY,tangcfg.imgtypes,tangcfg.imgtangcfg);
          if normalize || linearnorm,
            tangcfg.oVy=(tangcfg.oVy-xmu(:,ones(size(tangcfg.oVy,2),1)))./xsd(:,ones(size(tangcfg.oVy,2),1));
            if sum(xsd==0)>0,
              tangcfg.oVy(xsd==0,:)=[];
            end
          elseif whiten,
            tangcfg.oVy=W'*(tangcfg.oVy-cfg.xmu(:,ones(size(tangcfg.oVy,2),1)));
          end
          if tangcfg.knntangs,
            tangcfg.knntangsy=repmat([false(tangcfg.imgtangs,1);true(tangcfg.knntangs,1)],1,Ny);
            tangcfg.knntangsy=tangcfg.knntangsy(:);
            tangcfg.Vy=zeros(Dr,numel(tangcfg.knntangsy));
          end
        end
      end
    end
    tangcfg.L=tangcfg.knntangs+tangcfg.imgtangs;
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
  work.sel=Plabels(:,onesNx)'==Xlabels(:,onesNp);
  work.slope=slope;
  work.onesDr=onesDr;
  work.onesNp=onesNp;
  work.onesNx=onesNx;
  work.Np=Np;
  work.Nx=Nx;
  work.C=C;
  work.Dr=Dr;
  work.dtype=dtype;
  work.cfact=cfact;
  work.jfact=jfact;
  work.prior=prior;
  if dtype.tangent,
    work.L=tangcfg.L;
    if dtype.otangent || dtype.atangent,
      work.tidx=repmat([1:Nx],work.L,1);
      work.tidx=work.tidx(:);
    end
  end

  if stochastic,
    swork=work;
    onesS=ones(stochsamples,1);
    swork.onesNx=onesS;
    swork.overNx=1/stochsamples;
    swork.onesNp=onesNp;
    swork.Nx=stochsamples;
  end

  if devel,
    dwork=work;
    dwork.sel=Plabels(:,onesNy)'==Ylabels(:,onesNp);
    dwork.Nx=Ny;
    dwork.onesNx=onesNy;
    if dtype.tangent,
      dwork.L=tangcfg.L;
      if dtype.otangent || dtype.atangent,
        dwork.tidx=repmat([1:Ny],work.L,1);
        dwork.tidx=dwork.tidx(:);
      end
    end
  end

  tm=toc;
  info.time=tm;
  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,tm);
end

%%% Cross-validaton %%%
if crossvalidate,
  tic;

  cv_Ny=floor(Nx/crossvalidate);
  cv_Nx=cv_Ny*(crossvalidate-1);

  %%% Generate cross-validation partitions %%%
  cv_rparts=100;
  cv_badrpart=true;
  while cv_badrpart,
    if cv_rparts<=0,
      fprintf(logfile,'%s error: unable to find adequate cross-validation partitions\n',fn);
      return;
    end
    cv_badrpart=false;
    [v,cv_rnd]=sort(rand(Nx,1));
    cv_rnd=cv_rnd(1:cv_Ny*crossvalidate);
    for v=1:crossvalidate,
      cv_Xrnd=cv_rnd;
      cv_Xrnd((v-1)*cv_Ny+1:v*cv_Ny)=[];
      if numel(unique(Xlabels(cv_Xrnd)))~=C,
        cv_badrpart=true;
        break;
      end
    end
    cv_rparts=cv_rparts-1;
  end
    
  Bi=pca(X);
  Bi=Bi(:,1:min(D,32));

  %%% Constant data structures %%%
  cv_onesNx=ones(cv_Nx,1);
  cv_onesNy=ones(cv_Ny,1);

  cv_wk=work;
  cv_wk.Nx=cv_Nx;
  cv_wk.onesNx=cv_onesNx;

  cv_dwk=work;
  cv_dwk.Nx=cv_Ny;
  cv_dwk.onesNx=cv_onesNy;

  if stochastic,
    cv_swk=swork;
  end

  if dtype.tangent,
    if dtype.otangent || dtype.atangent,
      cv_wk.tidx=repmat([1:cv_Nx],work.L,1);
      cv_wk.tidx=cv_wk.tidx(:);
      cv_dwk.tidx=repmat([1:cv_Ny],work.L,1);
      cv_dwk.tidx=cv_dwk.tidx(:);
    end
  end

  cv_cfg.minI=minI;
  cv_cfg.maxI=maxI;
  cv_cfg.epsilon=epsilon;
  cv_cfg.stats=maxI+1;
  cv_cfg.orthoit=orthoit;
  cv_cfg.orthonormal=orthonormal;
  cv_cfg.orthogonal=orthogonal;
  cv_cfg.dtype=dtype;
  if dtype.tangent,
    cv_cfg.tangcfg=tangcfg;
    cv_cfg.tangcfg.devel=true;
  end
  cv_cfg.testJ=testJ;
  cv_cfg.stochastic=stochastic;
  cv_cfg.devel=true;
  if stochastic,
    cv_cfg.onesC=onesC;
    cv_cfg.onesS=onesS;
    cv_cfg.cumprior=cumprior;
    cv_cfg.stochsamples=stochsamples;
    cv_cfg.stocheck=stocheck;
    cv_cfg.stocheckfull=stocheckfull;
    cv_cfg.stochfinalexact=stochfinalexact;
  end
  cv_cfg.logfile=fopen('/dev/null');

  Nparam=numel(cv_slope)*numel(cv_Npc)*numel(cv_Dr)*numel(cv_rateB)*numel(cv_rateP);
  cv_E=zeros(Nparam,1);
  cv_I=zeros(Nparam,1);

  %%% Perform cross-validation %%%
  for v=1:crossvalidate,
    cv_Xrnd=cv_rnd;
    cv_Xrnd((v-1)*cv_Ny+1:v*cv_Ny)=[];
    cv_Xrnd=sort(cv_Xrnd);
    cv_Yrnd=cv_rnd((v-1)*cv_Ny+1:v*cv_Ny);
    cv_Yrnd=sort(cv_Yrnd);
    cv_X=X(:,cv_Xrnd);
    cv_Xlabels=Xlabels(cv_Xrnd);

    cv_cfg.Y=X(:,cv_Yrnd);
    cv_cfg.Ylabels=Xlabels(cv_Yrnd);

    if dtype.tangent,
      if dtype.otangent || dtype.atangent,
        sel=(repmat((cv_Xrnd-1)*work.L+1,1,work.L)+repmat([0:work.L-1],cv_Nx,1))';
        cv_cfg.tangcfg.oVx=tangcfg.oVx(:,sel(:));
        sel=(repmat((cv_Yrnd-1)*work.L+1,1,work.L)+repmat([0:work.L-1],cv_Ny,1))';
        cv_cfg.tangcfg.oVy=tangcfg.oVx(:,sel(:));
      end
    end

    cv_cfact=zeros(C,1);
    for c=1:C,
      cv_cfact(c)=prior(c)/sum(cv_Xlabels==c);
    end
    if dtype.euclidean || dtype.tangent,
      cv_cfact=2*cv_cfact;
    end
    cv_wk.cfact=cv_cfact(cv_Xlabels);

    if stochastic,
      cv_nc=zeros(C,1);
      cv_cnc=zeros(C,1);
      cv_nc(1)=sum(cv_Xlabels==1);
      for c=2:C,
        cv_nc(c)=sum(cv_Xlabels==c);
        cv_cnc(c)=cv_cnc(c-1)+cv_nc(c-1);
      end
      cv_cfg.nc=cv_nc;
      cv_cfg.cnc=cv_cnc;
    end

    %%% Vary the slope %%%
    param=1;
    for slope=cv_slope,
      cv_wk.slope=slope;
      cv_dwk.slope=slope;
      if stochastic,
        cv_swk.slope=slope;
      end

      %%% Vary the number of prototypes %%%
      for Np=cv_Npc,
        Np=C*Np;
        onesNp=ones(Np,1);

        [P0,Plabels]=ldpp_initP(cv_X,cv_Xlabels,[1:C]',Np/C,Bi);
        cv_wk.sel=Plabels(:,cv_onesNx)'==cv_Xlabels(:,onesNp);
        cv_dwk.sel=Plabels(:,cv_onesNy)'==cv_cfg.Ylabels(:,onesNp);
        cv_wk.onesNp=onesNp;
        cv_dwk.onesNp=onesNp;
        cv_wk.Np=Np;
        cv_dwk.Np=Np;

        if stochastic,
          cv_swk.Np=Np;
          cv_swk.onesNp=onesNp;
        end

        %%% Vary the reduced dimensionality %%%
        for Dr=cv_Dr,
          B0=Bi(:,1:Dr);
          onesDr=ones(Dr,1);
          cv_cfg.onesDr=onesDr;

          cv_wk.Dr=Dr;
          cv_wk.onesDr=onesDr;
          cv_dwk.Dr=Dr;
          cv_dwk.onesDr=onesDr;
          cv_cfg.work=cv_wk;
          cv_cfg.dwork=cv_dwk;
          if stochastic,
            cv_swk.Dr=Dr;
            cv_swk.onesDr=onesDr;
            cv_cfg.swork=cv_swk;
          end

          fprintf(logfile,'%s cv %d: slope=%g Np=%d Dr=%d ',fn,v,slope,Np,Dr);

          %%% Vary learning rates %%%
          for rateB=cv_rateB,
            cv_cfg.rateB=rateB;
            for rateP=cv_rateP,
              cv_cfg.rateP=rateP;

              [I,E]=ldpp(cv_X,cv_Xlabels,B0,P0,Plabels,'probemode',cv_cfg);
              cv_E(param)=cv_E(param)+E;
              cv_I(param)=cv_I(param)+I;
              cv_param{param}.slope=slope;
              cv_param{param}.Np=Np;
              cv_param{param}.Dr=Dr;
              cv_param{param}.rateB=rateB;
              cv_param{param}.rateP=rateP;
              param=param+1;
              fprintf(logfile,'.');
            end
          end
          fprintf(logfile,'\n');
        end
      end
    end
  end

  cv_E=cv_E./crossvalidate;
  cv_I=cv_I./crossvalidate;
  param=find(min(cv_E)==cv_E,1);
  if cv_save,
    save('ldpp_cv.mat','cv_E','cv_I','cv_param');
  end
  info.cv_E=cv_E;
  info.cv_impI=cv_I;
  info.cv_param=cv_param;

  %%% Get best cross-validaton parameters %%%
  slope=cv_param{param}.slope;
  Np=cv_param{param}.Np;
  Dr=cv_param{param}.Dr;
  rateB=cv_param{param}.rateB;
  rateP=cv_param{param}.rateP;
  onesDr=ones(Dr,1);
  onesNp=ones(Np,1);

  B0=Bi(:,1:Dr);
  [P0,Plabels]=ldpp_initP(X,Xlabels,[1:C]',Np/C,Bi);
  work.sel=Plabels(:,onesNx)'==Xlabels(:,onesNp);

  work.slope=slope;
  work.onesDr=onesDr;
  work.onesNp=onesNp;
  work.Np=Np;
  work.Dr=Dr;
  if stochastic,
    swork.slope=slope;
    swork.onesDr=onesDr;
    swork.onesNp=onesNp;
    swork.Np=Np;
    swork.Dr=Dr;
  end
  if devel,
    dwork.sel=Plabels(:,onesNy)'==Ylabels(:,onesNp);
    dwork.slope=slope;
    dwork.onesDr=onesDr;
    dwork.onesNp=onesNp;
    dwork.Np=Np;
    dwork.Dr=Dr;
  end

  cv_test='E';
  if testJ,
    cv_test='J';
  end

  tm=toc;
  info.time=info.time+tm;
  fprintf(logfile,'%s cv best statistics: %s=%g impI=%g\n',fn,cv_test,cv_E(param),cv_I(param));
  fprintf(logfile,'%s cv best parameters: slope=%g Np=%d Dr=%d rateB=%g rateP=%g\n',fn,slope,Np,Dr,rateB,rateP);
  fprintf(logfile,'%s total cross-validation time (s): %f\n',fn,tm);

  fclose(cv_cfg.logfile);
  clear cv_*;
end

Bi=B0;
Pi=P0;
bestB=B0;
bestP=P0;
bestIJE=[0 1 1 -1];

J00=1;
J0=1;
I=0;

if ~probemode,
  fprintf(logfile,'%s Nx=%d C=%d D=%d Dr=%d Np=%d\n',fn,Nx,C,D,Dr,Np);
  fprintf(logfile,'%s output: iteration | J | delta(J) | E\n',fn);
  tic;
end

%%% Batch gradient descent %%%
if ~stochastic,

  while true,

    %%% Compute statistics %%%
    rP=Bi'*Pi;
    rX=Bi'*X;
    if devel,
      rY=Bi'*Y;
    end
    if dtype.tangent,
      tangcfg.rP=rP;
      tangcfg.rX=rX;
      if devel,
        tangcfg.rY=rY;
      end
      tangcfg=comptangs(Bi,Pi,tangcfg);
      work.Vx=tangcfg.Vx;
      work.Vp=tangcfg.Vp;
      if devel,
        dwork.Vx=tangcfg.Vy;
        dwork.Vp=tangcfg.Vp;
      end
    end
    [E,J,fX,fP]=ldpp_index(rP,Plabels,rX,Xlabels,work);
    if devel,
      E=ldpp_index(rP,Plabels,rY,Ylabels,dwork);
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
      rP=Bi'*Pi;
      [E,J]=ldpp_index(rP,Plabels,Bi'*X,Xlabels,work);
      if devel,
        E=ldpp_index(rP,Plabels,Bi'*Y,Ylabels,dwork);
      end
    end

    %%% Select random samples %%%
    rands=rand(1,stochsamples);
    randc=(sum(rands(onesC,:)>cumprior(:,onesS))+1)';
    randn=cnc(randc)+round((nc(randc)-1).*rand(stochsamples,1))+1;
    sX=X(:,randn);

    %%% Compute statistics %%%
    if work.dtype.tangent,
      fprintf(logfile,'%s error: stochastic for tangent distances not implemented\n',fn);
      return;
    end
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
  info.time=info.time+tm;
  info.E=bestIJE(3);
  info.J=bestIJE(2);
  info.I=bestIJE(1);
  info.impI=bestIJE(4)/max(I,1);
  fprintf(logfile,'%s best iteration: I=%d J=%f E=%f\n',fn,bestIJE(1),bestIJE(2),bestIJE(3));
  fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJE(4)/max(I,1));
  fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
  fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);
end

if ~verbose,
  fclose(logfile);
end

if probemode,
  bestB=bestIJE(4)/max(I,1); % amount of improvement iterations
  if testJ,
    bestP=bestIJE(2); % optimization index
  else
    bestP=bestIJE(3); % error rate
  end
  return;
end

if nargout>2,
  Plabels=clab(Plabels);
end

if nargout>4,
  other=struct;
  if dtype.tangent,
    tangcfg.rX=bestB'*X;
    tangcfg=comptangs(bestB,bestP,tangcfg);
    if dtype.otangent || dtype.atangent,
      other.Vx=tangcfg.Vx;
      if devel,
        other.Vy=tangcfg.Vy;
      end
    end
    if dtype.rtangent || dtype.atangent,
      other.Vp=tangcfg.Vp;
    end
  end
end

%%% Compensate for normalization in the final parameters %%%
if normalize || linearnorm,
  if issparse(X) && ~dtype.cosine,
    bestP=bestP.*xsd(xsd~=0,onesNp);
  else
    bestP=bestP.*xsd(xsd~=0,onesNp)+xmu(xsd~=0,onesNp);
  end
  bestB=bestB./xsd(xsd~=0,onesDr);
  if sum(xsd==0)>0,
    P=bestP;
    B=bestB;
    bestP=zeros(D,Np);
    bestP(xsd~=0,:)=P;
    bestB=zeros(D,Dr);
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

  Dr=work.Dr;
  Np=work.Np;
  Nx=work.Nx;
  onesDr=work.onesDr;
  sel=work.sel;
  prior=work.prior;

  %%% Compute distances %%%
  if work.dtype.euclidean,
    x2=sum((X.^2),1)';
    p2=sum((P.^2),1);
    ds=X'*P;
    ds=x2(:,work.onesNp)+p2(work.onesNx,:)-ds-ds;
  elseif work.dtype.cosine,
    psd=sqrt(sum(P.*P,1));
    P=P./psd(onesDr,:);
    xsd=sqrt(sum(X.*X,1));
    X=X./xsd(onesDr,:);
    ds=1-X'*P;
  elseif work.dtype.rtangent,
    ds=zeros(Nx,Np);
    nlp=1;
    for np=1:Np,
      dXP=X-P(:,np(work.onesNx));
      VdXP=work.Vp(:,nlp:nlp+work.L-1)'*dXP;
      ds(:,np)=(sum(dXP.*dXP,1)-sum(VdXP.*VdXP,1))';
      nlp=nlp+work.L;
    end
  elseif work.dtype.otangent,
    ds=zeros(Nx,Np);
    nlx=1;
    for nx=1:Nx,
      dXP=X(:,nx(work.onesNp))-P;
      VdXP=work.Vx(:,nlx:nlx+work.L-1)'*dXP;
      ds(nx,:)=sum(dXP.*dXP,1)-sum(VdXP.*VdXP,1);
      nlx=nlx+work.L;
    end
  elseif work.dtype.atangent,
    ds=zeros(Nx,Np);
    nlp=1;
    for np=1:Np,
      dXP=X-P(:,np(work.onesNx));
      VdXP=work.Vp(:,nlp:nlp+work.L-1)'*dXP;
      ds(:,np)=(sum(dXP.*dXP,1)-0.5*sum(VdXP.*VdXP,1))';
      nlp=nlp+work.L;
    end
    nlx=1;
    for nx=1:Nx,
      dXP=X(:,nx(work.onesNp))-P;
      VdXP=work.Vx(:,nlx:nlx+work.L-1)'*dXP;
      ds(nx,:)=ds(nx,:)-0.5*sum(VdXP.*VdXP,1);
      nlx=nlx+work.L;
    end
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

    fP=zeros(Dr,Np);

    if work.dtype.euclidean,
      Xs=(X-P(:,is)).*sfact(:,onesDr)';
      Xd=(X-P(:,id)).*dfact(:,onesDr)';
      fX=Xs-Xd;
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    elseif work.dtype.cosine,
      Xs=X.*sfact(:,onesDr)';
      Xd=X.*dfact(:,onesDr)';
      fX=P(:,id).*dfact(:,onesDr)'-P(:,is).*sfact(:,onesDr)';
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    elseif work.dtype.rtangent,
      Xs=(X-P(:,is)).*sfact(:,onesDr)';
      Xd=(X-P(:,id)).*dfact(:,onesDr)';
      ml=1;
      for m=1:Np,
        Vp=work.Vp(:,ml:ml+work.L-1);
        ml=ml+work.L;
        sel=is==m;
        Xs(:,sel)=Xs(:,sel)-Vp*(Vp'*Xs(:,sel));
        sel=id==m;
        Xd(:,sel)=Xd(:,sel)-Vp*(Vp'*Xd(:,sel));
      end
      fX=Xs-Xd;
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    elseif work.dtype.otangent,
      Xs=(X-P(:,is)).*sfact(:,onesDr)';
      Xd=(X-P(:,id)).*dfact(:,onesDr)';
      %nl=1;
      %for n=1:Nx,
      %  Vx=work.Vx(:,nl:nl+work.L-1);
      %  nl=nl+work.L;
      %  Xs(:,n)=Xs(:,n)-Vx*(Vx'*Xs(:,n));
      %  Xd(:,n)=Xd(:,n)-Vx*(Vx'*Xd(:,n));
      %end
      tXs=sum(work.Vx.*Xs(:,work.tidx));
      tXd=sum(work.Vx.*Xd(:,work.tidx));
      Xs=Xs-permute(sum(reshape(work.Vx.*tXs(onesDr,:),Dr,work.L,Nx),2),[1 3 2]);
      Xd=Xd-permute(sum(reshape(work.Vx.*tXd(onesDr,:),Dr,work.L,Nx),2),[1 3 2]);
      fX=Xs-Xd;
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    elseif work.dtype.atangent,
      Xs=(X-P(:,is)).*sfact(:,onesDr)';
      Xd=(X-P(:,id)).*dfact(:,onesDr)';
      oXs=Xs;
      oXd=Xd;
      ml=1;
      for m=1:Np,
        Vp=work.Vp(:,ml:ml+work.L-1);
        ml=ml+work.L;
        sel=is==m;
        Xs(:,sel)=Xs(:,sel)-0.5*Vp*(Vp'*oXs(:,sel));
        sel=id==m;
        Xd(:,sel)=Xd(:,sel)-0.5*Vp*(Vp'*oXd(:,sel));
      end
      %nl=1;
      %for n=1:Nx,
      %  Vx=work.Vx(:,nl:nl+work.L-1);
      %  nl=nl+work.L;
      %  Xs(:,n)=Xs(:,n)-0.5*Vx*(Vx'*oXs(:,n));
      %  Xd(:,n)=Xd(:,n)-0.5*Vx*(Vx'*oXd(:,n));
      %end
      tXs=sum(work.Vx.*oXs(:,work.tidx));
      tXd=sum(work.Vx.*oXd(:,work.tidx));
      Xs=Xs-0.5*permute(sum(reshape(work.Vx.*tXs(onesDr,:),Dr,work.L,Nx),2),[1 3 2]);
      Xd=Xd-0.5*permute(sum(reshape(work.Vx.*tXd(onesDr,:),Dr,work.L,Nx),2),[1 3 2]);
      fX=Xs-Xd;
      for m=1:Np,
        fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
      end
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [E, J, fX, fP] = ldpp_sindex(P, Plabels, X, Xlabels, work)

  Dr=work.Dr;
  Np=work.Np;
  Nx=work.Nx;
  onesDr=work.onesDr;
  overNx=work.overNx;

  %%% Compute distances %%%
  if work.dtype.euclidean,
    x2=sum((X.^2),1)';
    p2=sum((P.^2),1);
    ds=X'*P;
    ds=x2(:,work.onesNp)+p2(work.onesNx,:)-ds-ds;
  elseif work.dtype.cosine,
    psd=sqrt(sum(P.*P,1));
    P=P./psd(onesDr,:);
    xsd=sqrt(sum(X.*X,1));
    X=X./xsd(onesDr,:);
    ds=1-X'*P;
  end

  dd=ds;
  ssel=Plabels(:,work.onesNx)'==Xlabels(:,work.onesNp);
  ds(~ssel)=inf;
  dd(ssel)=inf;
  [ds,is]=min(ds,[],2);
  [dd,id]=min(dd,[],2);
  ds(ds==0)=realmin;
  dd(dd==0)=realmin;
  ratio=ds./dd;
  expon=exp(work.slope*(1-ratio));
  sigm=1./(1+expon);

  %%% Compute statistics %%%
  J=overNx*sum(sigm);
  E=overNx*sum(dd<ds);

  %%% Compute gradient %%%
  dsigm=work.slope.*expon./((1+expon).*(1+expon));
  ratio=overNx.*ratio;
  dfact=ratio.*dsigm;
  sfact=dfact./ds;
  dfact=dfact./dd;

  fP=zeros(Dr,Np);

  if work.dtype.euclidean,
    Xs=(X-P(:,is)).*sfact(:,onesDr)';
    Xd=(X-P(:,id)).*dfact(:,onesDr)';
    fX=Xs-Xd;
    for m=1:Np,
      fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
    end
  elseif work.dtype.cosine,
    Xs=X.*sfact(:,onesDr)';
    Xd=X.*dfact(:,onesDr)';
    fX=P(:,id).*dfact(:,onesDr)'-P(:,is).*sfact(:,onesDr)';
    for m=1:Np,
      fP(:,m)=-sum(Xs(:,is==m),2)+sum(Xd(:,id==m),2);
    end
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mu, ind] = kmeans(X, K)
  maxI=100;
  N=size(X,2);
  onesN=ones(N,1);
  onesK=ones(K,1);
  [k,pind]=sort(rand(N,1));
  mu=X(:,pind(1:K));

  I=0;
  while true,
    x2=sum((X.^2),1)';
    mu2=sum((mu.^2),1);
    dist=X'*mu;
    dist=x2(:,onesK)+mu2(onesN,:)-dist-dist;
    [dist,ind]=min(dist,[],2);

    if I==maxI || sum(ind~=pind)==0,
      break;
    end

    kk=unique(ind);
    if size(kk,1)~=K,
      for k=1:K,
        if sum(kk==k)==0,
          mu(:,k)=X(:,round((N-1)*rand)+1);
        end
      end
    end

    for k=kk',
      mu(:,k)=mean(X(:,ind==k),2);
    end

    pind=ind;
    I=I+1;
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = orthonorm(X)
  [X,dummy]=qr(X,0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = orthounit(X)
  [onX,dummy]=qr(X,0);
  X=onX.*repmat(sum(onX'*X,1),size(X,1),1);
  X=sqrt(size(X,2)).*X./sqrt(sum(diag(X'*X)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = orthotangs(X,N)
  for n=N:N:size(X,2),
    [XX,dummy]=qr(X(:,n-N+1:n),0);
    X(:,n-N+1:n)=XX;
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cfg = comptangs(B,P,cfg)
  %%% X tangents %%%
  if cfg.dtype.otangent || cfg.dtype.atangent,
    if cfg.knntangs && cfg.imgtangs,
      cfg.Vx(:,cfg.knntangsx)=tangVects(cfg.rX,cfg.knntypes,'Xlabels',cfg.Xlabels);
      cfg.Vx(:,~cfg.knntangsx)=B'*cfg.oVx;
      if cfg.devel,
        cfg.Vy(:,cfg.knntangsy)=tangVects(cfg.rY,cfg.knntypes,'Xlabels',cfg.Ylabels);
        cfg.Vy(:,~cfg.knntangsy)=B'*cfg.oVy;
      end
    elseif cfg.knntangs,
      cfg.Vx=tangVects(cfg.rX,cfg.knntypes,'Xlabels',cfg.Xlabels);
      if cfg.devel,
        cfg.Vy=tangVects(cfg.rY,cfg.knntypes,'Xlabels',cfg.Ylabels);
      end
    elseif cfg.imgtangs,
      cfg.Vx=B'*cfg.oVx;
      if cfg.devel,
        cfg.Vy=B'*cfg.oVy;
      end
    end
    cfg.Vx=orthotangs(cfg.Vx,cfg.L);
    if cfg.devel,
      cfg.Vy=orthotangs(cfg.Vy,cfg.L);
    end
  end
  %%% P tangents %%%
  if cfg.dtype.rtangent || cfg.dtype.atangent,
    if cfg.imgtangs,
      if cfg.normalize || cfg.linearnorm,
        P=P.*cfg.xsd(cfg.xsd~=0,cfg.onesNp)+cfg.xmu(cfg.xsd~=0,cfg.onesNp);
        if sum(cfg.xsd==0)>0,
          oP=P;
          P=cfg.xmu(:,cfg.onesNp);
          P(cfg.xsd~=0,:)=oP;
        end
      elseif whiten,
        P=cfg.IW'*P+cfg.xmu(:,cfg.onesNp);
      end
      imgVp=tangVects(P,cfg.imgtypes,cfg.imgtangcfg);
      onesNivp=ones(size(imgVp,2),1);
      if cfg.normalize || cfg.linearnorm,
        imgVp=(imgVp-cfg.xmu(:,onesNivp))./cfg.xsd(:,onesNivp);
        if sum(cfg.xsd==0)>0,
          imgVp(cfg.xsd==0,:)=[];
        end
      elseif whiten,
        imgVp=cfg.W'*(imgVp-cfg.xmu(:,onesNivp));
      end
    end
    if cfg.knntangs && cfg.imgtangs,
      cfg.Vp(:,cfg.knntangsp)=tangVects(cfg.rP,cfg.knntypes,'Xlabels',cfg.Plabels,'knnprotos',cfg.rX,cfg.Xlabels);
      cfg.Vp(:,~cfg.knntangsp)=B'*imgVp;
    elseif cfg.knntangs,
      cfg.Vp=tangVects(cfg.rP,cfg.knntypes,'Xlabels',cfg.Plabels,'knnprotos',cfg.rX,cfg.Xlabels);
    elseif cfg.imgtangs,
      cfg.Vp=B'*imgVp;
    end
    cfg.Vp=orthotangs(cfg.Vp,cfg.L);
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P0, Plabels] = ldpp_initP(X, Xlabels, Clabels, Npc, B)
  D=size(X,1);
  CxNpc=numel(Clabels)*Npc;
  P0=zeros(D,CxNpc);
  Plabels=zeros(CxNpc,1);
  n=1;
  if Npc>1,
    for c=Clabels',
      Xc=X(:,Xlabels==c);
      rXc=B'*Xc;
      [mu,ind]=kmeans(rXc,Npc);
      for k=1:Npc,
        P0(:,n)=mean(Xc(:,ind==k),2);
        Plabels(n)=c;
        n=n+1;
      end
    end
  else
    for c=Clabels',
      P0(:,n)=mean(X(:,Xlabels==c),2);
      Plabels(n)=c;
      n=n+1;
    end
  end
