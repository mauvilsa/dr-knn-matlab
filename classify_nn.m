function [E, A, S, d] = classify_nn(P, Plabels, X, Xlabels, varargin)
%
% CLASSIFY_NN: Classify using Nearest Neighbor
%
% Usage:
%   [E, A, S, dists] = classify_nn(P, Plabels, X, Xlabels, ...)
%
% Input:
%   P        - Prototypes data matrix. Each column vector is a data point.
%   Plabels  - Prototypes labels.
%   X        - Testing data matrix. Each column vector is a data point.
%   Xlabels  - Testing data class labels.
%
% Input (optional):
%   'perclass',(true|false)   - Compute error/score for each class (default=false)
%   'euclidean'               - Euclidean distance (default=true)
%   'cosine'                  - Cosine distance (default=false)
%   'tangent'                 - Tangent distance (default=false)
%   'rtangent'                - Ref. tangent distance (default=false)
%   'otangent'                - Obs. tangent distance (default=false)
%   'atangent'                - Avg. tangent distance (default=false)
%   'tangVp',tangVp           - Tangent bases of prototypes
%   'tangVx',tangVx           - Tangent bases of testing data
%
% Output:
%   E        - Classification error
%   A        - Assigned Class
%   S        - Classification score
%   dists    - Pairwise distances
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

if strncmp(P,'-v',2),
  unix('echo "$Revision$* $Date$*" | sed "s/^:/classify_nn: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='classify_nn:';
minargs=4;

E=[];
A=[];
S=[];

[D,Np]=size(P);
Nx=size(X,2);
if size(Plabels,1)<size(Plabels,2),
  Plabels=Plabels';
end
if size(Xlabels,1)<size(Xlabels,2),
  Xlabels=Xlabels';
end

perclass=false;
dtype.euclidean=true;
dtype.cosine=false;
dtype.tangent=false;
dtype.rtangent=false;
dtype.otangent=false;
dtype.atangent=false;

logfile=2;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}),
    argerr=true;
  elseif strcmp(varargin{n},'perclass'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'euclidean') || ...
         strcmp(varargin{n},'tangent') || ...
         strcmp(varargin{n},'rtangent') || ...
         strcmp(varargin{n},'otangent') || ...
         strcmp(varargin{n},'atangent') || ...
         strcmp(varargin{n},'cosine'),
    dtype.euclidean=false;
    dtype.cosine=false;
    dtype.tangent=false;
    dtype.rtangent=false;
    dtype.otangent=false;
    dtype.atangent=false;
    eval(['dtype.',varargin{n},'=true;']);
    n=n+1;
  elseif strcmp(varargin{n},'tangVp') || ...
         strcmp(varargin{n},'tangVx'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}),
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
elseif size(X,1)~=D,
  fprintf(logfile,'%s error: dimensionality prototypes and data must be the same\n',fn);
  return;
elseif size(Plabels,1)~=Np || size(Plabels,2)~=1 || ...
      (sum(size(Xlabels))~=0&&(size(Xlabels,1)~=Nx || size(Xlabels,2)~=1)),
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
elseif ~exist('tangVp','var') && (dtype.tangent || dtype.atangent || dtype.rtangent),
  fprintf(logfile,'%s error: tangents of prototypes should be given\n',fn);
  return;
elseif ~exist('tangVx','var') && (dtype.tangent || dtype.atangent || dtype.otangent),
  fprintf(logfile,'%s error: tangents of testing data should be given\n',fn);
  return;
elseif (exist('tangVp','var') && mod(size(tangVp,2),Np)~=0) || ...
       (exist('tangVx','var') && mod(size(tangVx,2),Nx)~=0),
  fprintf(logfile,'%s error: number of tangents should be a multiple of the number of samples\n',fn);
  return;
end

onesNp=ones(Np,1);
onesNx=ones(Nx,1);
onesD=ones(D,1);

if exist('tangVp','var'),
  Lp=size(tangVp,2)/Np;
  if sum(sum(eye(Lp)-round(1000*tangVp(:,1:Lp)'*tangVp(:,1:Lp))./1000))~=0,
    fprintf(logfile,'%s warning: tangVp not orthonormal, orthonormalizing ...\n',fn);
    for nlp=1:Lp:size(tangVp,2),
      tangVp(:,nlp:nlp+Lp-1)=orthonorm(tangVp(:,nlp:nlp+Lp-1));
    end
  end
end
if exist('tangVx','var'),
  Lx=size(tangVx,2)/Nx;
  if sum(sum(eye(Lx)-round(1000*tangVx(:,1:Lx)'*tangVx(:,1:Lx))./1000))~=0,
    fprintf(logfile,'%s warning: tangVx not orthonormal, orthonormalizing ...\n',fn);
    for nlx=1:Lx:size(tangVx,2),
      tangVx(:,nlx:nlx+Lx-1)=orthonorm(tangVx(:,nlx:nlx+Lx-1));
    end
  end
end

% euclidean distance
if dtype.euclidean,
  x2=sum((X.^2),1)';
  p2=sum((P.^2),1);
  d=X'*P;
  d=x2(:,onesNp)+p2(onesNx,:)-d-d;
% cosine distance
elseif dtype.cosine,
  psd=sqrt(sum(P.*P,1));
  P=P./psd(onesD,:);
  xsd=sqrt(sum(X.*X,1));
  X=X./xsd(onesD,:);
  d=1-X'*P;
% reference single sided tangent distance
elseif dtype.rtangent,
  d=zeros(Nx,Np);
  Lp=size(tangVp,2)/Np;
  nlp=1;
  for np=1:Np,
    dXP=X-P(:,np(onesNx));
    VdXP=tangVp(:,nlp:nlp+Lp-1)'*dXP;
    d(:,np)=(sum(dXP.*dXP,1)-sum(VdXP.*VdXP,1))';
    nlp=nlp+Lp;
  end
% observation single sided tangent distance
elseif dtype.otangent,
  d=zeros(Nx,Np);
  Lx=size(tangVx,2)/Nx;
  nlx=1;
  for nx=1:Nx,
    dXP=X(:,nx(onesNp))-P;
    VdXP=tangVx(:,nlx:nlx+Lx-1)'*dXP;
    d(nx,:)=sum(dXP.*dXP,1)-sum(VdXP.*VdXP,1);
    nlx=nlx+Lx;
  end
% average single sided tangent distance
elseif dtype.atangent,
  d=zeros(Nx,Np);
  Lp=size(tangVp,2)/Np;
  nlp=1;
  for np=1:Np,
    dXP=X-P(:,np(onesNx));
    VdXP=tangVp(:,nlp:nlp+Lp-1)'*dXP;
    d(:,np)=(sum(dXP.*dXP,1)-0.5*sum(VdXP.*VdXP,1))';
    nlp=nlp+Lp;
  end
  Lx=size(tangVx,2)/Nx;
  nlx=1;
  for nx=1:Nx,
    dXP=X(:,nx(onesNp))-P;
    VdXP=tangVx(:,nlx:nlx+Lx-1)'*dXP;
    d(nx,:)=d(nx,:)-0.5*sum(VdXP.*VdXP,1);
    nlx=nlx+Lx;
  end
% tangent distance
elseif dtype.tangent,
  d=zeros(Nx,Np);
  Lp=size(tangVp,2)/Np;
  Lx=size(tangVx,2)/Nx;
  tangVpp=zeros(Lp,Lp*Np);
  itangVpp=zeros(Lp,Lp*Np);
  tangVxx=zeros(Lx,Lx*Nx);
  itangVxx=zeros(Lx,Lx*Nx);
  nlp=1;
  for np=1:Np,
    sel=nlp:nlp+Lp-1;
    Vp=tangVp(:,sel);
    tangVpp(:,sel)=Vp'*Vp;
    itangVpp(:,sel)=inv(tangVpp(:,sel));
    nlp=nlp+Lp;
  end
  nlx=1;
  for nx=1:Nx,
    sel=nlx:nlx+Lx-1;
    Vx=tangVx(:,sel);
    tangVxx(:,sel)=Vx'*Vx;
    itangVxx(:,sel)=inv(tangVxx(:,sel));
    nlx=nlx+Lx;
  end
  nlx=1;
  for nx=1:Nx,
    sel=nlx:nlx+Lx-1;
    nlx=nlx+Lx;
    Vx=tangVx(:,sel);
    Vxx=tangVxx(:,sel);
    iVxx=itangVxx(:,sel);
    x=X(:,nx);
    nlp=1;
    for np=1:Np,
      sel=nlp:nlp+Lp-1;
      nlp=nlp+Lp;
      Vp=tangVp(:,sel);
      Vpp=tangVpp(:,sel);
      iVpp=itangVpp(:,sel);
      p=P(:,np);
      Vpx=Vp'*Vx;
      Alp=(Vpx*iVxx*Vx'-Vp')*(x-p);
      Arp=Vpx*iVxx*Vpx'-Vpp;
      Alx=(Vpx'*iVpp*Vp'-Vx')*(x-p);
      Arx=Vxx-Vpx'*iVpp*Vpx;
      ap=Arp\Alp;
      ax=Arx\Alx;
      xx=x+Vx*ax;
      pp=p+Vp*ap;
      d(nx,np)=(xx-pp)'*(xx-pp);
    end
  end
end

Clabels=unique(Plabels)';
Cp=max(size(Clabels));

dist=zeros(Nx,Cp);
c=1;
for label=Clabels,
  csel=Plabels==label;
  dist(:,c)=min(d(:,csel),[],2);
  c=c+1;
end

[idist,A]=min(dist,[],2);

if nargout>2,
  idist=dist;
  idist(idist==0)=realmin;
  idist=1./idist;
  if perclass,
    S=idist./repmat(sum(idist,2),1,Cp);
  else
    S=idist(Nx*(A-1)+[1:Nx]')./sum(idist,2);
  end
end

if sum(size(Xlabels))~=0,
  if perclass,
    E=zeros(Cp,1);
    c=1;
    for label=Clabels,
      sel=Xlabels==label;
      E(c)=sum(Clabels(A(sel))'~=label)/sum(sel);
      c=c+1;
    end
  else
    E=sum(Clabels(A)'~=Xlabels)/Nx;
  end
end

if nargout>1,
  A=Clabels(A)';
end
