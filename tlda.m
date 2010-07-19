function [B V] = tlda(X, Xlabels, varargin)
%
% TLDA: Tangent Linear Discriminant Analysis
%
% [B, V] = tlda(X, Xlabels, ...)
%
%   Input:
%     X             - Data matrix. Each column vector is a data point.
%     Xlabels       - Data class labels.
%
%   Input (optional):
%     'tangs',TYPES   - Tangent types (hvrspdtADHV)
%     'sigma',SIGMA   - Importance of tangents in scatter (default=0.1)
%     'imSize',[W H]  - Image size (default=square)
%     'bw',BW         - Tangent derivative gaussian bandwidth (default=0.5)
%     'krh',KRH       - Supply tangent derivative kernel horizontal (default=false)
%     'krv',KRV       - Supply tangent derivative kernel vertical (default=false)
%     'dopca',DIMS    - Perform PCA before TLDA
%     'pcab',PCAB     - Supply the PCA basis
%
%   Output:
%     B               - Computed TLDA basis
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
  unix('echo "$Revision$* $Date$*" | sed "s/^:/tlda: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='tlda:';
minargs=2;

%%% Default values %%%
tangs='hvrs';
dopca=false;
sigma=0.1;
bw=0.5;
logfile=2;
verbose=true;

fprintf(logfile,'%s after default\n',fn);

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'sigma') || ...
         strcmp(varargin{n},'imSize') || ...
         strcmp(varargin{n},'bw') || ...
         strcmp(varargin{n},'krh') || ...
         strcmp(varargin{n},'krw') || ...
         strcmp(varargin{n},'dopca') || ...
         strcmp(varargin{n},'pcab') || ...
         strcmp(varargin{n},'logfile'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}),% || sum(sum(varargin{n+1}<0))~=0,
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'verbose'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'tangs'),
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

fprintf(logfile,'%s after parse\n',fn);

[D,Nx]=size(X);
L=size(tangs,2);

%%% Error detection %%%
if argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif max(size(Xlabels))~=Nx || min(size(Xlabels))~=1,
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

fprintf(logfile,'%s after error det\n',fn);

if ~exist('krv','var') || ~exist('krh','var'),
  krW=floor(16*bw);
  if krW<3,
    krW=3;
  end
  krH=krW;
  [krh,krv]=imgDervKern(krW,krH,bw);
end

keyboard

fprintf(logfile,'%s after kr\n',fn);

if ~exist('imSize','var'),
  imW=round(sqrt(D));
  imH=round(D/imW);
  imSize=[imW imH];
end

fprintf(logfile,'%s after imSize\n',fn);


V=imgTang(X,imSize,krh,krv,tangs,false);

fprintf(logfile,'%s after tang\n',fn);

keyboard
if dopca,
  mu=mean(X,2);
  X=X-repmat(mu,1,Nx);
  if ~exist('pcab','var'),
    %pcab=pca(X);
    %                |--- ok ---|
    [pcab,pcav]=eig((1/Nx)*(X*X')+(2*sigma*sigma/(Nx*L))*(V*V'));
    pcav=real(diag(pcav));
    [j,i]=sort(-1*pcav);
    %pcav=pcav(i);
    pcab=pcab(:,i);
  end
  pcab=pcab(:,1:dopca);
  X=pcab'*X;
  D=dopca;
end

Clabels=unique(Xlabels)';
C=size(Clabels,2);

mu=mean(X,2);
SB=zeros(D);
SW=zeros(D);
for c=Clabels,
  sel=Xlabels==c;
  Xc=X(:,sel);
  Nc=size(Xc,2);
  muc=mean(Xc,2);
  Xc=Xc-repmat(muc,1,Nc);
  sel=sel(:,ones(L,1))';
  sel=sel(:);
  Vc=V(:,sel);
  if dopca,
    Vc=pcab'*Vc;
  end
  SW=SW+(1/Nx).*((Xc*Xc')+(2*sigma*sigma/L).*(Vc*Vc'));
  muc=muc-mu;
  SB=SB+(Nc/Nx).*(muc*muc');
end
[B,V]=eig(inv(SW)*SB);
V=real(diag(V));
[j,i]=sort(-1*V);
P=min([D,C-1]);
i=i(1:P);
V=V(i);
B=B(:,i);

if dopca,
  B=pcab*B;
end
