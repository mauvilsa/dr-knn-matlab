function [B, V] = lda(X, Xlabels, varargin)
%
% LDA: Linear Discriminant Analysis
%
% Usage:
%   [B, V] = lda(X, Xlabels, ...)
%
% Input:
%   X                   - Data matrix. Each column vector is a data point.
%   Xlabels             - Data class labels.
%
% Input (optional):
%   'dopca',DIM         - Perform PCA before LDA (default=false)
%   'pcab',PCAB         - Supply the PCA basis
%   'tang',XTANGS       - Do tangent vector LDA (default=false)
%   'tfact',TFACT       - Importance of tangents (default=0.1)
%   'ptfact',PTFACT     - Importance of each tangent type (default=1)
%   'nda',(true|alpha)  - Use nonparametric between scatter (default=false)
%
% Output:
%   B                   - Computed LDA basis
%   V                   - Computed LDA eigenvalues
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
  unix('echo "$Revision$* $Date$*" | sed "s/^:/lda: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='lda:';
minargs=2;

%%% Default values %%%
B=[];
V=[];

dopca=false;
tfact=0.1;
nda=false;

logfile=2;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}),
    argerr=true;
  elseif strcmp(varargin{n},'tfact') || ...
         strcmp(varargin{n},'ptfact') || ...
         strcmp(varargin{n},'dopca') || ...
         strcmp(varargin{n},'pcab') || ...
         strcmp(varargin{n},'nda') || ...
         strcmp(varargin{n},'tang'),
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

[D,N]=size(X);

%%% Error detection %%%
if argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif max(size(Xlabels))~=N || min(size(Xlabels))~=1,
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
elseif exist('tang','var') && mod(size(tang,2),N)~=0,
  fprintf(logfile,'%s error: number of tangents should be a multiple of the number of samples\n',fn);
  return;
elseif exist('tang','var') && exist('ptfact','var') && ...
       (numel(ptfact)~=size(tang,2)/N || size(ptfact,1)~=1),
  fprintf(logfile,'%s error: ptfact should be a row vector with the same number of elements as tangent types\n',fn);
  return;
end

if exist('tang','var') && exist('ptfact','var'),
  tang=tang.*repmat(ptfact,D,N);
end

if exist('pcab','var') && ~dopca,
  dopca=size(pcab,2);
end
if dopca;
  if ~exist('pcab','var'),
    if ~exist('tang','var'),
      pcab=pca(X);
    else
      pcab=pca(X,'tang',tang,'tfact',tfact);
    end
  end
  if dopca>size(pcab,2) || dopca<1,
    fprintf(logfile,'%s error: inconsistent dimensions in PCA base\n',fn);
    return;
  end
  pcab=pcab(:,1:dopca);
  mu=mean(X,2);
  X=X-repmat(mu,1,N);
  X=pcab'*X;
  D=dopca;
  if exist('tang','var');
    tang=tang-repmat(mu,1,size(tang,2));
    tang=pcab'*tang;
  end
end

Clabels=unique(Xlabels)';
C=size(Clabels,2);

if exist('tang','var');
  L=size(tang,2)/N;
  tSW=zeros(D);
end

mu=mean(X,2);
SB=zeros(D);
SW=zeros(D);

if nda || ~islogical(nda),
  onesN=ones(N,1);
  X2=sum((X.^2),1);
  dd=X'*X;
  dd=X2(onesN,:)+X2(onesN,:)'-dd-dd;
  dd([0:N-1].*N+[1:N])=inf;
  sel=Xlabels(:,onesN)'==Xlabels(:,onesN);
  ds=dd;
  dd(sel)=inf;
  [dd,id]=min(dd,[],2);
  Xd=X-X(:,id);
  if ~islogical(nda),
    ds(~sel)=inf;
    [ds,is]=min(ds,[],2);
    ds=sqrt(ds).^nda;
    dd=sqrt(dd).^nda;
    w=sqrt((min(dd,ds)./(dd+ds)))';
    Xd=Xd.*w(ones(D,1),:);
  end
  SB=Xd*Xd';
  nda=true;
end

for c=Clabels,
  sel=Xlabels==c;
  Xc=X(:,sel);
  Nc=size(Xc,2);
  muc=mean(Xc,2);
  Xc=Xc-repmat(muc,1,Nc);
  SW=SW+Xc*Xc';
  if ~nda,
    muc=muc-mu;
    SB=SB+Nc.*muc*muc';
  end
end
if exist('tang','var'),
  tSW=(1/L).*(tang*tang');
  tfact=tfact*trace(SW)/trace(tSW);
  SW=(1/N).*(SW+tfact.*tSW);
else
  SW=(1/N).*SW;
end
SB=(1/N).*SB;

%[B,V]=eig(inv(SW)*SB);
[B,V]=eig(SB,SW);
V=real(diag(V));
[srt,idx]=sort(-1*V);
if ~nda,
  idx=idx(1:min([D,C-1]));
end
V=V(idx);
B=B(:,idx);
B=B.*repmat(1./sqrt(sum(B.*B,1)),D,1);

if dopca,
  B=pcab*B;
end

if sum(isinf(V))>0,
  fprintf(logfile,'%s error: infinite generalized eigenvalues\n',fn);
end
