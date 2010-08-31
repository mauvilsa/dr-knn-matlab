function [B, V] = srda(X, Xlabels, varargin)
%
% SRDA: Spectral Regression Discriminant Analysis
%
% Usage:
%   [B, V] = srda(X, Xlabels, ...)
%
% Input:
%   X              - Data matrix. Each column vector is a data point.
%   Xlabels        - Data class labels.
%
% Input (optional):
%   'dopca',DIM    - Perform PCA before SRDA (default=false)
%   'pcab',PCAB    - Supply the PCA basis
%   'regu',REGU    - Regularization factor (default=0.5)
%   'tang',XTANGS  - Do tangent vector SRDA (default=false)
%   'tfact',TFACT  - Importance of tangents (default=0.01)
%
% Output:
%   B              - Computed SRDA basis
%   V              - Computed SRDA eigenvalues
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
  unix('echo "$Revision$* $Date$*" | sed "s/^:/srda: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='srda:';
minargs=2;

%%% Default values %%%
B=[];
V=[];

brute=false;
dopca=false;
tfact=0.01;
regu=0.5;

logfile=2;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}),
    argerr=true;
  elseif strcmp(varargin{n},'tfact') || ...
         strcmp(varargin{n},'brute') || ...
         strcmp(varargin{n},'regu') || ...
         strcmp(varargin{n},'dopca') || ...
         strcmp(varargin{n},'pcab') || ...
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
  fprintf(logfile,'%s error: there must be the same number of labels as data points\n',fn);
  return;
elseif exist('tang','var') && mod(size(tang,2),N)~=0,
  fprintf(logfile,'%s error: number of tangents should be a multiple of the number of samples\n',fn);
  return;
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

if size(Xlabels,2)~=1,
  Xlabels=Xlabels';
end
Clabels=unique(Xlabels)';
C=size(Clabels,2);
oXlabels=Xlabels;
Xlabels=ones(N,1);
for c=2:C,
  Xlabels(oXlabels==Clabels(c))=c;
end

mu=mean(X,2);
Xo=X-repmat(mu,1,N);

%compW=false;
%if compW,
%  [Xlabels,idx]=sort(Xlabels);
%  X=X(:,idx);

%  Nc=hist(Xlabels,[1:C]);
%  Sc=cumsum(Nc);
%  Sc=[0,Sc(1:end-1)];

%  if exist('tang','var'),
%    W=zeros(N+N*L,N+N*L);
%  else
%    W=zeros(N,N);
%  end
%  for c=1:C,
%    W(Sc(c)+1:Sc(c)+Nc(c),Sc(c)+1:Sc(c)+Nc(c))=1/Nc(c);
%  end
%end

%randeig=true;
randeig=false;
if ~randeig, % from d.cai paper

  Nc=hist(Xlabels,[1:C]);
  Y=zeros(C);
  Y([0:C-1]*C+[1:C])=Nc.^(-1/2);
  [Y,Nc]=qr([ones(N,1) Y(Xlabels,:)],0);
  Y(:,1)=[];
  Y(:,end)=[];

else % from d.cai code

  rand('state',0);
  %rand('state',12345678);
  Y=rand(C,C);
  Z=zeros(N,C);
  for c=1:C,
    idx=find(Xlabels==c);
    Z(idx,:)=repmat(Y(c,:),length(idx),1);
  end
  Z(1:N,1)=ones(N,1);
  [Y,R]=qr(Z,0);
  Y(:,1)=[];

end

if exist('tang','var'),
  L=size(tang,2)/N;
  tang=sqrt(tfact/L)*tang;
  Xo=[Xo tang];
  Y=[Y; zeros(N*L,C-1)];
end

if brute,
  %fprintf(logfile,'%s brute force mode\n',fn);

  %ST=Xo*Xo';
  ST=(1/N)*(Xo*Xo');
  %ST=ST+regu*eye(D);
  ST=ST+regu*trace(ST)*eye(D);
  SB=zeros(D);
  for c=1:C,
    sel=Xlabels==c;
    muc=mean(Xo(:,sel),2);
    %SB=SB+sum(sel).*muc*muc';
    SB=SB+(sum(sel)/N).*muc*muc';
  end

  [B,V]=eig(SB,ST);
  V=real(diag(V));
  [srt,idx]=sort(-1*V);
  idx=idx(1:min([D,C-1]));
  V=V(idx);
  B=B(:,idx);
  B=B.*repmat(1./sqrt(sum(B.*B,1)),D,1);

elseif (~exist('tang','var') && D>N) || ...
       ( exist('tang','var') && D>N*L),
  %fprintf(logfile,'%s inner product mode\n',fn);

  %S=Xo'*Xo;
  S=(1/N)*(Xo'*Xo);
  %S([0:N-1]*N+[1:N])=S([0:N-1]*N+[1:N])+regu;
  %S([0:N-1]*N+[1:N])=S([0:N-1]*N+[1:N])+regu*trace(S); % this is wrong? trace increases proportional to N instead of D
  S([0:N-1]*N+[1:N])=S([0:N-1]*N+[1:N])+(D/N)*regu*trace(S);
  R=chol(S);

  B=Xo*(R\(R'\Y));

else
  %fprintf(logfile,'%s outer product mode\n',fn);

  %S=Xo*Xo';
  S=(1/N)*(Xo*Xo');
  %S([0:D-1]*D+[1:D])=S([0:D-1]*D+[1:D])+regu;
  S([0:D-1]*D+[1:D])=S([0:D-1]*D+[1:D])+regu*trace(S);
  R=chol(S);

  B=R\(R'\(Xo*Y));

end

if ~brute,
  V=sqrt(sum(B.*B,1))';
  B=B.*repmat(1./V',D,1);

  if nargout>1,
    Xo=B'*(X-repmat(mu,1,N));
    ST=(1/N)*(Xo*Xo');
    SB=zeros(size(B,2));
    for c=1:C,
      sel=Xlabels==c;
      muc=mean(Xo(:,sel),2);
      SB=SB+(sum(sel)/N).*muc*muc';
    end
    V=diag(SB)./diag(ST);
  end

  [srt,idx]=sort(-1*V);
  V=V(idx);
  B=B(:,idx);
end

if dopca,
  B=pcab*B;
end
