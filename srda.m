function B = srda(X, Xlabels, varargin)
%
% SRDA: Spectral Regression Discriminant Analysis
%
% Usage:
%   B = srda(X, Xlabels, ...)
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

usetangs=false;
dosparse=false;
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
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
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

mu=mean(X,2);
X=X-repmat(mu,1,N);

%[Xlabels,idx]=sort(Xlabels);
%X=X(:,idx);

if exist('tang','var');
  usetangs=true;
  L=size(tang,2)/N;
  tfact=tfact/L;
  tang=sqrt(tfact)*tang;
  X=[X tang];
end

Clabels=unique(Xlabels)';
C=size(Clabels,2);
oXlabels=Xlabels;
Xlabels=ones(N,1);
for c=2:C,
  Xlabels(oXlabels==Clabels(c))=c;
end

%Nc=hist(Xlabels,[1:C]);
%Sc=cumsum(Nc);
%Sc=[0,Sc(1:end-1)];

%if usetangs,
%  W=zeros(N+N*L,N+N*L);
%else
%  W=zeros(N,N);
%end
%for c=1:C,
%  W(Sc(c)+1:Sc(c)+Nc(c),Sc(c)+1:Sc(c)+Nc(c))=1/Nc(c);
%end

%if true,
if false,

if usetangs,
  Y=zeros(N+N*L,C+1);
else
  Y=zeros(N,C+1);
end
Y(1:N,1)=1;
for c=1:C,
  Y(find(Xlabels==c),c+1)=1;
end
[Y,Sc]=qr(Y,0);
Y(:,1)=[];
Y(:,end)=[];

else % from d.cai code

rand('state',0);
Y=rand(C,C);
if usetangs,
  Z=zeros(N+N*L,C);
else
  Z=zeros(N,C);
end
for i=1:C
  idx=find(Xlabels==Clabels(i));
  Z(idx,:)=repmat(Y(i,:),length(idx),1);
end
Z(1:N,1)=ones(N,1);
[Y,R]=qr(Z,0);
Y(:,1)=[];

end

if (~usetangs && D>N) || (usetangs && D>N*L),
%if true,
%if false,
  fprintf(logfile,'%s inner product mode\n',fn);
  S=X'*X;
  %S=full(S);
  S([0:N-1]*N+[1:N])=S([0:N-1]*N+[1:N])+regu;
  S=max(S,S');
  R=chol(S);

  B=R\(R'\Y);
  B=X*B;

  %if KernelWay
  %  ddata = data*data';
  %  ddata = full(ddata);
  %  ddata = single(ddata);

  %  for i=1:size(ddata,1)
  %    ddata(i,i) = ddata(i,i) + options.ReguAlpha;
  %  end

  %  ddata = max(ddata,ddata');
  %  R = chol(ddata);
  %  eigvector = R\(R'\Responses);

  %  eigvector = double(eigvector);
  %  eigvector = data'*eigvector;

else

  fprintf(logfile,'%s outer product mode\n',fn);
  S=X*X';
  %S=full(S);
  S([0:D-1]*D+[1:D])=S([0:D-1]*D+[1:D])+regu;
  S=max(S,S');
  R=chol(S);

  B=X*Y;
  B=R\(R'\B);

  %else
  %  ddata = data'*data;
  %  ddata = full(ddata);
  %  ddata = single(ddata);

  %  for i=1:size(ddata,1)
  %    ddata(i,i) = ddata(i,i) + options.ReguAlpha;
  %  end

  %  ddata = max(ddata,ddata');
  %  B = data'*Responses;

  %  R = chol(ddata);
  %  eigvector = R\(R'\B);
  %  eigvector = double(eigvector);
  %end

end

B=B./repmat(sqrt(sum(B.^2,1)),D,1);

if dopca,
  B=pcab*B;
end
