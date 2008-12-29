function B = lda(X, Xlabels, varargin)
%
% LDA: Linear Discriminant Analysis
%
% [B, V] = lda(X, Xlabels, ...)
%
%   Input:
%     X             - Data matrix. Each column vector is a data point.
%     Xlabels       - Data class labels.
%
%   Input (optional):
%     'dopca',DIM   - Perform PCA before LDA
%
%   Output:
%     B             - Computed LDA basis
%
%
% Version: 1.00 -- Jul/2008
%

%
%   Copyright (C) 2008 Mauricio Villegas (mvillegas AT iti.upv.es)
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program. If not, see <http://www.gnu.org/licenses/>.
%

dopca=false;
dimpca=0;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'dopca'),
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

[D,N]=size(X);

if argerr,
  fprintf(2,'lda: error: incorrect input argument (%d-%d)\n',n+5,n+6);
elseif max(size(Xlabels))~=N || min(size(Xlabels))~=1,
  fprintf(2,'lda: error: Xlabels must be a vector with size the same as the number of data points\n');
else

  if dopca;
    Bpca=pca(X);
    Bpca=Bpca(:,1:dimpca);
    mu=mean(X,2);
    X=X-repmat(mu,1,N);
    X=Bpca'*X;
    D=dimpca;
  end

  Clabels=unique(Xlabels)';
  C=size(Clabels,2);
  
  mu=mean(X,2);
  SB=zeros(D);
  SW=zeros(D);
  for c=Clabels,
    Xc=X(:,Xlabels==c);
    Nc=size(Xc,2);
    muc=mean(Xc,2);
    Xc=Xc-repmat(muc,1,Nc);
    SW=SW+Xc*Xc';
    muc=muc-mu;
    SB=SB+Nc.*(muc*muc');
  end
  [B,V]=eig(inv(SW)*SB);
  V=real(diag(V));
  [j,i]=sort(-1*V);
  P=min([D,C-1]);
  i=i(1:P);
  V=V(i);
  B=B(:,i);

  if dopca,
    B=Bpca*B;
  end

end
