function [B, V] = pca(X, varargin)
%
% PCA: Principal Component Analysis
%
% [B, V] = pca(X, ['algorithm'])
%
%   Input:
%     X       - Data matrix. Each column vector is a data point.
%
%   Input (optional):
%     'automatic'   - Choose algorithm automatically
%     'covariance'  - Use eigenvalue decomposition of covariance matrix
%     'gram'        - Use eigenvalue decomposition of gram matrix
%     'svd'         - Use singular value decomposition of data
%
%   Output:
%     B             - Computed PCA basis
%
%
% Version: 1.00 -- Apr/2008
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

algorithm='automatic';

if size(varargin,2)>0 && ischar(varargin{1})
  algorithm=varargin{1};
end

[D,N]=size(X);
mu=mean(X,2);
X=X-repmat(mu,1,N);

if strcmp(algorithm,'automatic'),
  if N<D,
    algorithm='gram';
  else
    algorithm='covariance';
  end
end

switch lower(algorithm),
  case 'covariance'
    [B,V]=eig(X*X');
    V=(1/N)*real(diag(V));
    [j,i]=sort(-1*V);
    V=V(i);
    B=B(:,i);
  case 'gram'
    epsilon=1e-10;
    [A,V]=eig(X'*X);
    V=real(diag(V));
    [j,i]=sort(-1*V);
    V=[V(i);zeros(D-N,1)];
    A=A(:,i);
    if sum(V<=epsilon)>0
      s=V(V<=epsilon);
      i=find(V==s(1));
      V(i:N)=0;
      R=i-1;
    else
      R=N;
    end
    for n=1:R
      A(:,n)=(1/sqrt(V(n))).*A(:,n);
    end
    B=X*A;
    V=(1/N)*V;
  case 'svd'
    [A,V,B]=svd(X');
    V=real(diag(V));
    [j,i]=sort(-1*V);
    V=V(i);
    V=(1/N).*(V.*V);
    B=B(:,i);
  otherwise
    fprintf(logfile,'pca: error: incorrect input argument\n');
end
