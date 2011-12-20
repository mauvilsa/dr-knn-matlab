function [ B, V ] = pca( X, varargin )
%
% PCA: Principal Component Analysis
%
% Usage:
%   [ B, V ] = pca( X, ... )
%
% Input:
%   X                - Data matrix. Each column vector is a data point.
%
% Input (optional):
%   'auto'           - Choose algorithm automatically (default=true)
%   'cova'           - Use covariance matrix algorithm (default=false)
%   'grma'           - Use gram matrix algorithm (default=false)
%   'svda'           - Use SVD algorithm (default=false)
%   'tang',XTANGS    - Do tangent vector PCA (default=false)
%   'tfact',TFACT    - Importance of tangents (default=0.1)
%   'ptfact',PTFACT  - Importance of each tangent type (default=1)
%
% Output:
%   B                - Computed PCA basis
%   V                - Computed PCA eigenvalues
%
% $Revision$
% $Date$
%

% Copyright (C) 2008-2011 Mauricio Villegas (mvillegas AT iti.upv.es)
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

fn = 'pca:';
minargs = 1;

%%% File version %%%
if ischar(X)
  unix(['echo "$Revision$* $Date$*" | sed "s/^:/' fn ' revision/g; s/ : /[/g; s/ (.*)/]/g;"']);
  return;
end

%%% Default values %%%
B = [];
V = [];

auto = true;
cova = false;
grma = false;
svda = false;
tfact = 0.1;

logfile = 2;

%%% Input arguments parsing %%%
n = 1;
argerr = false;
while size(varargin,2)>0
  if ~ischar(varargin{n})
    argerr = true;
  elseif strcmp(varargin{n},'tfact') || ...
         strcmp(varargin{n},'ptfact') || ...
         strcmp(varargin{n},'tang')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  elseif strcmp(varargin{n},'auto') || ...
         strcmp(varargin{n},'cova') || ...
         strcmp(varargin{n},'grma') || ...
         strcmp(varargin{n},'svda')
    auto = false;
    cova = false;
    grma = false;
    svda = false;
    eval([varargin{n},'=true;']);
    n = n+1;
  else
    argerr = true;
  end
  if argerr || n>size(varargin,2)
    break;
  end
end

[ D, N ] = size(X);

if auto
  if N>=D || exist('tang','var')
    cova = true;
  else
    grma = true;
  end
end

%%% Error detection %%%
if argerr
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif exist('tang','var') && mod(size(tang,2),N)~=0
  fprintf(logfile,'%s error: number of tangents should be a multiple of the number of samples\n',fn);
  return;
elseif exist('tang','var') && exist('ptfact','var') && ...
       (numel(ptfact)~=size(tang,2)/N || size(ptfact,1)~=1)
  fprintf(logfile,'%s error: ptfact should be a row vector with the same number of elements as tangent types\n',fn);
  return;
end

if exist('tang','var') && ~cova
  fprintf(logfile,'%s warning: tangent vector PCA only possible with covariance algorithm\n',fn);
  cova = true;
end

mu = mean(X,2);

if cova
  %X = X-repmat(mu,1,N);
  %covm = (1/(N-1))*(X*X');
  covm = (1/(N-1))*(X*X'-N*mu*mu');
  %%covm = cov(X');
  if exist('tang','var')
    L = size(tang,2)/N;
    if exist('ptfact','var')
      tang = tang.*repmat(ptfact,D,N);
    end
    tcovm = (1/(L*N))*(tang*tang');
    tfact = tfact*trace(covm)/trace(tcovm);
    covm = covm+tfact.*tcovm;
  end
  [ B, V ] = eig(covm);
  V = real(diag(V));
  [ srt, idx ] = sort(-1*V);
  V = V(idx);
  B = B(:,idx);

elseif grma
  X = X-repmat(mu,1,N);
  grm = (1/(N-1))*(X'*X);
  [ A, V ] = eig(grm);
  V = real(diag(V));
  [ srt, idx ] = sort(-1*V);
  V = [ V(idx); zeros(D-N,1) ];
  A = A(:,idx);
  if sum(V<=eps)>0
    s = V(V<=eps);
    i = find(V==s(1));
    V(i:N) = 0;
    R = i-1;
  else
    R = N;
  end
  for n=1:R
    A(:,n) = (1/sqrt(N*V(n))).*A(:,n);
  end
  B = X*A;

elseif svda
  X = X-repmat(mu,1,N);
  [ A, V, B ] = svd(X');
  V = real(diag(V));
  [ srt, idx ] = sort(-1*V);
  V = V(idx);
  V = (1/(N-1)).*(V.*V);
  B = B(:,idx);

end
