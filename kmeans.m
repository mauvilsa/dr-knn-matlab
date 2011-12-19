function [ mu, idx ] = kmeans( X, K, varargin )
%
% KMEANS: K-means Algorithm
%
% [ mu, idx ] = kmeans( X, K, ... )
%
%   Input:
%     X       - Data matrix. Each column vector is a data point.
%     K       - Number of means.
%
%   Input (optional):
%     'mu0',mu                   - Specify initial means (defaul=false)
%     'maxI',MAXI                - Maximum number of iterations (default=100)
%     'seed',SEED                - Random seed (default=system)
%
%   Output:
%     mu      - Final learned means
%     idx     - Indexes of the learned means
%
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

fn = 'kmeans:';
minargs = 2;

%%% File version %%%
if ischar(X)
  unix(['echo "$Revision$* $Date$*" | sed "s/^:/' fn ' revision/g; s/ : /[/g; s/ (.*)/]/g;"']);
  return;
end

minargs=2;

[ D, N ] = size(X);

if K==1
  mu = mean(X,2);
  if nargout>1
    idx = ones(N,1);
  end
  return;
elseif K==N
  mu = X;
  if nargout>1
    idx = [1:N]';
  end
  return;
end

maxI = 100;
logfile = 2;
verbose = false;

n = 1;
argerr = false;
while size(varargin,2)>0
  if ~ischar(varargin{n}) || size(varargin,2)<n+1
    argerr = true;
  elseif strcmp(varargin{n},'fastmode')
    eval([varargin{n},'=varargin{n+1};']);
    n = n+2;
  elseif strcmp(varargin{n},'mu0') || ...
         strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'seed')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  elseif strcmp(varargin{n},'verbose')
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  else
    argerr = true;
  end
  if argerr || n>size(varargin,2)
    break;
  end
end

if exist('mu0','var')
  [ D0, K0 ] = size(mu0);
else
  D0 = D;
  K0 = K;
end

if exist('fastmode','var')
  logfile = fastmode.logfile;
  verbose = true;
  fastmode = true;
else
  fastmode = false;
end

if fastmode
elseif argerr
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif K>N
  fprintf(logfile,'%s error: K must be lower than the number of data points\n',fn);
  return;
elseif D0~=D || K0~=K
  fprintf(logfile,'%s error: incompatible size of initial means\n',fn);
  return;
end

if ~verbose
  logfile = fopen('/dev/null');
end

pidx = zeros(N,1);
if ~exist('mu0','var')
  if exist('seed','var')
    rand('state',seed);
  end
  [ k, pidx ] = sort(rand(N,1));
  mu = X(:,pidx(1:K));
else
  mu = mu0;
end

cfg.dtype.euclidean = true;

It = 0;

tic;

while true

  dst = dstmat(mu,X,cfg);
  [ dst, idx ] = min(dst,[],2);
  
  fprintf(logfile,'%d %f\n',It,sum(dst)/N);

  if It==maxI
    fprintf(logfile,'%s reached maximum number of iterations\n',fn);
    break;
  end

  if sum(idx~=pidx)==0,
    fprintf(logfile,'%s algorithm converged\n',fn);
    break;
  end

  kk = unique(idx);
  if size(kk,1)~=K
    fprintf(logfile,'%s warning: %d empty means, setting them to random samples (I=%d)\n',fn,K-size(kk,1),It);
    for k=1:K
      if sum(kk==k)==0
        mu(:,k) = X(:,round((N-1)*rand)+1);
      end
    end
  end

  for k=kk'
    mu(:,k) = mean(X(:,idx==k),2);
  end

  pidx = idx;
  It = It+1;
end

fprintf(logfile,'%s number of iterations: %d\n',fn,It);
fprintf(logfile,'%s average iteration time (s): %f\n',fn,toc/It);

if ~verbose
  fclose(logfile);
end
