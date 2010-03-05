function [mu, ind] = kmeans(X, K, varargin)
%
% KMEANS: K-means Algorithm
%
% [mu, ind] = kmeans(X, K, ...)
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
%     ind     - Indexes of the learned means
%
%
% Version: 1.02 -- Sep/2009
%

%
% Copyright (C) 2008 Mauricio Villegas (mvillegas AT iti.upv.es)
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

[D,N]=size(X);

if K==1,
  mu=mean(X,2);
  if nargout>1,
    ind=ones(N,1);
  end
  return;
elseif K==N,
  mu=X;
  if nargout>1,
    ind=[1:N]';
  end
  return;
end

maxI=100;
logfile=2;
verbose=0;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'mu0') || ...
         strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'verbose') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'seed'),
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

if exist('mu0','var'),
  [D0, K0]=size(mu0);
else
  D0=D;
  K0=K;
end

if argerr,
  fprintf(logfile,'kmeans: error: incorrect input argument (%d %d)\n',varargin{n},varargin{n+1});
elseif nargin-size(varargin,2)~=2,
  fprintf(logfile,'kmeans: error: not enough input arguments\n');
elseif K>N,
  fprintf(logfile,'kmeans: error: K must be lower than the number of data points\n');
elseif D0~=D || K0~=K,
  fprintf(logfile,'kmeans: error: incompatible size of initial means\n');
else

  ind=zeros(N,1);
  if ~exist('mu0','var'),
    if exist('seed','var'),
      rand('state',seed);
    end
    [k,ind]=sort(rand(N,1));
    mu=X(:,ind(1:K));
  else
    mu=mu0;
  end

  I=0;

  tic;

  while 1,

    pind=ind;
    ind=ones(N,1);
    dist=sum(power(X-mu(:,ones(N,1)),2));
    for k=2:K,
      distk=sum(power(X-mu(:,k*ones(N,1)),2));
      ind(distk<dist)=k;
      dist(distk<dist)=distk(distk<dist);
    end

    if verbose>0,
      fprintf(logfile,'%d %f\n',I,sum(dist)/N);
    end

    if I==maxI,
      if verbose>0,
        fprintf(logfile,'kmeans: reached maximum number of iterations\n');
      end
      break;
    end

    if sum(ind~=pind)==0,
      if verbose>0,
        fprintf(logfile,'kmeans: algorithm converged\n');
      end
      break;
    end

    kk=unique(ind);
    if size(kk,1)~=K,
      if verbose>0,
        fprintf(logfile,'kmeans: warning: %d empty means, setting them to random samples (I=%d)\n',K-size(kk,1),I);
      end
      for k=1:K,
        if sum(kk==k)==0,
          mu(:,k)=X(:,round((N-1)*rand)+1);
        end
      end
    end

    for k=kk',
      mu(:,k)=mean(X(:,ind==k),2);
    end

    I=I+1;
  end

  tm=toc;

  if verbose>0,
    fprintf(logfile,'kmeans: number of iterations %d\n',I);
    fprintf(logfile,'kmeans: average iteration time %f\n',tm/I);
  end

end
