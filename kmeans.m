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
% $Revision: 86 $
% $Date: 2010-03-03 17:12:26 +0000 (Wed, 03 Mar 2010) $
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

fn='kmeans:';
minargs=2;

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
verbose=false;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'fastmode'),
    eval([varargin{n},'=varargin{n+1};']);
    n=n+2;
  elseif strcmp(varargin{n},'mu0') || ...
         strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'seed'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}),
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

if exist('fastmode','var'),
  ind1=fastmode.ind1;
  ind2=fastmode.ind2;
  logfile=fastmode.logfile;
  verbose=true;
  fastmode=true;
else
  fastmode=false;
end

if fastmode,
elseif argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif K>N,
  fprintf(logfile,'%s error: K must be lower than the number of data points\n',fn);
  return;
elseif D0~=D || K0~=K,
  fprintf(logfile,'%s error: incompatible size of initial means\n',fn);
  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

pind=zeros(N,1);
if ~exist('mu0','var'),
  if exist('seed','var'),
    rand('state',seed);
  end
  [k,pind]=sort(rand(N,1));
  mu=X(:,pind(1:K));
else
  mu=mu0;
end

if ~fastmode,
  onesN=ones(N,1);
  onesK=ones(K,1);

  ind1=[1:N]';
  ind1=ind1(:,onesK);
  ind1=ind1(:);

  ind2=1:K;
  ind2=ind2(onesN,:);
  ind2=ind2(:);
end

I=0;

tic;

while true,

  if false,

  ind=onesN;
  dist=sum((X-mu(:,onesN)).^2);
  for k=2:K,
    distk=sum((X-mu(:,k*onesN)).^2);
    ind(distk<dist)=k;
    dist(distk<dist)=distk(distk<dist);
  end

  else

  dist=reshape(sum((X(:,ind1)-mu(:,ind2)).^2,1),N,K);
  [dist,ind]=min(dist,[],2);

  end

  fprintf(logfile,'%d %f\n',I,sum(dist)/N);

  if I==maxI,
    fprintf(logfile,'%s reached maximum number of iterations\n',fn);
    break;
  end

  if sum(ind~=pind)==0,
    fprintf(logfile,'%s algorithm converged\n',fn);
    break;
  end

  kk=unique(ind);
  if size(kk,1)~=K,
    fprintf(logfile,'%s warning: %d empty means, setting them to random samples (I=%d)\n',fn,K-size(kk,1),I);
    for k=1:K,
      if sum(kk==k)==0,
        mu(:,k)=X(:,round((N-1)*rand)+1);
      end
    end
  end

  for k=kk',
    mu(:,k)=mean(X(:,ind==k),2);
  end

  pind=ind;
  I=I+1;
end

fprintf(logfile,'%s number of iterations: %d\n',fn,I);
fprintf(logfile,'%s average iteration time (s): %f\n',fn,toc/I);

if ~verbose,
  fclose(logfile);
end
