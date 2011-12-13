function [ B, V, BB ] = cca( X, Xlabels, varargin )
%
% CCA: Canonical Correlation Analysis
%
% Usage:
%   [B, V] = cca(X, Xlabels, ...)
%
% Input:
%   X                   - Data matrix. Each column vector is a data point.
%   Xlabels             - Label matrix. Each column is a label vector.
%
% Input (optional):
%   'dopca',DIM         - Perform PCA before CCA (default=false)
%   'pcab',PCAB         - Supply the PCA basis
%
% Output:
%   B                   - Computed CCA basis
%   V                   - Computed CCA eigenvalues
%
% $Revision$
% $Date$
%

% Copyright (C) 2011 Mauricio Villegas (mvillegas AT iti.upv.es)
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

fn = 'cca:';
minargs = 2;

%%% File version %%%
if ischar(X)
  unix(['echo "$Revision$* $Date$*" | sed "s/^:/' fn ' revision/g; s/ : /[/g; s/ (.*)/]/g;"']);
  return;
end

%%% Default values %%%
B=[];
V=[];

dopca=false;

logfile=2;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0
  if ~ischar(varargin{n})
    argerr=true;
  elseif strcmp(varargin{n},'dopca') || ...
         strcmp(varargin{n},'pcab')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1})
      argerr=true;
    else
      n=n+2;
    end
  else
    argerr=true;
  end
  if argerr || n>size(varargin,2)
    break;
  end
end

[D,N]=size(X);
C=size(Xlabels,1);

%%% Error detection %%%
if argerr
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif size(Xlabels,2)~=N
  fprintf(logfile,'%s error: there must be the same number of labels as data points\n',fn);
  return;
end

if exist('pcab','var') && ~dopca
  dopca=size(pcab,2);
end
if dopca
  if ~exist('pcab','var')
    pcab=pca(X);
  end
  if dopca>size(pcab,2) || dopca<1
    fprintf(logfile,'%s error: inconsistent dimensions in PCA base\n',fn);
    return;
  end
  pcab=pcab(:,1:dopca);
  mu=mean(X,2);
  X=X-repmat(mu,1,N);
  X=pcab'*X;
  D=dopca;
end

XX=cor(X');
YY=cor(Xlabels');
XY=cor(X',Xlabels');

[B,V]=eig(inv(XX)*XY*inv(YY)*XY');
V=real(diag(V));
[srt,idx]=sort(-1*V);
idx=idx(1:min([D,C-1]));
V=V(idx);
B=B(:,idx);
if nargout>2
  BB=inv(YY)*XY'*B;
end

if dopca
  B=pcab*B;
end

if sum(~isfinite(V))>0
  fprintf(logfile,'%s warning: unexpected eigenvalues\n',fn);
end
