function [ B, V, BB ] = cca( X, Xlabels, varargin )
%
% CCA: Canonical Correlation Analysis
%
% Usage:
%   [ B, V ] = cca( X, Xlabels, ... )
%
% Input:
%   X                   - Data matrix. Each column vector is a data point.
%   Xlabels             - Label matrix. Each column is a label vector.
%
% Input (optional):
%   'dopca',DIM         - Perform PCA before CCA (default=false)
%   'cor',(true|false)  - Use correlation matrices (default=true)
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
B = [];
V = [];

dopca = false;
cor = false;

logfile = 2;

%%% Input arguments parsing %%%
n = 1;
argerr = false;
while size(varargin,2)>0
  if ~ischar(varargin{n})
    argerr = true;
  elseif strcmp(varargin{n},'dopca') || ...
         strcmp(varargin{n},'pcab')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  elseif strcmp(varargin{n},'cor')
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

[ D, N ] = size(X);
C = size(Xlabels,1);

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
  dopca = size(pcab,2);
end
if dopca
  if ~exist('pcab','var')
    pcab = pca(X);
  end
  if dopca>size(pcab,2) || dopca<1
    fprintf(logfile,'%s error: inconsistent dimensions in PCA base\n',fn);
    return;
  end
  pcab = pcab(:,1:dopca);
  mu = mean(X,2);
  X = X-repmat(mu,1,N);
  X = pcab'*X;
  D = dopca;
end

%if true

xmu = mean(X,2);
ymu = mean(Xlabels,2);

%XX = (1/(N-1))*(X*X'-N*xmu*xmu');
XX = (1/(N-1))*(X*X');
XX = XX-(N/(N-1))*(xmu*xmu');
XX = 0.5*(XX+XX');
%YY = (1/(N-1))*(Xlabels*Xlabels'-N*ymu*ymu');
YY = (1/(N-1))*(Xlabels*Xlabels');
YY = YY-(N/(N-1))*(ymu*ymu');
YY = 0.5*(YY+YY');
%XY = (1/(N-1))*(X*Xlabels'-N*xmu*ymu');
XY = (1/(N-1))*(X*Xlabels');
XY = XY-(N/(N-1))*(N*xmu*ymu');

if cor
  xsd = std(X,0,2);
  ysd = std(Xlabels,0,2);

  XX = XX./(xsd*xsd');
  YY = YY./(ysd*ysd');
  XY = XY./(xsd*ysd');
end

%else

%%XX = cor(X');
%%YY = cor(Xlabels');
%%XY = cor(X',Xlabels');
%XX = corrcoef(X');
%YY = corrcoef(Xlabels');
%XY = corrcoef(X',Xlabels');

%end

iYY = inv(YY);
%[ B, V ] = eig(inv(XX)*XY*iYY*XY');
[ B, V ] = eig(XY*iYY*XY',XX);
V = real(diag(V));
[ srt, idx ] = sort(-1*V);
idx = idx(1:min([D,C]));
V = V(idx);
B = B(:,idx);
B = B.*repmat(1./sqrt(sum(B.*B,1)),D,1);
if nargout>2
  BB = iYY*XY'*B;
end

if dopca
  B = pcab*B;
end

if ~isreal(B)
  fprintf(logfile,'%s warning: returning complex eigenvectors\n',fn);
end
if sum(~isfinite(V))>0
  fprintf(logfile,'%s warning: unexpected eigenvalues\n',fn);
end
