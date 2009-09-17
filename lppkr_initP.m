function [P0, PP] = lppkr_initP(X, XX, M, varargin)
%
% LPPKR_INITP: Initialize Prototypes for LPPKR
%
% P0 = lppkr_initP(X, XX, M)
%
%   Input:
%     X       - Independent data matrix. Each column vector is a data point.
%     XX      - Dependent data matrix.
%     M       - Number of prototypes.
%
%   Input (optional):
%     'extra',EXTRA              - Extrapolate EXTRA from extreme values (defaul=false)
%     'multi',MULT               - Multimodal prototypes, MULT-means (defaul=false)
%     'seed',SEED                - Random seed (default=system)
%
%   Output:
%     P0      - Initialized prototypes. Independent data.
%     P0      - Initialized prototypes. Dependent data.
%
%
% Version: 1.01 -- Sep/2009
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

seed=rand('state');

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'extra') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'multi'),
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

if argerr,
  fprintf(2,'lppkr_initP: error: incorrect input argument (%d-%d)\n',varargin{n},varargin{n+1});
elseif nargin-size(varargin,2)~=3,
  fprintf(2,'lppkr_initP: error: not enough input arguments\n');
elseif exist('multi','var') && mod(M,multi)~=0,
  fprintf(2,'lppkr_initP: error: the number of prototypes should be a multiple of MULT\n');
else

  DD=size(XX,1);

  mx=max(XX');
  mn=min(XX');

  if exist('extra','var'),
    omx=mx;
    omn=mn;
    mx=omx+extra.*(omx-omn);
    mn=omn-extra.*(omx-omn);
  end

  if ~exist('multi','var'),
    multi=1;
  else
    M=M/multi;
  end

  d=(mx-mn)/(M-1);

  rand('state',seed);

  if DD==1,

    P0=[];
    PP=[];

    for m=mn:d:mx,
      s=XX>=m-d/2 & XX<m+d/2;
      if sum(s)>multi,
        P0=[P0,kmeans(X(:,s),multi,'seed',seed)];
        seed=rand('state');
      else
        [mdist,idx]=sort(abs(XX-m));
        P0=[P0,X(:,idx(1:multi))];
      end
      PP=[PP,m*ones(1,multi)];
    end

  else
    fprintf(2,'lppkr_initP: error: dimensionality of dependent data higher than one, not implemented yet\n');
%    dd=1,
%    for m=mn(dd):(mx(dd)-mn(dd))/(M-1):mx(dd),
  end

end
