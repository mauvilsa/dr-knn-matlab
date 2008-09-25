function P = postprob_nn(TR, TRlabels, TE, C, varargin)
%
% POSTPROB_NN: Estimate the Posterior Probability by Nearest Neighbor for Class C
%
% P = postprob_nn(TR, TRlabels, TE, C, ...)
%
%   Input:
%     TR       - Training data matrix. Each column vector is a data point.
%     TRlabels - Training data class labels.
%     TE       - Testing data matrix. Each column vector is a data point.
%     C        - Class being tested.
%
%   Input (optional):
%     'distance',('euclidean'|  - NN distance (default='euclidean')
%                 'cosine')
%
%   Output:
%     P        - Posterior Probability for Class C
%
%
% Version: 1.00 -- Sep/2008
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
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

[D,N]=size(TR);
M=size(TE,2);

distance='euclidean';

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'distance'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~ischar(varargin{n+1}),
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
  fprintf(2,'postprob_nn: error: incorrect input argument (%d-%d)\n',n+5,n+6);
elseif size(TRlabels,1)~=N,
  fprintf(2,'postprob_nn: error: size of TRlabels must be the same as the number of training points\n');
elseif size(TE,1)~=D,
  fprintf(2,'postprob_nn: error: dimensionality of train and test data points must be the same\n');
elseif ~(strcmp(distance,'euclidean') || strcmp(distance,'cosine')),
  fprintf(2,'postprob_nn: error: invalid distance\n');
else

  euclidean=true;
  if strcmp(distance,'cosine'),
    euclidean=false;
  end

  dist=zeros(N,1);
  P=zeros(M,1);

  if euclidean,

    for m=1:M,
      dist=sum(power(TE(:,m*ones(N,1))-TR,2));
      ds=min(dist(TRlabels==C));
      dd=min(dist(TRlabels~=C));
      P(m)=dd(1)/(dd(1)+ds(1));
    end
    
  else

    for n=1:N,
      TR(:,n)=TR(:,n)./sqrt(TR(:,n)'*TR(:,n));
    end
    for m=1:M,
      TE(:,m)=TE(:,m)./sqrt(TE(:,m)'*TE(:,m));
      dist=1-sum(TE(:,m*ones(N,1)).*TR);
      ds=min(dist(TRlabels==C));
      dd=min(dist(TRlabels~=C));
      P(m)=dd(1)/(dd(1)+ds(1));
    end

  end

end
