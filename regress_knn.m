function XX = regress_knn(P, PP, K, X, varargin)
%
% REGRESS_KNN: K-NN regression
%
% XX = regress_knn(P, PP, K, X, ...)
%
%   Input:
%     P        - Regression model independent variables.
%     PP       - Regression model dependent variables.
%     K        - Number of neighbors for the K-NN.
%     X        - Data independent variables.
%
%   Input (optional):
%     'distance',('euclidean'|  - NN distance (default='euclidean')
%                 'cosine')
%
%   Output:
%     XX       - Regression
%
%
% Version: 1.00 -- Sep/2009
%

%
% Copyright (C) 2008 Mauricio Villegas <mauvilsa@upv.es>
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

[D,M]=size(P);
DD=size(PP,1);
N=size(X,2);

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
  fprintf(2,'regress_knn: error: incorrect input argument (%d-%d)\n',varargin{n},varargin{n+1});
elseif nargin-size(varargin,2)~=4,
  fprintf(2,'regress_knn: error: not enough input arguments\n');
elseif size(PP,2)~=M,
  fprintf(2,'regress_knn: error: the number of vectors in the dependent and independent data must be the same\n');
elseif size(X,1)~=D,
  fprintf(2,'regress_knn: error: dimensionality of the model and evaluation data must be the same\n');
elseif ~(strcmp(distance,'euclidean') || strcmp(distance,'cosine')),
  fprintf(2,'regress_knn: error: invalid distance\n');
else

  euclidean=true;
  if strcmp(distance,'cosine'),
    euclidean=false;
  end

  if euclidean,

    ind=1:M;
    ind=ind(ones(N,1),:);
    ind=ind(:);

    ind2=1:DD;
    ind2=ind2(ones(N,1),:);
    ind2=ind2(:);

    mindist=100*sqrt(1/realmax); %%% g(d)=1/d

    %dist=reshape(exp(-sum(power(repmat(X,1,M)-P(:,ind),2),1)),N,M); %%% g(d)=exp(-d)
    dist=sum(power(repmat(X,1,M)-P(:,ind),2),1); dist(dist<mindist)=mindist; dist=reshape(1./dist,N,M); %%% g(d)=1/d

    if K==0,
      S=sum(dist,2);
      XX=(reshape(sum(repmat(dist,DD,1).*PP(ind2,:),2),N,DD)./S(:,ones(DD,1)))';
    else
      [dist,idx]=sort(dist',1);
      dist=dist(1:K,:);
      S=sum(dist,1)';
      tPP=PP(ind2,:)';
      ind3=M*[0:DD*N-1];
      ind3=ind3(ones(M,1),:);
      tPP=tPP(repmat(idx,1,DD)+ind3);
      tPP=tPP(1:K,:);
      XX=(reshape(sum(repmat(dist',DD,1).*tPP',2),N,DD)./S(:,ones(DD,1)))';
    end

  end

end
