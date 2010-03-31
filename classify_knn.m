function [E, A, S] = classify_knn(P, Plabels, K, X, Xlabels, varargin)
%
% CLASSIFY_KNN: Classify using K Nearest Neighbor
%
% [E, A, S] = classify_nn(P, Plabels, K, X, Xlabels, ...)
%
% Input:
%   P        - Prototypes data matrix. Each column vector is a data point.
%   Plabels  - Prototypes labels.
%   K        - K neighbors.
%   X        - Testing data matrix. Each column vector is a data point.
%   Xlabels  - Testing data class labels.
%
% Input (optional):
%   'euclidean',(true|false)  - Euclidean distance (default=true)
%   'cosine',(true|false)     - Cosine distance (default=false)
%
% Output:
%   E        - Classification error
%   A        - Assigned Class
%   S        - Classification score
%
%
% $Revision$
% $Date$
%

%
% Copyright (C) 2008-2009 Mauricio Villegas (mvillegas AT iti.upv.es)
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

if strncmp(P,'-v',2),
  unix('echo "$Revision$* $Date$*" | sed "s/^:/classify_nn: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='classify_knn:';
minargs=5;

E=[];
A=[];
S=[];

[D,Np]=size(P);
K=min(Np,K);
Nx=size(X,2);
if size(Plabels,1)<size(Plabels,2),
  Plabels=Plabels';
end
if size(Xlabels,1)<size(Xlabels,2),
  Xlabels=Xlabels';
end

euclidean=true;
cosine=false;

logfile=2;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'euclidean') || ...
         strcmp(varargin{n},'cosine'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      if varargin{n+1}==true,
        if strcmp(varargin{n},'euclidean'),
          cosine=false;
        elseif strcmp(varargin{n},'cosine'),
          euclidean=false;
        end
      end
      n=n+2;
    end
  end
  if argerr || n>size(varargin,2),
    break;
  end
end

if argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif size(X,1)~=D,
  fprintf(logfile,'%s error: dimensionality prototypes and data must be the same\n',fn);
  return;
elseif size(Plabels,1)~=Np || size(Plabels,2)~=1 || ...
      (sum(size(Xlabels))~=0&&(size(Xlabels,1)~=Nx || size(Xlabels,2)~=1)),
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
end

onesNp=ones(Np,1);
onesNx=ones(Nx,1);
onesD=ones(D,1);

if euclidean,
  x2=sum((X.^2),1)';
  p2=sum((P.^2),1);
  d=X'*P;
  d=x2(:,onesNp)+p2(onesNx,:)-d-d;
elseif cosine,
  psd=sqrt(sum(P.*P,1));
  P=P./psd(onesD,:);
  xsd=sqrt(sum(X.*X,1));
  X=X./xsd(onesD,:);
  d=1-X'*P;
end

Clabels=unique(Plabels);
Cp=max(size(Clabels));
nPlabels=ones(size(Plabels));
nXlabels=ones(size(Xlabels));
for c=2:Cp,
  nPlabels(Plabels==Clabels(c))=c;
  nXlabels(Xlabels==Clabels(c))=c;
end

[d,idx]=sort(d,2);
lab=nPlabels(idx);

E=zeros(K,1);
cnt=zeros(Nx,Cp);

for k=1:K,
  labk=lab(:,k);
  for c=1:Cp,
    sel=labk==c;
    cnt(sel,c)=cnt(sel,c)+1;
  end
  [sel,labk]=max(cnt,[],2);
  E(k)=sum(nXlabels~=labk);
end

E=E/Nx;
