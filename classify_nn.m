function [E, A, S, cdist] = classify_nn(P, Plabels, X, Xlabels, varargin)
%
% CLASSIFY_NN: Classify using Nearest Neighbor
%
% [E, A, S] = classify_nn(P, Plabels, X, Xlabels, ...)
%
% Input:
%   P        - Prototypes data matrix. Each column vector is a data point.
%   Plabels  - Prototypes labels.
%   X        - Testing data matrix. Each column vector is a data point.
%   Xlabels  - Testing data class labels.
%
% Input (optional):
%   'prior',PRIOR             - A priori probabilities (default=Ncx/Nx)
%   'perclass',(true|false)   - Compute error for each class (default=false)
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

fn='classify_nn:';
minargs=4;

E=[];
A=[];
S=[];

[D,Np]=size(P);
Nx=size(X,2);
if size(Plabels,1)<size(Plabels,2),
  Plabels=Plabels';
end
if size(Xlabels,1)<size(Xlabels,2),
  Xlabels=Xlabels';
end

perclass=false;
euclidean=true;
cosine=false;

logfile=2;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'prior'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(sum(varargin{n+1}<0))~=0,
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'perclass') || ...
         strcmp(varargin{n},'euclidean') || ...
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

ind1=[1:Nx]';
ind1=ind1(:,ones(Np,1));
ind1=ind1(:);

ind2=1:Np;
ind2=ind2(ones(Nx,1),:);
ind2=ind2(:);

if euclidean,
  d=reshape(sum((X(:,ind1)-P(:,ind2)).^2,1),Nx,Np);
elseif cosine,
  psd=sqrt(sum(P.*P,1));
  P=P./psd(ones(D,1),:);
  xsd=sqrt(sum(X.*X,1));
  X=X./xsd(ones(D,1),:);
  d=reshape(1-sum(X(:,ind1).*rP(:,ind2),1),Nx,Np);
end

Clabels=unique(Plabels)';
Cp=max(size(Clabels));

dist=zeros(Nx,Cp);
c=1;
for label=Clabels,
  csel=Plabels==label;
  dist(:,c)=min(d(:,csel),[],2);
  c=c+1;
end

[d1,A]=min(dist,[],2);

if sum(size(Xlabels))~=0,
  if perclass || exist('prior','var'),
    E=zeros(C,1);
    c=1;
    for label=Clabels,
      sel=Xlabels==label;
      E(c)=sum(Clabels(A(sel))'~=label)/sum(sel);
      c=c+1;
    end
    if exist('prior','var'),
      E=E'*prior;
    end
  else
    E=sum(Clabels(A)'~=Xlabels)/Nx;
  end
end

if nargout>3,
  cdist=dist;
end
if nargout>2,
  dist(Nx*(A-1)+[1:Nx]')=inf;
  d2=min(dist,[],2);
  S=d2./(d1+d2);
end
if nargout>1,
  A=Clabels(A)';
end
