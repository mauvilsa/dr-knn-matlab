function [E, A, S] = classify_nn(P, Plabels, X, Xlabels, varargin)
%
% CLASSIFY_NN: Classify using Nearest Neighbor
%
% [E, S] = classif_nn(P, Plabels, X, Xlabels, ...)
%
% Input:
%   P        - Prototypes data matrix. Each column vector is a data point.
%   Plabels  - Prototypes labels.
%   X        - Testing data matrix. Each column vector is a data point.
%   Xlabels  - Testing data class labels.
%
% Input (optional):
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

[D,Np]=size(P);
Nx=size(X,2);

perclass=false;
euclidean=true;
cosine=false;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
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
elseif max(size(Xlabels))~=Nx || min(size(Xlabels))~=1 || ...
       max(size(Plabels))~=Np || min(size(Plabels))~=1,
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return
end

ind1=[1:Nx]';
ind1=ind1(:,ones(Np,1));
ind1=ind1(:);

ind2=1:Np;
ind2=ind2(ones(Nx,1),:);
ind2=ind2(:);

sel=Plabels(:,ones(Nx,1))'==Xlabels(:,ones(Np,1));

if euclidean,
  ds=reshape(sum((X(:,ind1)-P(:,ind2)).^2,1),Nx,Np);
elseif cosine,
  psd=sqrt(sum(P.*P,1));
  P=P./psd(ones(D,1),:);
  xsd=sqrt(sum(X.*X,1));
  X=X./xsd(ones(D,1),:);
  ds=reshape(1-sum(X(:,ind1).*rP(:,ind2),1),Nx,Np);
end

dd=ds;
ds(~sel)=inf;
dd(sel)=inf;
[ds,is]=min(ds,[],2);
[dd,id]=min(dd,[],2);

Cp=max(size(unique(Plabels)));
Fp=(Cp-1)/Cp;

if ~perclass,
  %E=(sum(dd<ds))/Nx;
  E=(sum(dd<ds)+Fp*sum(dd==ds))/Nx;
else
  Clabels=unique(Xlabels)';
  C=max(size(Clabels));
  E=zeros(C,1);
  c=1;
  for label=Clabels,
    sel=Xlabels==label;
    %E(c)=sum(dd(sel)<ds(sel))/sum(sel);
    E(c)=(sum(dd(sel)<ds(sel))+Fp*sum(dd(sel)==ds(sel)))/sum(sel);
    c=c+1;
  end
end

if nargout>1,
  A=Plabels(is);
  A(dd<=ds)=Plabels(id(dd<=ds));
end
if nargout>2,
  S=dd./(dd+ds);
end
