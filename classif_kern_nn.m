function E = classif_kern_nn(TR, TRlabels, kernel, K, TE, TElabels, varargin)
%
% CLASSIF_KERN_NN: Classify using Kernel Nearest Neighbor with the euclidean distance
%
% E = classif_kern_nn(TR, TRlabels, TE, TElabels, ...)
%
%   Input:
%     TR       - Training data matrix. Each column vector is a data point.
%     TRlabels - Training data class labels.
%     kernel   - One of the supported kernel types ('rbf'|...).
%     K        - Kernel parameters.
%     TE       - Testing data matrix. Each column vector is a data point.
%     TElabels - Testing data class labels.
%
%   Input (optional):
%     'perclass',(true|false)   - Compute error for each class (default=false)
%     'distance',('euclidean'|  - NN distance (default='euclidean')
%                 'cosine')
%
%   Output:
%     E        - Classification error
%
%
% Version: 1.02 -- Jul/2008
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

[D,N]=size(TR);
M=size(TE,2);

distance='euclidean';
perclass=false;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'perclass'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      n=n+2;
    end
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

rbf=false;
if strcmp(kernel,'rbf'),
  rbf=true;
  if max(size(K))~=1,
    fprintf(2,'classif_kern_nn: error: the parameters for RBF kernel must be a scalar\n');
    argerr=true;
  end
else
  fprintf(2,'classif_kern_nn: error: unsupported kernel (%s)\n',kernel);
  argerr=true;
end

if argerr,
  fprintf(2,'classif_kern_nn: error: incorrect input argument (%d-%d)\n',n+5,n+6);
elseif size(TRlabels,1)~=N,
  fprintf(2,'classif_kern_nn: error: size of TRlabels must be the same as the number of training points\n');
keyboard
elseif size(TElabels,1)~=M,
  fprintf(2,'classif_kern_nn: error: size of TElabels must be the same as the number of testing points\n');
elseif size(TE,1)~=D,
  fprintf(2,'classif_kern_nn: error: dimensionality of train and test data points must be the same\n');
elseif ~(strcmp(distance,'euclidean') || strcmp(distance,'cosine')),
  fprintf(2,'classif_kern_nn: error: invalid distance\n');
else

  euclidean=true;
  if strcmp(distance,'cosine'),
    euclidean=false;
  end

  dist=zeros(N,1);

  if euclidean,

    if ~perclass,

      E=0;

      if rbf,

        fprintf(2,'classif_kern_nn: kernel %s\n',kernel);
        ngamma2=-K(1)*K(1);
        for m=1:M,
          dist=sum(power(TE(:,m*ones(N,1))-TR,2));
          dist=1-exp(ngamma2.*dist);
          if min(dist(TRlabels==TElabels(m)))>min(dist(TRlabels~=TElabels(m))),
            E=E+1;
          end
        end

      end

      E=E/M;

    %else

    %  Clabels=unique(TRlabels)';
    %  C=max(size(Clabels));
    %  E=zeros(C,1);
    %  for m=1:M,
    %    for n=1:N,
    %      dist(n)=(TR(:,n)-TE(:,m))'*(TR(:,n)-TE(:,m));
    %    end
    %    if min(dist(TRlabels==TElabels(m)))>min(dist(TRlabels~=TElabels(m))),
    %      c=find(Clabels==TElabels(m));
    %      E(c)=E(c)+1;
    %    end
    %  end
    %  c=1;
    %  for label=Clabels,
    %    E(c)=E(c)/sum(TElabels==label);
    %    c=c+1;
    %  end

    end

  %else

  %  for n=1:N,
  %    TR(:,n)=TR(:,n)./sqrt(TR(:,n)'*TR(:,n));
  %  end

  %  if ~perclass,

  %    E=0;
  %    for m=1:M,
  %      TE(:,m)=TE(:,m)./sqrt(TE(:,m)'*TE(:,m));
  %      for n=1:N,
  %        dist(n)=1-TR(:,n)'*TE(:,m);
  %      end
  %      if min(dist(TRlabels==TElabels(m)))>min(dist(TRlabels~=TElabels(m))),
  %        E=E+1;
  %      end
  %    end
  %    E=E/M;

  %  else

  %    Clabels=unique(TRlabels)';
  %    C=max(size(Clabels));
  %    E=zeros(C,1);
  %    for m=1:M,
  %      TE(:,m)=TE(:,m)./sqrt(TE(:,m)'*TE(:,m));
  %      for n=1:N,
  %        dist(n)=1-TR(:,n)'*TE(:,m);
  %      end
  %      if min(dist(TRlabels==TElabels(m)))>min(dist(TRlabels~=TElabels(m))),
  %        c=find(Clabels==TElabels(m));
  %        E(c)=E(c)+1;
  %      end
  %    end
  %    c=1;
  %    for label=Clabels,
  %      E(c)=E(c)/sum(TElabels==label);
  %      c=c+1;
  %    end

  %  end

  end

end
