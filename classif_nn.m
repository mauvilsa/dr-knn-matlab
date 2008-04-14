function E = classif_nn(TR, TRid, TE, TEid)
%
% CLASSIF_NN: Classify using Nearest Neighbor with the euclidean distance
%
% E = classif_nn(TR, TRid, TE, TEid)
%
%   Input:
%     TR      - Training data matrix. Each column vector is a data point.
%     TRid    - Training data labels.
%     TE      - Testing data matrix. Each column vector is a data point.
%     TEid    - Testing data labels.
%
%   Output:
%     E       - Classification error
%

%
%   version 1.0 -- Apr/2008
%
%   Copyright (C) 2008 Mauricio Villegas (mvillegas AT iti.upv.es)
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

[D,N]=size(TR);
M=size(TE,2);

if size(TRid,1)~=N,
  fprintf(2,'ldpp: error: size of TRid must be the same as the number of training points\n');
elseif size(TEid,1)~=M,
  fprintf(2,'ldpp: error: size of TEid must be the same as the number of testing points\n');
elseif size(TE,1)~=D,
  fprintf(2,'ldpp: error: dimensionality of train and test data points must be the same\n');
else

  dist=zeros(N,1);

  E=0;
  for m=1:M,
    for n=1:N,
      dist(n)=(TR(:,n)-TE(:,m))'*(TR(:,n)-TE(:,m));
    end
    if min(dist(TRid==TEid(m)))>min(dist(TRid~=TEid(m))),
      E=E+1;
    end
  end

  E=E/M;

end
