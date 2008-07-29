function A = aur(POS, NEG)
%
% AUR: Compute the area under the ROC curve
%
% A = aur(POS, NEG)
%
%   Input:
%     POS                  - Positive scores
%     NEG                  - Negative scores
%
%   Output:
%     A                    - Area under ROC
%
%
% Version: 1.00 -- Jul/2008
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

D=size(POS,2);
NP=size(POS,1);
NN=size(NEG,1);

if D>1,

  SG=zeros(1,D);
  SE=zeros(1,D);

  for p=1:size(POS,1),
    pos=POS(p,:);
    SG=SG+sum(pos(ones(NN,1),:)>NEG);
    SE=SE+sum(pos(ones(NN,1),:)==NEG);
  end

  A=(SG+0.5.*SE)./(NP*NN);

else

  SG=0;
  SE=0;

  for p=1:size(POS,1),
    SG=SG+sum(POS(p,:)>NEG);
    SE=SE+sum(POS(p,:)==NEG);
  end

  A=(SG+0.5.*SE)./(NP*NN);

end
