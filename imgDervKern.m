function [krh, krv] = imgDervKern(krW, krH, bandwidth)
%
% IMGDERVKERN: Compute Image Derivative Kernel
%
% [krh, krv] = imgDervKern(krW, krH, bandwidth)
%
%   Input:
%     krW       - Kernel Width.
%     krH       - Kernel Height.
%     bandwidth - Bandwidth.
%
%   Output:
%     krh       - Horizontal Kernel.
%     hrv       - Vertical Kernel.
%
%
% $Revision$
% $Date$
%

%
% Copyright (C) 2008-2010 Mauricio Villegas (mvillegas AT iti.upv.es)
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

xmin=-floor(krW/2);
ymin=-floor(krH/2);

x=xmin:xmin+krW-1;
x=x(ones(krH,1),:);
y=(ymin:ymin+krH-1)';
y=y(:,ones(krW,1));

bandwidth=bandwidth*bandwidth;

e=exp(-(x.*x+y.*y)./(2*bandwidth))/bandwidth;

krh=-x.*e;
krv=y.*e;

krh=(krh./sqrt(sum(sum(krh.*krh))))';
krv=(krv./sqrt(sum(sum(krv.*krv))))';
