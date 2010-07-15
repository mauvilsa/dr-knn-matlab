function V = imgTang(X, imSize, krh, krv, types, basis)
%
% IMGTANG: Compute Image Tangent Vectors
%
% V = imgTang(X, imW, imH, krh, krv, types, basis)
%
%   Input:
%     X         - Input Data. Each column is an image.
%     imSize    - Image size. [W H].
%     krh       - Horizontal Kernel.
%     hrv       - Vertical Kernel.
%     types     - Tangent types. 'hvrspdtHVAD'.
%     basis     - (true|false) Compute Tangent Basis.
%
%   Output:
%     V         - Tangent Vectors.
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

[D,N]=size(X);
L=size(types,2);
V=zeros(D,N*L);

xmin=-ceil(imSize(1)/2);
ymin=-ceil(imSize(2)/2);

x=[xmin:imSize(1)+xmin-1]';
x=x(:,ones(imSize(2),1));
y=[ymin:imSize(2)+ymin-1];
y=y(ones(imSize(1),1),:);

for n=1:N,
  im=reshape(X(:,n),imSize(1),imSize(2));
  derh=conv2(im,krh,'same');
  derv=conv2(im,krv,'same');

  mh=sum(sum(derh.*derh));
  mv=sum(sum(derv.*derv));
  mag=0.5*(sqrt(mh)+sqrt(mv));

  v=zeros(D,L);
  for l=1:L,
    switch types(l)
      case 'h'
        v(:,l)=derh(:);
      case 'v'
        v(:,l)=derv(:);
      case 'r'
        v(:,l)=reshape(-y.*derh-x.*derv,D,1);
      case 's'
        v(:,l)=reshape(x.*derh-y.*derv,D,1);
      case 'p'
        v(:,l)=reshape(x.*derh+y.*derv,D,1);
      case 'd'
        v(:,l)=reshape(-y.*derh+x.*derv,D,1);
      case 't'
        v(:,l)=derh(:).*derh(:)+derv(:).*derv(:);
      case 'H'
        v(:,l)=reshape(x.*im,D,1);
      case 'V'
        v(:,l)=reshape(y.*im,D,1);
      case 'A'
        v(:,l)=reshape((x-y).*im,D,1);
      case 'D'
        v(:,l)=reshape((x+y).*im,D,1);
    end

    v(:,l)=v(:,l).*(mag/sqrt(sum(v(:,l).*v(:,l))));
    if types(l)>='A' && types(l)<='Z',
      v(:,l)=10*v(:,l);
    end

    if basis,
      v=orthonorm(v);
    end

    V(:,(n-1)*L+1:n*L)=v;
  end
end
