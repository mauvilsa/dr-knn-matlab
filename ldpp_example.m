% read the training data
load('gender.data');
X=gender(:,1:size(gender,2)-1)';
Xlabels=gender(:,size(gender,2));

% create an initial projection base (using PCA)
B0 = pca(X);
B0 = B0(:,1:16);

% create initial prototypes (class means)
P0 = [];
Plabels = [];
for c = unique(Xlabels)',
  P0 = [ P0, repmat(mean(X(:,Xlabels==c),2),1,4) ];
  Plabels = [ Plabels; c*ones(4,1) ];
end;

% learn with LDPP
[B,P] = ldpp(X, Xlabels, B0, P0, Plabels);
