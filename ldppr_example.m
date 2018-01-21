% read the training data
load('gender.data');
X=gender(:,1:end-1)';
XX=gender(:,end)';

% create an initial projection base (using PCA)
B0 = pca(X);
B0 = B0(:,1:16);

% create initial prototypes
[P0, PP0] = ldppr('initP', X, XX, 2);

% learn with LDPPR
[B, P, PP] = ldppr(X, XX, B0, P0, PP0);

% compute RMSE
K=0; % use all of the prototypes for regression
xx=regress_knn(B'*P, PP, K, B'*X);
rmse=sqrt(mean((XX-xx).^2))
