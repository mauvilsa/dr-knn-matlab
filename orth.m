function X = orth(X)

for n=1:size(X,2),
  for m=1:n-1,
    X(:,n)=X(:,n)-(X(:,n)'*X(:,m))*X(:,m);
  end
  X(:,n)=(1/sqrt(X(:,n)'*X(:,n)))*X(:,n);
end
