function S = rand_scores(num,mu,sigma)

S=zeros(sum(num),1);

n=1;
for m=1:max(size(mu)),
  S(n:n+num(m)-1)=sigma(m)*sqrt(-2*log(rand(num(m),1))).*cos(2*pi*rand(num(m),1))+mu(m);
  n=n+num(m);
end
[trash,ind]=sort(rand(sum(num),1));
S=S(ind);
