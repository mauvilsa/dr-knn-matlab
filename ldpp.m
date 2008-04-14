function [bestB, bestP] = ldpp(X, Xid, B0, P0, Pid, varargin)
%
% LDPP: Learning Discriminative Projections and Prototypes
%
% [B, P] = ldpp(X, Xid, B0, P0, Pid, ...)
%
%   Input:
%     X       - Data matrix. Each column vector is a data point.
%     Xid     - Data labels.
%     B0      - Initial projection base.
%     P0      - Initial prototypes.
%     Pid     - Prototype labels.
%
%   Input (optional):
%     'beta',BETA                - Sigmoid slope (defaul=10)
%     'gamma',GAMMA              - Projection base learning rate (default=1)
%     'eta',ETA                  - Prototypes learning rate (default=1)
%     'epsilon',EPSILON          - Convergence criterium (default=1e-7)
%     'minI',MINI                - Minimum number of iterations (default=100)
%     'maxI',MAXI                - Maximum number of iterations (default=1000)
%     'orthonormal',(true|false) - Orthonormal projection base (default=false)
%     'normalize',(true|false)   - Normalize training data (default=false)
%     'squared',(true|false)     - Squared euclidean distance (default=false)
%     'logfile',FID              - Output log file (default=stderr)
%
%   Output:
%     B   - Final learned projection base
%     P   - Final learned prototypes
%
%
% Reference:
%
%   M. Villegas and R. Paredes. "Simultaneous Learning of a Discriminative
%   Projection and Prototypes for Nearest-Neighbor Classification."
%   CVPR'2008.
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

beta=10;
gamma=1;
eta=1;

epsilon=1e-7;
minI=100;
maxI=1000;

orthonormal=false;
squared=false;
normalize=false;

logfile=2;

% temporal %
test=false;
%test=true;
gamma=1;
eta=1000;
minI=10;
maxI=10;
% temporal %

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'beta') || strcmp(varargin{n},'gamma') || ...
         strcmp(varargin{n},'eta')  || strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'logfile'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || varargin{n+1}<0,
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'normalize') || strcmp(varargin{n},'squared') || ...
         strcmp(varargin{n},'orthonormal'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
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

N=size(X,2);
D=size(X,1);
R=size(B0,2);
M=size(P0,2);

bestB=B0;
bestP=P0;

if argerr,
  fprintf(logfile,'ldpp: error: incorrect input argument (%d-%d)\n',n+5,n+6);
elseif size(Xid,1)~=N,
  fprintf(logfile,'ldpp: error: size of Xid must be the same as the number of data points\n');
elseif size(Pid,1)~=M,
  fprintf(logfile,'ldpp: error: size of Pid must be the same as the number of prototypes\n');
elseif size(B0,1)~=D,
  fprintf(logfile,'ldpp: error: dimensionality of base and data must be the same\n');
elseif size(P0,1)~=D,
  fprintf(logfile,'ldpp: error: dimensionality of prototypes and data must be the same\n');
else

  if normalize,
    xmu=mean(X,2);
    xsd=std(X,1,2);
    X=(X-repmat(xmu,1,N))./repmat(xsd,1,N);
    P0=(P0-repmat(xmu,1,M))./repmat(xsd,1,M);
  end

  if orthonormal,
    for n=1:R,
      for m=1:n-1,
        B0(:,n)=B0(:,n)-(B0(:,n)'*B0(:,m))*B0(:,m);
      end
     B0(:,n)=(1/sqrt(B0(:,n)'*B0(:,n)))*B0(:,n);
    end
  end

  dist=zeros(M,1);
  ds=zeros(N,1);
  dd=zeros(N,1);
  is=zeros(N,1);
  id=zeros(N,1);

  B=B0;
  P=P0;
  bestI=0;
  bestJ=1;
  bestE=1;

  J=1;
  I=0;

  gamma=2*gamma/N;
  eta=2*eta/N;
  if ~squared,
    gamma=gamma/N;
    eta=eta/N;
  end

  fprintf(logfile,'ldpp: output: iteration | J | delta(J) | training set error\n');

  while 1,

    Y=B'*X;
    Q=B'*P;

    for n=1:N,
      for m=1:M,
        dist(m)=(Y(:,n)-Q(:,m))'*(Y(:,n)-Q(:,m));
      end
      dsn=min(dist(Pid==Xid(n)));
      ddn=min(dist(Pid~=Xid(n)));
      isn=find(dist==dsn);
      idn=find(dist==ddn);
      ds(n)=dsn;
      dd(n)=ddn;
      is(n)=isn(1);
      id(n)=idn(1);
    end
    ds(ds==0)=realmin;
    dd(dd==0)=realmin;
    if ~squared,
      ds=sqrt(ds);
      dd=sqrt(dd);
    end
    ratio=ds./dd;
    expon=exp(beta*(1-ratio));
    J0=sum(1./(1+expon))/N;
    E=sum(dd<ds)/N;

    fprintf(logfile,'%d\t%f\t%f\t%f\n',I,J0,J0-J,E);

    if J0<=bestJ,
      bestB=B;
      bestP=P;
      bestI=I;
      bestJ=J0;
      bestE=E;
    end

    if I>=maxI,
      fprintf(logfile,'ldpp: reached maximum number of iterations\n');
      break;
    end

    if I>=minI,
      if abs(J0-J)<epsilon,
        fprintf(logfile,'ldpp: index has stabilized\n');
        break;
      end
    end

    J=J0;
    I=I+1;

    ratio=beta.*ratio.*expon./((1+expon).*(1+expon));
    ds=ratio./ds;
    dd=ratio./dd;
    Xs=X-P(:,is);
    Xd=X-P(:,id);
    Ys=Y-Q(:,is);
    Yd=Y-Q(:,id);

    for n=1:N,
      Ys(:,n)=ds(n)*Ys(:,n);
      Yd(:,n)=dd(n)*Yd(:,n);
    end

    for m=1:M,
      Q0(:,m)=sum(Yd(:,id==m),2)-sum(Ys(:,is==m),2);
    end
    P0=B*Q0;
    B0=Xs*Ys'-Xd*Yd';


    if test,

    fprintf(logfile,'ldpp: using learning rate balancing\n');

    %for g=gamma*[1,2,4,8,16,32],
    for g=gamma,

    B1=B-g*B0;
    P1=P-g*P0;
    Y1=B1'*X;
    Q1=B'*P1;
    for n=1:N,
      for m=1:M,
        distU(m)=(Y1(:,n)-Q(:,m))'*(Y1(:,n)-Q(:,m));
        distL(m)=(Y(:,n)-Q1(:,m))'*(Y(:,n)-Q1(:,m));
      end
      dsn=min(distU(Pid==Xid(n)));
      ddn=min(distU(Pid~=Xid(n)));
      isn=find(distU==dsn);
      idn=find(distU==ddn);
      dsU(n)=dsn;
      ddU(n)=ddn;
      isU(n)=isn(1);
      idU(n)=idn(1);
      dsn=min(distL(Pid==Xid(n)));
      ddn=min(distL(Pid~=Xid(n)));
      isn=find(distL==dsn);
      idn=find(distL==ddn);
      dsL(n)=dsn;
      ddL(n)=ddn;
      isL(n)=isn(1);
      idL(n)=idn(1);
    end
    ratioU=dsU./ddU;
    ratioL=dsL./ddL;
    exponU=exp(beta*(1-ratioU));
    exponL=exp(beta*(1-ratioL));
    JU=sum(1./(1+exponU))/N;
    JL=sum(1./(1+exponL))/N;

    %fprintf(logfile,'t ------------- %f\n',(J0-JU)/(J0-JL));

    %B1=B-g*B0;
    %P1=P-g*((J0-JU)/(J0-JL))*P0;
    %Y1=B1'*X;
    %Q1=B'*P1;
    %for n=1:N,
    %  for m=1:M,
    %    distU(m)=(Y1(:,n)-Q(:,m))'*(Y1(:,n)-Q(:,m));
    %    distL(m)=(Y(:,n)-Q1(:,m))'*(Y(:,n)-Q1(:,m));
    %  end
    %  dsn=min(distU(Pid==Xid(n)));
    %  ddn=min(distU(Pid~=Xid(n)));
    %  isn=find(distU==dsn);
    %  idn=find(distU==ddn);
    %  dsU(n)=dsn;
    %  ddU(n)=ddn;
    %  isU(n)=isn(1);
    %  idU(n)=idn(1);
    %  dsn=min(distL(Pid==Xid(n)));
    %  ddn=min(distL(Pid~=Xid(n)));
    %  isn=find(distL==dsn);
    %  idn=find(distL==ddn);
    %  dsL(n)=dsn;
    %  ddL(n)=ddn;
    %  isL(n)=isn(1);
    %  idL(n)=idn(1);
    %end
    %ratioU=dsU./ddU;
    %ratioL=dsL./ddL;
    %exponU=exp(beta*(1-ratioU));
    %exponL=exp(beta*(1-ratioL));
    %JU=sum(1./(1+exponU))/N;
    %JL=sum(1./(1+exponL))/N;

    %fprintf(logfile,'m ------------- %f\n',(J0-JU)/(J0-JL));

    eta=gamma;
    eta=gamma*(J0-JU)/(J0-JL);

    %ysU=mean(Y-Q(:,is),2);
    %ydU=mean(Y-Q(:,id),2);
    %ysL=mean(Y-(1-gamma)*Q(:,is),2);
    %ydL=mean(Y-(1-gamma)*Q(:,id),2);
    %exponU=exp(beta*(1-(ysU'*ysU)/(ydU'*ydU)));
    %exponL=exp(beta*(1-(ysL'*ysL)/(ydL'*ydL)));
    %JU=sum(1./(1+exponU))/N;
    %JL=sum(1./(1+exponL))/N;

    %fprintf(logfile,'s ------------- %f\n',(J0-JU)/(J0-JL));

    end % for gamma

    end % if test


    B=B-gamma*B0;
    P=P-eta*P0;

    if orthonormal,
      for n=1:R,
        for m=1:n-1,
          B(:,n)=B(:,n)-(B(:,n)'*B(:,m))*B(:,m);
        end
        B(:,n)=(1/sqrt(B(:,n)'*B(:,n)))*B(:,n);
      end
    end

  end

  if normalize,
    bestP=bestP.*repmat(xsd,1,M)+repmat(xmu,1,M);
    bestB=bestB./repmat(xsd,1,R);
  end

  fprintf(logfile,'ldpp: best iteration %d, J=%f, error=%f\n',bestI,bestJ,bestE);

end
