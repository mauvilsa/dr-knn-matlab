function [ bestB, bestP, einfo ] = shmahd( X, Xlabels, B0, P0, varargin )
%
% SHMAHD: Semantic Hashing by Minimizing Approximate Hamming Distance
%
% Usage (learn):
%   [ B, P ] = shmahd( X, ( Xlabels | [] ), B0, P0, ...)
%
% Usage (compute hashes):
%   hX = shmahd( X, 'hash', B [, P] )
%
% Input:
%   X       - Data matrix. Each column vector is a data point.
%   Xlabels - Data class labels. Each column vector is a label point.
%   B0      - Initial projection base.
%   P0      - Initial median vector.
%
% Output:
%   B       - Final learned projection base.
%   P       - Final learned median vector.
%
% Learning options:
%   'slope',SLOPE              - Tanh slope (defaul=1)
%   'rateB',RATEB              - Projection base learning rate (default=0.1)
%   'rateP',RATEP              - Median vector learning rate (default=0.1)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'dist',('euclidean'|       - Distance for Affinity matrix (default=cosine)
%           'cosine')
%
% Stochastic options:
%   'stochastic',(true|false)  - Stochastic gradient descend (default=false)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=1000)
%   'stocheck',SIT             - Check stats every SIT iterations (default=1)
%   'stocheckfull',(true|f...  - Stats for whole data set (default=false)
%   'stochfinalexact',(tru...  - Final stats for whole data set (default=false)
%
% Verbosity options:
%   'verbose',(true|false)     - Verbose (default=true)
%   'stats',STAT               - Statistics every STAT (default=10)
%   'logfile',FID              - Output log file (default=stderr)
%
% Other options:
%   'seed',SEED                - Random seed (default=system)
%   'simbw',SIMBW              - Similarity matrix bandwidth (default=0.2)
%
%
% $Revision$
% $Date$
%

% Copyright (C) 2011 Mauricio Villegas (mvillegas AT iti.upv.es)
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

fn = 'shmahd:';
minargs = 4;

%%% File version %%%
if ischar(X) && ( strncmp(X,'-v',2) || strncmp(X,'--v',3) )
  unix(['echo "$Revision$* $Date$*" | sed "s/^:/' fn ' revision/g; s/ : /[/g; s/ (.*)/]/g;"']);
  return;
end

%%% Compute hashes of data %%%
if strncmp(Xlabels,'hash',4)
  if size(argn,1)<4 || sum(size(P0))==0
    bestB = ( B0'*X ) >= 0;
  else
    bestB = ( B0'*X-P0(:,ones(size(X,2),1)) ) >= 0;
  end
  return;
end

%%% Default values %%%
bestB = [];
bestP = [];

slope = 1;
rateB = 0.1;
rateP = 0.1;

epsilon = 1e-7;
minI = 100;
maxI = 1000;
stats = 10;

stochastic = false;
stochsamples = 1000;
stocheck = 1;
stocheckfull = false;
stochfinalexact = false;

orthonormal = true;
testJ = false;
dtype.euclidean = false;
dtype.cosine = true;
simbw = 0.2;

logfile = 2;
verbose = true;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0
  if ~ischar(varargin{n}) || size(varargin,2)<n+1
    argerr = true;
  elseif strcmp(varargin{n},'slope') || ...
         strcmp(varargin{n},'rateB') || ...
         strcmp(varargin{n},'rateP')  || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'stochsamples') || ...
         strcmp(varargin{n},'stocheck') || ...
         strcmp(varargin{n},'logfile')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(sum(varargin{n+1}<0))~=0
      argerr = true;
    else
      n = n+2;
    end
  elseif strcmp(varargin{n},'orthonormal') || ...
         strcmp(varargin{n},'stochastic') || ...
         strcmp(varargin{n},'stocheckfull') || ...
         strcmp(varargin{n},'stochfinalexact') || ...
         strcmp(varargin{n},'testJ') || ...
         strcmp(varargin{n},'verbose')
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  elseif strcmp(varargin{n},'dist') && ( ...
         strcmp(varargin{n+1},'euclidean') || ...
         strcmp(varargin{n+1},'cosine') )
    dtype.euclidean = false;
    dtype.cosine = false;
    eval(['dtype.',varargin{n+1},'=true;']);
    n = n+2;
  else
    argerr = true;
  end
  if argerr || n>size(varargin,2)
    break;
  end
end

[ D, Nx ] = size(X);
Dl = size(Xlabels,1);

%%% Error detection %%%
if argerr
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif max(size(B0))>1 && size(B0,1)~=D
  fprintf(logfile,'%s error: dimensionality of base and data must be the same\n',fn);
  return;
elseif sum(size(P0))>0 && ( size(P0,1)~=Dr || size(P0,2)~=1 )
  fprintf(logfile,'%s error: P0 must be a column vector with the same dimensions as number of bits\n',fn);
  return;
elseif ( sum(size(Xlabels))~=0 && size(Xlabels,2)~=Nx )
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
elseif ( sum(size(Xlabels))~=0 && sum(sum(Xlabels<0))>0 )
  fprintf(logfile,'%s error: expected labels to be non-negative\n',fn);
  return;
end

if ~verbose
  logfile = fopen('/dev/null');
end

%%% Preprocessing %%%
  tic;

  %%% Automatic initial parameters %%%
  if max(size(B0))==1
    Bi = pca(X);
    B0 = Bi(:,1:min(B0,D));
  end
  Dr = size(B0,2);
  if sum(size(P0))==0
    P0 = zeros(Dr,1);
  end

  %%% Initial parameter constraints %%%
  B0 = double(B0);
  P0 = double(P0);

  if orthonormal
    B0 = orthonorm(B0);
  end

  %%% Constant data structures %%%
  work.simbw = simbw;
  work.slope = slope;
  work.overNxNxDr = 1/(Nx*Nx*Dr);
  work.dtype = dtype;
  work.Nx = Nx;
  work.onesNx = ones(Nx,1);
  work.onesD = ones(D,1);
  if Dl>0
    work.onesDl = ones(Dl,1);
  end

  if stochsamples>Nx
    stochastic = false;
  end

  if stochastic
    if exist('seed','var')
      rand('state',seed);
    end

    rndidx = randperm(Nx);
    nn = 1;

    swork = work;
    swork.overNxNxDr = 1/(stochsamples*stochsamples*Dr);
    swork.Nx = stochsamples;
    swork.onesNx = ones(stochsamples,1);

    if Dl==0
      sXlabels = [];
    end

    prevJ = -1;
    prevE = -1;
  end

  if ~stochastic || stocheckfull || stochfinalexact
    L = shmahd_laplacian( X, Xlabels, work );
  end

  tm = toc;
  einfo.time = tm;
  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,tm);
%%%

Bi = B0;
Pi = P0;
bestB = B0;
bestP = P0;
bestIJE = [0 inf inf -1];

J00 = 1;
J0 = 1;
I = 0;

fprintf(logfile,'%s Nx=%d D=%d Dr=%d\n',fn,Nx,D,Dr);
fprintf(logfile,'%s output: iteration | J | delta(J) | E\n',fn);
tic;

while true

  if ~stochastic
    %%% Compute statistics %%%
    [ E, J, fX, gP ] = shmahd_objfunc( Bi'*X-Pi(:,work.onesNx), L, work );

  else
    %%% Compute statistics %%%
    if mod(I,stocheck)==0 && stocheckfull
      [ E, J ] = shmahd_objfunc( Bi'*X-Pi(:,work.onesNx), L, work );
    end

    %%% Select random samples %%%
    if nn+stochsamples>Nx
      randn = rndidx(nn:Nx);
      rndidx = randperm(Nx);
      nn = 2+size(randn,2);
      randn = [ randn rndidx(1:nn-1) ];
    else
      randn = rndidx(nn:nn+stochsamples-1);
      nn = nn+stochsamples;
    end
    if Dl>0
      sXlabels = Xlabels(:,randn);
    end
    sX = X(:,randn);
    sL = shmahd_laplacian( sX, sXlabels, swork );

    %%% Compute statistics %%%
    [ Ei, Ji, fX, gP ] = shmahd_objfunc( Bi'*sX-Pi(:,swork.onesNx), sL, swork );
    if ~stocheckfull
      if prevJ<0
        prevJ = Ji;
        prevE = Ei;
      end
      J = Ji;
      E = Ei;
      %J = 0.5*(prevJ+Ji);
      %prevJ = J;
      %E = 0.5*(prevE+Ei);
      %prevE = E;
    end
  end

  if ~stochastic || mod(I,stocheck)==0
    %%% Determine if there was improvement %%%
    mark = '';
    if (  testJ && (J<bestIJE(2)||(J==bestIJE(2)&&E<=bestIJE(3))) ) || ...
       ( ~testJ && (E<bestIJE(3)||(E==bestIJE(3)&&J<=bestIJE(2))) )
      bestB = Bi;
      bestP = Pi;
      bestIJE = [ I J E bestIJE(4)+1 ];
      mark = ' *';
    end

    %%% Print statistics %%%
    if mod(I,stats)==0
      fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J00,E,mark);
      J00=J;
    end

    %%% Determine if algorithm has to be stopped %%%
    if I>=maxI || ~isfinite(J) || ~isfinite(E) || (I>=minI && abs(J-J0)<epsilon)
      fprintf(logfile,'%s stopped iterating, ',fn);
      if I>=maxI
        fprintf(logfile,'reached maximum number of iterations\n');
      elseif ~isfinite(J) || ~isfinite(E)
        fprintf(logfile,'reached unstable state\n');
      else
        fprintf(logfile,'objective function value has stabilized\n');
      end
      break;
    end

    J0 = J;
  end % if ~stochastic || mod(I,stocheck)==0

  I = I+1;

  %%% Update parameters %%%
  Pi = Pi-rateP.*gP;
  if ~stochastic
    G = 2*slope*(X*fX');
  else
    G = 2*slope*(sX*fX');
  end
  %%% not orthonorm %%%
  %Bi = Bi-rateB.*G;
  %%% orthonorm 1st order approx %%%
  Bi = Bi-rateB.*(G-Bi*(G'*Bi));
  %%% orthonorm 2nd order approx (check this!!!) %%%
  %Bp = Bi; GBp = G'*Bp; Bi = Bi-rateB.*(G-Bp*GBp); Bi = Bi+((rateB^2)/2).*(G*(GBp'-GBp)-Bp*(G'*G)+Bp*(GBp*GBp));
  %%% orthonorm higher order approx %%%
  %P = rateB.*(G*Bi'-Bi*G');
  %PP = P*Bi;
  %Bi = Bi-PP;
  %PP = P*PP;
  %Bi = Bi+(1/2).*PP;
  %PP = P*PP;
  %Bi = Bi-(1/6).*PP;
  %PP = P*PP;
  %Bi = Bi+(1/24).*PP;
  %PP = P*PP;
  %Bi = Bi-(1/120).*PP;
  %PP = P*PP;
  %Bi = Bi+(1/720).*PP;
  %PP = P*PP;
  %Bi = Bi-(1/540).*PP;
  %PP = P*PP;
  %Bi = Bi+(1/40320).*PP;

  %%% Parameter constraints %%%
  if orthonormal
    Bi = orthonorm(Bi);
  end

end % while true

%%% Parameter constraints %%%
if orthonormal
  bestB = orthonorm(bestB);
end

%%% Compute final statistics %%%
if stochastic && stochfinalexact && ~stocheckfull
  [ E, J ] = shmahd_objfunc( bestB'*X-bestP(:,work.onesNx), L, work );

  fprintf(logfile,'%s best iteration approx: I=%d J=%f E=%f\n',fn,bestIJE(1),bestIJE(2),bestIJE(3));
  bestIJE(2) = J;
  bestIJE(3) = E;
end

tm = toc;
einfo.time = einfo.time+tm;
einfo.E = bestIJE(3);
einfo.J = bestIJE(2);
einfo.I = bestIJE(1);
einfo.impI = bestIJE(4)/max(I,1);
fprintf(logfile,'%s best iteration: I=%d J=%f E=%f\n',fn,bestIJE(1),bestIJE(2),bestIJE(3));
fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJE(4)/max(I,1));
fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);

if ~verbose
  fclose(logfile);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                            Helper functions                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = shmahd_laplacian( X, Xlabels, work )

  if sum(size(Xlabels))>0
    if work.dtype.euclidean
      y2 = sum((Xlabels.*Xlabels),1);
      L = Xlabels'*Xlabels;
      L = y2(work.onesNx,:)'+y2(work.onesNx,:)-L-L;
      L = L./max(L(:)); L=check_this;
      L = exp(-L./work.simbw);
    elseif work.dtype.cosine
      ysd = sqrt(sum(Xlabels.*Xlabels,1));
      Xlabels = Xlabels./ysd(work.onesDl,:);
      %L = (Xlabels'*Xlabels+1)./2;
      L = Xlabels'*Xlabels; %%% use if all labels are positive
      L(L<0) = 0;
    end

  else
    if work.dtype.euclidean
      x2 = sum((X.*X),1);
      L = X'*X;
      L = x2(work.onesNx,:)'+x2(work.onesNx,:)-L-L;
      L = L./max(L(:)); L=check_this;
      L = exp(-L./work.simbw);
    elseif work.dtype.cosine
      xsd = sqrt(sum(X.*X,1));
      X = X./xsd(work.onesD,:);
      L = (X'*X+1)./2;
      L(L<0) = 0;
    end
  end

  L = diag(sum(L))-L;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ E, J, fX, gP ] = shmahd_objfunc( rX, L, work )
  %%% Compute statistics %%%
  Y = sign(rX);
  E = work.overNxNxDr*sum(sum( Y.*(Y*L) ));
  Y = tanh(work.slope*rX);
  YL = Y*L;
  J = work.overNxNxDr*sum(sum( Y.*YL ));

  %%% Compute gradient %%%
  if nargout>2
    fX = work.overNxNxDr*(YL.*(1-Y.*Y));
    gP = -median(rX,2);
    %gP = zeros(size(rX,1),1);
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ E, J, fX, gP ] = shmahd_objfunc2( rX, Ln, work )

  Ld = work.Nx*eye(work.Nx)-ones(work.Nx)-Ln;

  %%% Compute statistics %%%
  Y = sign(rX);
  En = sum(sum( Y.*(Y*Ln) ));
  Ed = sum(sum( Y.*(Y*Ld) ));
  E = En/Ed;

  Y = tanh(work.slope*rX);
  YLn = Y*Ln;
  YLd = Y*Ld;
  Jn = sum(sum( Y.*YLn ));
  Jd = sum(sum( Y.*YLd ));
  J = Jn/Jd;

  %%% Compute gradient %%%
  if nargout>2
    YY = 1-Y.*Y;
    fX = (J/Jn)*(YLn.*YY)-(J/Jd)*(YLd.*YY);
    gP = -median(rX,2);
    %gP = zeros(size(rX,1),1);
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = orthonorm( X )
  [ X, dummy ] = qr(X,0);
