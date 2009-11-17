function [bestW, bestU, bestV] = sfma(POS, NEG, W0, U0, V0, varargin)
%
% SFMA: Score Fusion by Maximizing the AUC
%
% [W, U, V] = sfma(POS, NEG, W0, U0, V0, ...)
%
% Input:
%   POS                        - Positive scores matrix
%   NEG                        - Negative scores matrix
%   W0                         - Initial score weights
%   U0                         - Initial sigmoid normalization slopes
%   V0                         - Initial sigmoid normalization displacements
%
% Input (optional):
%   'slope',SLOPE              - Sigmoid slope (defaul=10)
%   'rates',RATES              - Set all learning rates (default=0.5)
%   'rateW',RATEW              - Weights learning rate (default=0.5)
%   'rateU',RATEU              - Slopes learning rate (default=0.5)
%   'rateV',RATEV              - Displacements learning rate (default=0.5)
%   'epsilon',EPSILON          - Convergence criterium (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'approx',AMOUNT            - Use AUC approximation (default=false)
%   'seed',SEED                - Random seed (default=system)
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=10)
%   'stats',STAT               - Statistics every STAT (default={b:1,s:100})
%   'logfile',FID              - Output log file (default=stderr)
%   'verbose',(true|false)     - Verbose (default=true)
%
% Output:
%   W                          - Final learned score weights
%   U                          - Final learned sigmoid slopes
%   V                          - Final learned sigmoid displacements
%
%
% Reference:
%
%   M. Villegas and R. Paredes. "Score Fusion by Maximizing the Area
%   Under the ROC Curve." IbPria 2009.
%
%
% $Revision: 70 $
% $Date: 2009-10-19 09:34:45 +0200 (Mon, 19 Oct 2009) $
%

%
% Copyright (C) 2008-2009 Mauricio Villegas (mvillegas AT iti.upv.es)
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

if strncmp(POS,'-v',2),
  unix('echo "$Revision: 70 $* $Date: 2009-10-19 09:34:45 +0200 (Mon, 19 Oct 2009) $*" | sed "s/^:/sfma: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='sfma:';
minargs=5;

bestW=[];
bestU=[];
bestV=[];

slope=10;
rateW=0.5;
rateU=0.5;
rateV=0.5;

adaptrates=false;
adaptrate=0.1;
adaptdecay=0.9;

epsilon=1e-7;
minI=100;
maxI=1000;

approx=false;
stochastic=false;
stochsamples=10;

logfile=2;
verbose=true;

n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'slope') || ...
         strcmp(varargin{n},'rates') || ...
         strcmp(varargin{n},'rateW') || ...
         strcmp(varargin{n},'rateU') || ...
         strcmp(varargin{n},'rateV') || ...
         strcmp(varargin{n},'adaptrate') || ...
         strcmp(varargin{n},'adaptdecay') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'approx') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'stochsamples') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'logfile'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(varargin{n+1}<0),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'stochastic') || ...
         strcmp(varargin{n},'adaptrates') || ...
         strcmp(varargin{n},'verbose'),
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

[P,D]=size(POS);
N=size(NEG,1);
overPN=1/(P*N);

if exist('rates','var'),
  rateW=rates;
  rateU=rates;
  rateV=rates;
end

sd=std(POS,1,1)+std(NEG,1,1)+std([mean(POS);mean(NEG)],1,1);
if sum(sd==0)>0,
  D=sum(sd~=0);
  POS(:,sd==0)=[];
  NEG(:,sd==0)=[];
  if ~islogical(W0),
    W0(sd==0)=[];
  end
  if ~islogical(U0),
    U0(sd==0)=[];
  end
  if ~islogical(V0),
    V0(sd==0)=[];
  end
  fprintf(logfile,'% warning: some dimensions have a standard deviation of zero\n',fn);
end

if islogical(W0) && W0==true,
  %W0=power(10,6.*(auc(POS,NEG)-0.5));
  W0=exp(20.*(auc(POS,NEG)-0.5))-1;
end
if islogical(U0) && U0==true,
  U0=3./sd(sd~=0);
  sinv=mean(POS)<mean(NEG);
  U0(sinv)=-U0(sinv);
end
if islogical(V0) && V0==true,
  V0=0.5.*(mean(POS)+mean(NEG));
end

if ~islogical(approx),
  origPOS=POS;
  origNEG=NEG;
  origP=P;
  origN=N;
  if max(size(approx))==1,
    if approx<=1,
      P=round(approx*origP);
      N=round(approx*origN);
    else
      P=min([round(approx),origP]);
      N=min([round(approx),origN]);
    end
  else
    if approx(1)<=1,
      P=round(approx(1)*origP);
    else
      P=min([round(approx(1)),origP]);
    end
    if approx(2)<=1,
      N=round(approx(2)*origN);
    else
      N=min([round(approx(2)),origN]);
    end
  end
  overPN=1/(P*N);
  approx=true;
end

if argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif size(W0,1)~=1 || size(U0,1)~=1 || size(V0,1)~=1 || ...
       size(W0,2)~=D || size(U0,2)~=D || size(V0,2)~=D,
  fprintf(logfile,'%s error: W0, U0 and V0 must be row vectors and have the same dimensionality\n',fn);
  return;
elseif size(POS,2)~=D || size(NEG,2)~=D,
  fprintf(logfile,'%s error: POS and NEG must have the same number of columns\n',fn);
  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

if stochastic,
  if exist('seed','var'),
    rand('state',seed);
  end
  if ~exist('stats','var'),
    stats=100;
  end
  onesS=ones(stochsamples,1);
  overS=1/stochsamples;
else
  if ~exist('stats','var'),
    stats=1;
  end
end

W0(W0<0)=0;
W0=W0./sum(W0);

Wi=W0;
Ui=U0;
Vi=V0;
bestWUV=[W0;U0;V0];
bestIJA=[0 0 0 -1];

J0=0;
I=0;

onesP=ones(P,1);
onesN=ones(N,1);
label=[false(N,1);true(P,1)];
adjsA=0.5*P*(P+1);
slope2=slope.^2;

if adaptrates,
  rateW=rateW*ones(size(W0));
  rateU=rateU*ones(size(U0));
  rateV=rateV*ones(size(V0));
  Wv=zeros(size(W0));
  Uv=zeros(size(U0));
  Vv=zeros(size(V0));
  prevW=Wi;
end

fprintf(logfile,'%s D=%d P=%d N=%d\n',fn,D,P,N);
%fprintf(logfile,'%s output: iteration | J | delta(J) | AUC\n',fn);
fprintf(logfile,'%s output: I | 100^J | delta(100^J) | 100^AUC, S=%d\n',fn,stats);

tic;

if ~stochastic,

  while 1,

    if approx,

      nPOS=1./(1+exp(Ui(ones(origP,1),:).*(Vi(ones(origP,1),:)-origPOS)));
      fPOS=nPOS*Wi';
      nNEG=1./(1+exp(Ui(ones(origN,1),:).*(Vi(ones(origN,1),:)-origNEG)));
      fNEG=nNEG*Wi';

      [fPOS,indPOS]=sort(fPOS);
      [fNEG,indNEG]=sort(fNEG);

      [A,ind]=sort([fNEG;fPOS]);
      A=(sum(find(label(ind)))-0.5*origP*(origP+1))./(origP*origN);

      fPOS=fPOS(1:P);
      iPOS=indPOS(1:P);
      nPOS=nPOS(iPOS,:);
      POS=origPOS(iPOS,:);

      fNEG=fNEG(origN-N+1:origN);
      iNEG=indNEG(origN-N+1:origN);
      nNEG=nNEG(iNEG,:);
      NEG=origNEG(iNEG,:);

    else

      expPOS=exp(Ui(onesP,:).*(Vi(onesP,:)-POS));
      nPOS=1./(1+expPOS);
      fPOS=nPOS*Wi';
      expNEG=exp(Ui(onesN,:).*(Vi(onesN,:)-NEG));
      nNEG=1./(1+expNEG);
      fNEG=nNEG*Wi';

      [A,ind]=sort([fNEG;fPOS]);
      A=100^(overPN*(sum(find(label(ind)))-adjsA));

    end

    fPOS=exp(-slope*fPOS);
    fNEG=exp(slope*fNEG);

    J=0;
    cPOS=zeros(P,1);
    cNEG=zeros(N,1);
    for p=1:P,
      fact=fNEG.*fPOS(p);
      J=J+sum(1./(1+fact));
      fact=slope.*fact./((1+fact).^2);
      cPOS(p)=cPOS(p)+sum(fact);
      cNEG=cNEG+fact;
    end
    %J=overPN*J;
    J=100^(overPN*J);

    if isnan(J),
      fprintf(logfile,'%s reached unsable state\n',fn);
      break;
    end

    mark='';
    if J>=bestIJA(2),
      bestWUV=[Wi;Ui;Vi];
      bestIJA=[I J A bestIJA(4)+1];
      mark=' *';
    end

    if mod(I,stats)==0,
      fprintf(logfile,'%dS\t%.6f\t%.6f\t%.6f%s\n',I/stats,J,J-J0,A,mark);
      %fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J0,A,mark);
    end

    if I>=maxI,
      fprintf(logfile,'%s reached maximum number of iterations\n',fn);
      break;
    end

    if I>=minI,
      if abs(J-J0)<epsilon,
        fprintf(logfile,'%s index has stabilized\n',fn);
        break;
      end
    end

    J0=J;
    I=I+1;

    W0=cPOS'*nPOS-cNEG'*nNEG;
    dnPOS=expPOS./((1+expPOS).^2);
    dnNEG=expNEG./((1+expNEG).^2);
    U0=Wi.*(cPOS'*((POS-Vi(onesP,:)).*dnPOS)-cNEG'*((NEG-Vi(onesN,:)).*dnNEG));
    V0=Wi.*Ui.*(cNEG'*dnNEG-cPOS'*dnPOS);

    W0=overPN.*W0;
    U0=overPN.*U0;
    V0=overPN.*V0;

    if adaptrates,
      rateW=rateW.*max(0.5,1+adaptrate*Wv.*W0);
      rateU=rateU.*max(0.5,1+adaptrate*Uv.*U0);
      rateV=rateV.*max(0.5,1+adaptrate*Vv.*V0);

      %if mod(I,stats)==0,
      %  rates=[rateW(:);rateU(:);rateV(:)];
      %  fprintf(logfile,'[%f %f]     ',mean(rates),std(rates));
      %end
    end

    Wi=Wi+rateW.*W0;
    Ui=Ui+rateU.*U0;
    Vi=Vi+rateV.*V0;

    Wi(Wi<0)=0;
    Wi=Wi./sum(Wi);

    if adaptrates,
      ccPOS=zeros(P,1);
      ccNEG=zeros(N,1);
      cW=zeros(1,D);
      cU=zeros(1,D);
      cV=zeros(1,D);
      POSmV=POS-Vi(onesP,:);
      NEGmV=NEG-Vi(onesN,:);
      for p=1:P,
        fact=fNEG.*fPOS(p);
        fact=slope2.*fact.*(fact-1)./((fact+1).^3);
        ccPOS(p)=ccPOS(p)+sum(fact);
        ccNEG=ccNEG+fact;
        nPOSp=nPOS(p,:);
        dnPOSp=dnPOS(p,:);
        POSmVp=POSmV(p,:);
        cW=cW+fact'*(nNEG.*nPOSp(onesN,:));
        cV=cV+fact'*(dnNEG.*dnPOSp(onesN,:));
        cU=cU+fact'*(dnNEG.*dnPOSp(onesN,:).*NEGmV.*POSmVp(onesN,:));
      end

      %ddnPOS=expPOS.*(((1+expPOS).^2)-2.*expPOS.*(1+expPOS))./((1+expPOS).^4);
      %ddnNEG=expNEG.*(((1+expNEG).^2)-2.*expNEG.*(1+expNEG))./((1+expNEG).^4);
      ddnPOS=(dnPOS.^2).*(1./dnPOS-expPOS-expPOS-2);
      ddnNEG=(dnNEG.^2).*(1./dnNEG-expNEG-expNEG-2);

      HWv=Wv.*(ccPOS'*(nPOS.^2)+ccNEG'*(nNEG.^2)-2*cW);
      HUv=Uv.*Wi.*( cPOS'*((POSmV.^2).*ddnPOS)-cNEG'*((NEGmV.^2).*ddnNEG) ...
                   +Wi.*(ccPOS'*((POSmV.*dnPOS).^2)+ccNEG'*((NEGmV.*dnNEG).^2)) ...
                   -2*Wi.*cU );
      HVv=Vv.*Wi.*(Ui.^2).*( cPOS'*ddnPOS-cNEG'*ddnNEG ...
                            +Wi.*(ccPOS'*(dnPOS.^2)+ccNEG'*(dnNEG.^2)) ...
                            -2*Wi.*cV );

      HWv=overPN.*HWv;
      HUv=overPN.*HUv;
      HVv=overPN.*HVv;

      W0=(Wi-prevW)./rateW;

      Wv=adaptdecay*Wv+rateW.*(W0-adaptdecay.*HWv);
      Uv=adaptdecay*Uv+rateU.*(U0-adaptdecay.*HUv);
      Vv=adaptdecay*Vv+rateV.*(V0-adaptdecay.*HVv);

      prevW=Wi;
    end

  end

else % stochastic

  while 1,

    if mod(I,stats)==0,

      nPOS=1./(1+exp(Ui(onesP,:).*(Vi(onesP,:)-POS)));
      fPOS=nPOS*Wi';
      nNEG=1./(1+exp(Ui(onesN,:).*(Vi(onesN,:)-NEG)));
      fNEG=nNEG*Wi';

      [A,ind]=sort([fNEG;fPOS]);
      A=100^(overPN*(sum(find(label(ind)))-adjsA));

      fPOS=exp(-slope*fPOS);
      fNEG=exp(slope*fNEG);

      J=0;
      for p=1:P,
        fact=fNEG.*fPOS(p);
        J=J+sum(1./(1+fact));
      end
      %J=overPN*J;
      J=100^(overPN*J);

      if isnan(J),
        fprintf(logfile,'%s reached unsable state\n',fn);
        break;
      end

      mark='';
      if J>=bestIJA(2),
        bestWUV=[Wi;Ui;Vi];
        bestIJA=[I J A bestIJA(4)+1];
        mark=' *';
      end

      fprintf(logfile,'%dS\t%.6f\t%.6f\t%.6f%s\n',I/stats,J,J-J0,A,mark);
      %fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J0,A,mark);

      if I>=maxI,
        fprintf(logfile,'%s reached maximum number of iterations\n',fn);
        break;
      end

      if I>=minI,
        if abs(J-J0)<epsilon,
          fprintf(logfile,'%s index has stabilized\n',fn);
          break;
        end
      end

      J0=J;

    end

    I=I+1;

    sPOS=POS(1+round((P-1)*rand(stochsamples,1)),:);
    sNEG=NEG(1+round((N-1)*rand(stochsamples,1)),:);
    POSmV=sPOS-Vi(onesS,:);
    NEGmV=sNEG-Vi(onesS,:);

    %if approx,

      %nPOS=1./(1+exp(Ui(ones(origP,1),:).*(Vi(ones(origP,1),:)-origPOS)));
      %fPOS=nPOS*Wi';
      %nNEG=1./(1+exp(Ui(ones(origN,1),:).*(Vi(ones(origN,1),:)-origNEG)));
      %fNEG=nNEG*Wi';

      %[fPOS,indPOS]=sort(fPOS);
      %[fNEG,indNEG]=sort(fNEG);

      %[A,ind]=sort([fNEG;fPOS]);
      %A=(sum(find(label(ind)))-0.5*origP*(origP+1))./(origP*origN);

      %fPOS=fPOS(1:P);
      %iPOS=indPOS(1:P);
      %nPOS=nPOS(iPOS,:);
      %POS=origPOS(iPOS,:);

      %fNEG=fNEG(origN-N+1:origN);
      %iNEG=indNEG(origN-N+1:origN);
      %nNEG=nNEG(iNEG,:);
      %NEG=origNEG(iNEG,:);

    %else

      expPOS=exp(-Ui(onesS,:).*POSmV);
      nPOS=1./(1+expPOS);
      fPOS=nPOS*Wi';
      expNEG=exp(-Ui(onesS,:).*NEGmV);
      nNEG=1./(1+expNEG);
      fNEG=nNEG*Wi';

    %end

    fPOS=exp(-slope*fPOS);
    fNEG=exp(slope*fNEG);

    fact=fNEG.*fPOS;
    dsigm=slope.*fact./((1+fact).^2);

    dnPOS=expPOS./((1+expPOS).^2);
    dnNEG=expNEG./((1+expNEG).^2);

    W0=dsigm'*(nPOS-nNEG);
    U0=Wi.*(dsigm'*(POSmV.*dnPOS-NEGmV.*dnNEG));
    V0=Wi.*Ui.*(dsigm'*(dnNEG-dnPOS));

    W0=overS.*W0;
    U0=overS.*U0;
    V0=overS.*V0;

    if adaptrates,
      rateW=rateW.*max(0.5,1+adaptrate*Wv.*W0);
      rateU=rateU.*max(0.5,1+adaptrate*Uv.*U0);
      rateV=rateV.*max(0.5,1+adaptrate*Vv.*V0);

      %if mod(I,stats)==0,
      %  rates=[rateW(:);rateU(:);rateV(:)];
      %  fprintf(logfile,'[%f %f]     ',mean(rates),std(rates));
      %end
    end

    Wi=Wi+rateW.*W0;
    Ui=Ui+rateU.*U0;
    Vi=Vi+rateV.*V0;

    Wi(Wi<0)=0;
    Wi=Wi./sum(Wi);

    if adaptrates,
      ddsigm=slope2.*fact.*(fact-1)./((fact+1).^3);

      ddnPOS=(dnPOS.^2).*(1./dnPOS-expPOS-expPOS-2);
      ddnNEG=(dnNEG.^2).*(1./dnNEG-expNEG-expNEG-2);

      HWv=Wv.*(ddsigm'*((nPOS-nNEG).^2));
      HUv=Uv.*Wi.*( dsigm'*((POSmV.^2).*ddnPOS-(NEGmV.^2).*ddnNEG) ...
                   +Wi.*(ddsigm'*((POSmV.*dnPOS-NEGmV.*dnNEG).^2)));
      HVv=Vv.*Wi.*(Ui.^2).*( dsigm'*(ddnPOS-ddnNEG) ...
                            +Wi.*(ddsigm'*((dnNEG-dnPOS).^2)));

      HWv=overS.*HWv;
      HUv=overS.*HUv;
      HVv=overS.*HVv;

      W0=(Wi-prevW)./rateW;

      Wv=adaptdecay*Wv+rateW.*(W0-adaptdecay.*HWv);
      Uv=adaptdecay*Uv+rateU.*(U0-adaptdecay.*HUv);
      Vv=adaptdecay*Vv+rateV.*(V0-adaptdecay.*HVv);

      prevW=Wi;
    end

  end

  bestIJA(4)=bestIJA(4)*max(I,1)/(max(I,1)/stats);
end

tm=toc;
fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);
fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJA(4)/max(I,1));
fprintf(logfile,'%s best iteration: I=%d J=%f A=%f\n',fn,bestIJA(1),bestIJA(2),bestIJA(3));

if exist('sd','var'),
  if sum(sd==0)>0,
    WUV=bestWUV;
    D=size(sd,2);
    bestWUV=zeros(3,D);
    bestWUV(:,sd~=0)=WUV;
  end
end

bestW=bestWUV(1,:);
bestU=bestWUV(2,:);
bestV=bestWUV(3,:);
