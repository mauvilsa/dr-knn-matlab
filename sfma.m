function [bestW, bestU, bestV] = sfma_new(POS, NEG, W0, U0, V0, varargin)
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
%   'rateW',RATEW              - Weights learning rate (default=0.1)
%   'rateU',RATEU              - Slopes learning rate (default=0.1)
%   'rateV',RATEV              - Displacements learning rate (default=0.1)
%   'epsilon',EPSILON          - Convergence criteria (default=1e-7)
%   'minI',MINI                - Minimum number of iterations (default=100)
%   'maxI',MAXI                - Maximum number of iterations (default=1000)
%   'stats',STAT               - Statistics every STAT (default=1)
%   'probe',PROBE              - Probe learning rates (default=false)
%   'probeI',PROBEI            - Iterations for probing (default=100)
%   'autoprobe',(true|false)   - Automatic probing (default=false)
%   'adaptrates',(true|false)  - Adapt learning rates (default=false)
%   'adaptrate',RATE           - Adaptation rate (default=false)
%   'adaptdecay',DECAY         - Adaptation decay (default=false)
%   'devel',dPOS,dNEG          - Set the development set (default=false)
%   'seed',SEED                - Random seed (default=system)
%   'stochastic',(true|false)  - Stochastic gradient descend (default=true)
%   'stochsamples',SAMP        - Samples per stochastic iteration (default=1)
%   'stocheck',SIT             - Stats every SIT stoch. iterations (default=100)
%   'stocheckfull',(true|f...  - Stats for whole data set (default=false)
%   'stochfinalexact',(tru...  - Final stats for whole data set (default=true)
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
% $Revision$
% $Date$
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
  unix('echo "$Revision$* $Date$*" | sed "s/^:/sfma: revision/g; s/ : /[/g; s/ (.*)/]/g;"');
  return;
end

fn='sfma:';
minargs=5;

%%% Default values %%%
bestW=[];
bestU=[];
bestV=[];

slope=10;
rateW=0.1;
rateU=0.1;
rateV=0.1;

probeI=100;
probeunstable=0.2;
autoprobe=false;

adaptrates=false;
adaptrate=0.1;
adaptdecay=0.9;

epsilon=1e-7;
minI=100;
maxI=1000;
stats=1;

devel=false;

stochastic=false;
stochsamples=1;
stocheck=100;
stocheckfull=false;
stochfinalexact=true;
testJ=false;
altrange=false;

logfile=2;
verbose=true;

%%% Input arguments parsing %%%
n=1;
argerr=false;
while size(varargin,2)>0,
  if ~ischar(varargin{n}) || size(varargin,2)<n+1,
    argerr=true;
  elseif strcmp(varargin{n},'probemode'),
    eval([varargin{n},'=varargin{n+1};']);
    n=n+2;
  elseif strcmp(varargin{n},'slope') || ...
         strcmp(varargin{n},'rates') || ...
         strcmp(varargin{n},'rateW') || ...
         strcmp(varargin{n},'rateU') || ...
         strcmp(varargin{n},'rateV') || ...
         strcmp(varargin{n},'epsilon') || ...
         strcmp(varargin{n},'minI') || ...
         strcmp(varargin{n},'maxI') || ...
         strcmp(varargin{n},'stats') || ...
         strcmp(varargin{n},'probe') || ...
         strcmp(varargin{n},'probeI') || ...
         strcmp(varargin{n},'adaptrate') || ...
         strcmp(varargin{n},'adaptdecay') || ...
         strcmp(varargin{n},'seed') || ...
         strcmp(varargin{n},'stochsamples') || ...
         strcmp(varargin{n},'stocheck') || ...
         strcmp(varargin{n},'logfile'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1}) || sum(sum(varargin{n+1}<0))~=0,
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'adaptrates') || ...
         strcmp(varargin{n},'stochastic') || ...
         strcmp(varargin{n},'stocheckfull') || ...
         strcmp(varargin{n},'stochfinalexact') || ...
         strcmp(varargin{n},'autoprobe') || ...
         strcmp(varargin{n},'testJ') || ...
         strcmp(varargin{n},'altrange') || ...
         strcmp(varargin{n},'verbose'),
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1}),
      argerr=true;
    else
      n=n+2;
    end
  elseif strcmp(varargin{n},'devel'),
    devel=true;
    devPOS=varargin{n+1};
    devNEG=varargin{n+2};
    n=n+3;
  else
    argerr=true;
  end
  if argerr || n>size(varargin,2),
    break;
  end
end

if exist('probemode','var'),
  rateW=probemode.rateW;
  rateU=probemode.rateU;
  rateV=probemode.rateV;
  minI=probemode.minI;
  maxI=probemode.maxI;
  probeunstable=minI;
  stats=probemode.stats;
  orthoit=probemode.orthoit;
  stochastic=probemode.stochastic;
  devel=probemode.devel;
  work=probemode.work;
  if devel,
    dwork=probemode.dwork;
    devPOS=probemode.devPOS;
    devNEG=probemode.devNEG;
  end
  verbose=false;
  epsilon=0;
  probemode=true;
else
  probemode=false;
end

[P,D]=size(POS);
N=size(NEG,1);

%%% Error detection %%%
if probemode,
elseif argerr,
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs,
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif (sum(size(W0))~=0 && (size(W0,1)~=1 || size(W0,2)~=D)) || ...
       (sum(size(U0))~=0 && (size(U0,1)~=1 || size(U0,2)~=D)) || ...
       (sum(size(V0))~=0 && (size(V0,1)~=1 || size(V0,2)~=D)),
  fprintf(logfile,'%s error: W0, U0 and V0 must be row vectors and have the same dimensionality\n',fn);
  return;
elseif size(POS,2)~=D || size(NEG,2)~=D,
  fprintf(logfile,'%s error: POS and NEG must have the same number of columns\n',fn);
  return;
end

if ~verbose,
  logfile=fopen('/dev/null');
end

%%% Preprocessing %%%
if ~probemode,
  tic;

  if exist('rates','var'),
    rateW=rates;
    rateU=rates;
    rateV=rates;
  end

  onesP=ones(P,1);
  onesN=ones(N,1);
  overPN=1/(P*N);
  label=[false(N,1);true(P,1)];
  adjsA=0.5*P*(P+1);
  slope2=slope*slope;

  sd=std(POS,1,1)+std(NEG,1,1)+std([mean(POS);mean(NEG)],1,1);

  if sum(size(W0))==0,
    W0=exp(20.*(auc(POS,NEG)-0.5))-1;
  end
  if sum(size(U0))==0,
    U0=3./sd;
    sinv=mean(POS)<mean(NEG);
    U0(sinv)=-U0(sinv);
  end
  if sum(size(V0))==0,
    V0=0.5.*(mean(POS)+mean(NEG));
  end

  if sum(sd==0)>0,
    D=sum(sd~=0);
    W0(sd==0)=[];
    U0(sd==0)=[];
    V0(sd==0)=[];
    POS(:,sd==0)=[];
    NEG(:,sd==0)=[];
    if devel,
      devPOS(:,sd==0)=[];
      devNEG(:,sd==0)=[];
    end
    fprintf(logfile,'% warning: some dimensions have a standard deviation of zero\n',fn);
  end

  if stochastic,
    if exist('seed','var'),
      rand('state',seed);
    end
    minI=minI*stocheck;
    maxI=maxI*stocheck;
    stats=stats*stocheck;
    onesS=ones(stochsamples,1);
    overS=1/stochsamples;
  end

  work.onesP=onesP;
  work.onesN=onesN;
  work.overPN=overPN;
  work.label=label;
  work.adjsA=adjsA;
  work.altrange=altrange;
  work.slope=slope;

  if devel,
    dP=size(devPOS,1);
    dN=size(devNEG,1);
    dwork.onesP=ones(dP,1);
    dwork.onesN=ones(dN,1);
    dwork.overPN=1/(dP*dN);
    dwork.label=[false(dN,1);true(dP,1)];
    dwork.adjsA=0.5*dP*(dP+1);
    dwork.altrange=altrange;
  end

  fprintf(logfile,'%s total preprocessing time (s): %f\n',fn,toc);
end

if autoprobe,
  probe=[zeros(2,1),10.^[-4:4;-4:4;-4:4]];
end
if exist('probe','var'),
  tic;
  probecfg.minI=round(probeunstable*probeI);
  probecfg.maxI=probeI;
  probecfg.stats=stats;
  probecfg.stochastic=stochastic;
  probecfg.devel=devel;
  probecfg.work=work;
  if devel,
    probecfg.dwork=dwork;
    probecfg.devPOS=devPOS;
    probecfg.devNEG=devNEG;
  end
  bestIJA=[0,0];
  ratesW=unique(probe(1,probe(1,:)>=0));
  ratesU=unique(probe(2,probe(2,:)>=0));
  ratesV=unique(probe(2,probe(2,:)>=0));
  nW=1;
  while nW<=size(ratesW,2),
    nU=1;
    while nU<=size(ratesU,2),
      nV=1;
      while nV<=size(ratesV,2),
        if ~(ratesW(nW)==0 && ratesU(nU)==0 && ratesV(nV)==0),
          probecfg.rateW=ratesW(nW);
          probecfg.rateU=ratesU(nU);
          probecfg.rateV=ratesV(nV);
          [I,J]=sfma_new(POS,NEG,W0,U0,V0,'probemode',probecfg);
          mark='';
          if I>bestIJA(1) || (I==bestIJA(1) && J>bestIJA(2)),
            bestIJA=[I,J];
            rateW=ratesW(nW);
            rateU=ratesU(nU);
            rateV=ratesV(nV);
            mark=' +';
          end
          if I<probeunstable*probeI,
            if nV==1,
              if nU==1,
                nW=size(ratesB,2)+1;
              end
              nU=size(ratesP,2)+1;
            end
            break;
          end
          fprintf(logfile,'%s rates={%.2E %.2E %.2E} => impI=%.2f J=%.4f%s\n',fn,ratesW(nW),ratesU(nU),ratesV(nV),I/probeI,J,mark);
        end
        nV=nV+1;
      end
      nU=nU+1;
    end
    nW=nW+1;
  end
  fprintf(logfile,'%s total probe time (s): %f\n',fn,toc);
  fprintf(logfile,'%s selected rates={%.2E %.2E %.2E} impI=%.2f J=%.4f\n',fn,rateW,rateU,rateV,bestIJA(1)/probeI,bestIJA(2));
end

W0(W0<0)=0;
W0=W0./sum(W0);

Wi=W0;
Ui=U0;
Vi=V0;
bestWUV=[W0;U0;V0];
bestIJA=[0 0 0 -1];

J00=0;
J0=0;
I=0;

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
fprintf(logfile,'%s output: iteration | J | delta(J) | AUC\n',fn);

if ~probemode,
  tic;
end

%%% Batch gradient descent %%%
if ~stochastic,

  while true,

    %%% Compute statistics %%%
    expPOS=exp(Ui(onesP,:).*(Vi(onesP,:)-POS));
    nPOS=1./(1+expPOS);
    fPOS=nPOS*Wi';
    expNEG=exp(Ui(onesN,:).*(Vi(onesN,:)-NEG));
    nNEG=1./(1+expNEG);
    fNEG=nNEG*Wi';

    [A,ind]=sort([fNEG;fPOS]);
    A=overPN*(sum(find(label(ind)))-adjsA);

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
    J=overPN*J;

    if altrange,
      A=100^A;
      J=100^J;
    end

    if devel,
      A=sfma_index(devPOS,devNEG,Wi,Ui,Vi,dwork);
    end

    %%% Determine if there was improvement %%%
    mark='';
    if isfinite(J) && isfinite(A) && ( ...
       (  testJ && (J>bestIJA(2)||(J==bestIJA(2)&&A>=bestIJA(3))) ) || ...
       ( ~testJ && (A>bestIJA(3)||(A==bestIJA(3)&&J>=bestIJA(2))) )),
      bestWUV=[Wi;Ui;Vi];
      bestIJA=[I J A bestIJA(4)+1];
      mark=' *';
    end

    %%% Print statistics %%%
    if mod(I,stats)==0,
      fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J00,A,mark);
      J00=J;
    end

    %%% Determine if algorithm has to be stopped %%%
    if probemode,
      if bestIJA(4)+(maxI-I)<probeunstable,
        break;
      end
    end
    if I>=maxI || ~isfinite(J) || ~isfinite(A) || (I>=minI && abs(J-J0)<epsilon),
      fprintf(logfile,'%s stopped iterating, ',fn);
      if I>=maxI,
        fprintf(logfile,'reached maximum number of iterations\n');
      elseif ~isfinite(J) || ~isfinite(A),
        fprintf(logfile,'reached unstable state\n');
      else
        fprintf(logfile,'index has stabilized\n');
      end
      break;
    end

    J0=J;
    I=I+1;

    %%% Compute gradient %%%
    W0=cPOS'*nPOS-cNEG'*nNEG;
    dnPOS=expPOS./((1+expPOS).^2);
    dnNEG=expNEG./((1+expNEG).^2);
    U0=Wi.*(cPOS'*((POS-Vi(onesP,:)).*dnPOS)-cNEG'*((NEG-Vi(onesN,:)).*dnNEG));
    V0=Wi.*Ui.*(cNEG'*dnNEG-cPOS'*dnPOS);

    W0=overPN.*W0;
    U0=overPN.*U0;
    V0=overPN.*V0;

    %%% Update parameters %%%
    if adaptrates,
      rateW=rateW.*max(0.5,1+adaptrate*Wv.*W0);
      rateU=rateU.*max(0.5,1+adaptrate*Uv.*U0);
      rateV=rateV.*max(0.5,1+adaptrate*Vv.*V0);
    end
    Wi=Wi+rateW.*W0;
    Ui=Ui+rateU.*U0;
    Vi=Vi+rateV.*V0;

    %%% Parameter constraints %%%
    Wi(Wi<0)=0;
    Wi=Wi./sum(Wi);

    %%% Adapt rates %%%
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

  end % while true

%%% Stochasitc gradient descent %%%
else

  prevJ=1;
  prevA=1;
  
  while true,
    %%% Compute statistics %%%
    if mod(I,stocheck)==0 && stocheckfull,
      nPOS=1./(1+exp(Ui(onesP,:).*(Vi(onesP,:)-POS)));
      fPOS=nPOS*Wi';
      nNEG=1./(1+exp(Ui(onesN,:).*(Vi(onesN,:)-NEG)));
      fNEG=nNEG*Wi';

      [A,ind]=sort([fNEG;fPOS]);
      A=overPN*(sum(find(label(ind)))-adjsA);

      fPOS=exp(-slope*fPOS);
      fNEG=exp(slope*fNEG);

      J=0;
      for p=1:P,
        J=J+sum(1./(1+fPOS(p).*fNEG));
      end
      J=overPN*J;

      if altrange,
        A=100^A;
        J=100^J;
      end
    
      if devel,
        A=sfma_index(devPOS,devNEG,Wi,Ui,Vi,dwork);
      end
    end

    %%% Select random samples %%%
    sPOS=POS(1+round((P-1)*rand(stochsamples,1)),:);
    sNEG=NEG(1+round((N-1)*rand(stochsamples,1)),:);

    %%% Compute statistics %%%
    POSmV=sPOS-Vi(onesS,:);
    NEGmV=sNEG-Vi(onesS,:);

    expPOS=exp(-Ui(onesS,:).*POSmV);
    nPOS=1./(1+expPOS);
    fPOS=nPOS*Wi';
    expNEG=exp(-Ui(onesS,:).*NEGmV);
    nNEG=1./(1+expNEG);
    fNEG=nNEG*Wi';

    Ai=overS*(sum(fPOS>fNEG)+0.5*sum(fPOS==fNEG));

    expon=exp(slope*(fNEG-fPOS));
    Ji=overS*sum(1./(1+expon));

    if altrange,
      Ai=100^Ai;
      Ji=100^Ji;
    end

    if ~stocheckfull,
      J=0.5*(prevJ+Ji);
      prevJ=J;
      if ~devel,
        A=0.5*(prevA+Ai);
        prevA=A;
      end
    end

    if mod(I,stocheck)==0,
      if ~stocheckfull && devel,
        A=sfma_index(devPOS,devNEG,Wi,Ui,Vi,dwork);
      end

      %%% Determine if there was improvement %%%
      mark='';
      if isfinite(J) && isfinite(A) && ( ...
         (  testJ && (J>bestIJA(2)||(J==bestIJA(2)&&A>=bestIJA(3))) ) || ...
         ( ~testJ && (A>bestIJA(3)||(A==bestIJA(3)&&J>=bestIJA(2))) )),
        bestWUV=[Wi;Ui;Vi];
        bestIJA=[I J A bestIJA(4)+1];
        mark=' *';
      end

      %%% Print statistics %%%
      if mod(I,stats)==0,
        fprintf(logfile,'%d\t%.8f\t%.8f\t%.8f%s\n',I,J,J-J00,A,mark);
        J00=J;
      end

      %%% Determine if algorithm has to be stopped %%%
      if probemode,
        if bestIJA(4)+(maxI-I)<probeunstable,
          break;
        end
      end
      if I>=maxI || ~isfinite(J) || ~isfinite(A) || (I>=minI && abs(J-J0)<epsilon),
        fprintf(logfile,'%s stopped iterating, ',fn);
        if I>=maxI,
          fprintf(logfile,'reached maximum number of iterations\n');
        elseif ~isfinite(J) || ~isfinite(A),
          fprintf(logfile,'reached unstable state\n');
        else
          fprintf(logfile,'index has stabilized\n');
        end
        break;
      end

      J0=J;
    end % if mod(I,stocheck)==0

    I=I+1;

    %%% Compute gradient %%%
    dsigm=slope.*expon./((1+expon).^2);

    dnPOS=expPOS./((1+expPOS).^2);
    dnNEG=expNEG./((1+expNEG).^2);

    W0=dsigm'*(nPOS-nNEG);
    U0=Wi.*(dsigm'*(POSmV.*dnPOS-NEGmV.*dnNEG));
    V0=Wi.*Ui.*(dsigm'*(dnNEG-dnPOS));

    W0=overS.*W0;
    U0=overS.*U0;
    V0=overS.*V0;

    %%% Update parameters %%%
    if adaptrates,
      rateW=rateW.*max(0.5,1+adaptrate*Wv.*W0);
      rateU=rateU.*max(0.5,1+adaptrate*Uv.*U0);
      rateV=rateV.*max(0.5,1+adaptrate*Vv.*V0);
    end
    Wi=Wi+rateW.*W0;
    Ui=Ui+rateU.*U0;
    Vi=Vi+rateV.*V0;

    %%% Parameter constraints %%%
    Wi(Wi<0)=0;
    Wi=Wi./sum(Wi);

    %%% Adapt rates %%%
    if adaptrates,
      ddsigm=slope2.*expon.*(expon-1)./((expon+1).^3);

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

  end % while true

  %%% Compute final statistics %%%
  if stochfinalexact && ~stocheckfull,
    Wi=bestWUV(1,:);
    Ui=bestWUV(2,:);
    Vi=bestWUV(3,:);
    [A,J]=sfma_index(POS,NEG,Wi,Ui,Vi,work);
    if devel,
      A=sfma_index(devPOS,devNEG,Wi,Ui,Vi,dwork);
    end

    fprintf(logfile,'%s best iteration approx: I=%d J=%f A=%f\n',fn,bestIJA(1),bestIJA(2),bestIJA(3));
    bestIJA(2)=J;
    bestIJA(3)=A;
  end

  bestIJA(4)=bestIJA(4)*stocheck;
end

if ~probemode,
  tm=toc;
  fprintf(logfile,'%s best iteration: I=%d J=%f A=%f\n',fn,bestIJA(1),bestIJA(2),bestIJA(3));
  fprintf(logfile,'%s amount of improvement iterations: %f\n',fn,bestIJA(4)/max(I,1));
  fprintf(logfile,'%s average iteration time (ms): %f\n',fn,1000*tm/(I+0.5));
  fprintf(logfile,'%s total iteration time (s): %f\n',fn,tm);
end

if ~verbose,
  fclose(logfile);
end

if probemode,
  bestW=bestIJA(4);
  bestU=bestIJA(2);
  return;
end

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                   Helper functions                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, J] = sfma_index(POS, NEG, W, U, V, work)

  P=size(POS,1);
  N=size(NEG,1);
  onesP=work.onesP;
  onesN=work.onesN;
  overPN=work.overPN;
  label=work.label;
  adjsA=work.adjsA;
  altrange=work.altrange;

  %%% Compute fused scores %%%
  nPOS=1./(1+exp(U(onesP,:).*(V(onesP,:)-POS)));
  fPOS=nPOS*W';
  nNEG=1./(1+exp(U(onesN,:).*(V(onesN,:)-NEG)));
  fNEG=nNEG*W';

  %%% Compute AUC %%%
  [A,ind]=sort([fNEG;fPOS]);
  A=overPN*(sum(find(label(ind)))-adjsA);
  if altrange,
    A=100^A;
  end

  %%% Compute index %%%
  if nargout>1,
    slope=work.slope;

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
    J=overPN*J;
    if altrange,
      J=100^J;
    end
  end
