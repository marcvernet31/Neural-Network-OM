clear;

% Parameters for dataset generation
%
num_target = [1,2,3,4];
tr_freq    = 0.5;        
tr_p       = 250;       
te_q       = 250;       
tr_seed    = 123456;    
te_seed    = 789101;    


% Parameters for optimization
%   Search Direction:
%       isd=1 : GM; isd=2 : CGM; isd=3 : BFGS(QN); isd=4 SGM,
%       icg=1 : FR; icg=2 : PR+;
%       irc=0 : no restart; irc=1 : RC1; irc=2 : RC2; 
la = 1.00;                                                    % L2 regularization.
epsG = 10^-6; kmax = 10000;                                   % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 7; icg = 2; irc = 2 ; nu = 1.0;                         % Search direction.
sg_ga1 = 0.05; sg_al0=2; sg_ga2=0.3;                           % stochastic gradient


% Optimization
%
t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]= uo_nn_solve(num_target, tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu);
t2=clock;

fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
fprintf(' train accuracy = %d .\n', tr_acc);
fprintf(' test accuracy = %d .\n', te_acc);
uo_nn_Xyplot(Xte, yte, wo);




