function uo_nn_batch(tr_seed,te_seed)
%
% Parameters.
%
tr_p = 250; te_q = 250; tr_freq = .5;                         % Datasets generation
epsG = 10^-6; kmax = 1000;                                    % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=10; epsal=10^-3; c1=0.01; c2=0.45; % Linesearch.
icg = 2; irc = 2 ; nu = 1.0;                                  % Search direction.
isg_ga1 = 0.05; isg_al0=2; isg_ga2=0.3;                       % stochastic gradient



%
% Optimization
%
iheader = 1;
csvfile = strcat('uo_nn_batch_',num2str(tr_seed),'-',num2str(te_seed),'.csv');
fileID = fopen(csvfile ,'w');
t1=clock;

for num_target = 1:10
    for la = [0.0, 1.0, 10.0]
        for isd = [1 3 7]
            [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,isg_ga1,isg_al0,isg_ga2,icg,irc,nu,iheader);
            if iheader == 1 fprintf(fileID,'num_target;   la; isd; niter;     tex; tr_acc; te_acc;  L*\n'); end
            fprintf(fileID,'         %1i; %4.1f; %1i;  %4i; %7.4f;  %5.1f;  %5.1f;  %8.2e\n', mod(num_target,10), la, isd, niter, tex, tr_acc, te_acc, fo);

            iheader=0;
        end
    end
end
t2=clock;
fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
fclose(fileID);
%
end
