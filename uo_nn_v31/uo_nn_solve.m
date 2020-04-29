%INPUT PARAMETERS
%Dataset generation
    %   num_target     %tr_freq        
    %   tr_p           %te_q      
    %   tr_seed        %te_seed  
    %   la : L2 regularization.
    %   Stopping criterium:     epsG: conv. tolerance
    %                           kmax: maxim. iterations
    %   Linesearch:
    %       ils : ?;  ialmax : ?;
    %       kmaxBLS : max.num. iterations   epsal : tolerance
    %       c1; c2;
    %   Search Direction:
    %       isd=1 : GM; isd=2 : CGM; isd=3 : BFGS(QN); isd=4 SGM,
    %       icg=1 : FR; icg=2 : PR+;
    %       irc=0 : no restart; irc=1 : RC1; irc=2 : RC2; 
    %   Stochastic Gradient(?)
    %       isg_m; isg_al; isg_k;
    
%OUTPUT PARAMETERS
    %   Xtr : (TRAINING)array of vectorized digits
    %   ytr : (TRAINING) associated output 
    
    %   Xte : (TEST) array of vectorized digits
    %   yte : (TEST) associated output
    %   wo : vector of weights
    %   fo : objective function value at w*
    
    %   tr_acc : training acuracy
    %   te_acc : test acuracy
    %   niter : num. iterations
    %   tex : execution time of uo_nn_optimize

function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target, tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu)

    %Dataset generation
    te_freq = -5;
    [Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    [Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);

        
    %Calculate the w*
    sig = @(X) 1./(1+exp(-X)); 
    y = @(X,w) sig(w'*sig(X)); 
    L = @(w) norm(y(Xtr,w)-ytr)^2 + (la*norm(w)^2)/2;
    gL = @(w) 2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))'+la*w;
    w = zeros(35,1);
    
    t1=clock;
    if(isd == 4)
        [wo, niter] = uo_SGM(w, gL, Xtr, ytr, sg_ga1, sg_al0, sg_ga2, kmax, la);
    else
        [wo, niter] = uo_nn_optimize(w,L,gL,epsG,kmax, ialmax,c1,c2,isd,icg,irc,nu, kmaxBLS,epsal);
    end
    t2=clock;
    tex = etime(t2,t1);
    fo = L(wo);

    %Train acuracy
    y_calc = y(Xtr, wo);
    sumTr = 0;
    for i = 1:tr_p
        sumTr = sumTr + (round(y_calc(i)) == ytr(i));
    end
    tr_acc = double(sumTr * (100/tr_p));
    
    %Test acuracy
    y_calc = y(Xte, wo);
    sumTe = 0;
    for i = 1:te_q
        sumTe = sumTe + (round(y_calc(i)) == yte(i));
    end
    te_acc = double(sumTe * (100/te_q));
end



