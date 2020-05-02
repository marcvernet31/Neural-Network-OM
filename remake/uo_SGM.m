function [wk, niter] = uo_SGM(w,Xtr, ytr, sg_ga1, sg_al0, sg_ga2, kmax, la)
    wk = w; niter = 0;
    k = 0; k_sg = floor(kmax * sg_ga2);
    al = 1; al_sg = 0.01 * sg_al0;
    sig = @(X) 1./(1+exp(-X)); 
    y = @(X,w) sig(w'*sig(X)); 

    while k < kmax
        minibatch = randi(size(Xtr, 2), round(size(Xtr, 2) * sg_ga1), 1);
        XtrS = Xtr(:,minibatch);
        ytrS = ytr(:,minibatch);
        gLS = @(w) 2*sig(XtrS)*((y(XtrS,w)-ytrS).*y(XtrS,w).*(1-y(XtrS,w)))'+la*w;
        
        %descend direction
        d = -(1/(size(XtrS, 2))) * gLS(wk);
        
        %alpha
        if(k <= k_sg)
            al = (1 - (k/k_sg)) * sg_al0 + (k/k_sg) * al_sg;
        elseif(k > k_sg)
            al = al_sg;
        end

        wk = wk + al*d;
        k = k + 1;
    end
    niter = k;
end

