function[wk,niter]=uo_GM(w,f,g,epsG,kmax,epsal,kmaxBLS,almax,c1,c2)
    wk=[w];

    k=1;
    while norm(g(w)) > epsG && k < kmax 
        d = -g(w); 
        if k ~= 1
            almax = 2*(f(wk(:,k)) - f(wk(:,k-1)))/(g(wk(:,k))'*d);
            %almax = almax_1 * (g(wk(:,k-1))'*d_1) / (g(wk(:,k))'*d);
        end
        [al, iout] = uo_BLSNW32(f, g, w, d, almax, c1, c2, kmaxBLS, epsal); 

        w = w + al*d;
        wk = [wk w];
   
        k = k + 1;
    end
    niter=k;
end

