function [wk,niter]=uo_BFGS(w,f,g,epsG,kmax,epsal,kmaxBLS,almax,c1,c2)

I = eye(35);
wk = [w];
H = I;
w_1 = w;

k = 1;
while norm(g(w)) > epsG && k < kmax

    if k ~= 1
        sk = w - w_1;
        yk = g(w) - g(w_1);
        pk = 1/(yk'*sk);
        H = (I - pk*sk*yk')*H*(I - pk*yk*sk') + pk*sk*sk';
    end

    d = -H*g(w);

    if k ~= 1
        almax = 2*(f(wk(:,k)) - f(wk(:,k-1)))/(g(wk(:,k))'*d);
    end
    [al, iout] = uo_BLSNW32(f, g, w, d, almax, c1, c2, kmaxBLS, epsal);

    w_1 = w;
    w = w + al*d;
    wk = [wk w];
    k = k + 1;

end
niter=k;

end
