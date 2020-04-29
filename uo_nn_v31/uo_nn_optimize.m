    %FIRST DERIVATIVE METHOD
    %   Stopping criterium:
    %       epsG: conv. tolerance
    %       kmax: maxim. iterations
    %   BLS:
    %       kmaxBLS : max.num. iterations BLS
    %       epsal : BLS parameter
    %   Search Direction:
    %       isd=1 : GM; isd=2 : CGM; isd=3 : BFGS; 
    %       icg=1 : FR; icg=2 : PR+;
    %       irc=0 : no restart; irc=1 : RC1; irc=2 : RC2; 
function [x, niter] = uo_nn_optimize(x,f,g,epsG,kmax,almax,c1,c2,isd,icg,irc,nu, kmaxBLS,epsal)

    I = eye(35);
    H = I;
    k = 0; x_1 = x; d_1 = 0;
    
    while norm(g(x)) >= epsG && k < kmax
        
        [d] = uo_descent_direction(isd, icg, irc, nu, x_1, x, g, d_1, H, k);
        [al,iout] = uo_BLSNW32(f,g,x,d,almax,c1,c2,kmaxBLS,epsal);
        
        x_1 = x;
        x = x + al * d;
        
        %Actualitzations for Quasi-Newton
        if(isd==3)
            s = x - x_1;
            y = g(x) - g(x_1);
            p = 1 / (y'*s);
            H = (I - p*s*y') * H * (I - p*y*s') + p*s*s';
        end  
        
        d_1 = d;
        k = k + 1;
    end
    wo = x;
    niter = k;
end
