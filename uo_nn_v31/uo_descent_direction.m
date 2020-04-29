function [d] = uo_descent_direction(isd, icg, irc, nu, x_1, x, g, d_1, H, k)

    %Restart conditions
    nRestart = 10; %Nombre de iteracions per RC1
    RC1 = @(k) mod(k, nRestart) == 0; 
    RC2 = @(x) abs(g(x)'*g(x_1))/norm(g(x))^2 >= nu;

    I=eye(35);
    
    %GM
    if(isd==1)
        d = -g(x);
        
    %CGM
    elseif(isd==2)
        %Restart
        if((irc == 1 && RC1(k)) || (irc == 2 && RC2(x)))
            d = -g(x);
        %No restart
        else
            if(icg == 1)%FR
                b = (g(x)' * g(x)) / norm(g(x_1))^2;
            elseif(icg == 2)%PR+
                b = max(0, (g(x)' * (g(x) - g(x_1))) / norm(g(x_1))^2);
            end
            d = -g(x) + b*d_1;
        end
        
    %BFGS(Q-N)
    elseif(isd==3)
        d = -H*g(x);
end
