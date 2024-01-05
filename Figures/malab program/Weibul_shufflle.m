function [WS_theta,WS_kappa, WS_lamda, LS_sample, Sq_shuffle_sample] = Weibul_shufflle(complex_bits, l, type)
%   complex_bits: type and l=400 で100回シャッフルした結果を記述
    WS_theta = zeros(1, 7);
    WS_kappa = zeros(1, 7);
    WS_lamda = zeros(1, 7);
    
    for ind = 1 : 7
        test = complex_bits{type, ind}(20*l:end-1, :);   
        LS_sample = zeros(100, 500);
        Sq_shuffle_sample = zeros(100, 500);
        thetas = zeros(1, 100);
        kappa = zeros(1, 100);
        lamdas = zeros(1, 100);

        for i = 1 : 100
            shuffle_single;
            fitfun = fittype( @(theta,k, x) exp(-k*(x/lamda).^theta));
            
            x0 = [0, 0];
            L = 500;
            [fitted_curve,gof] = fit((0:L-1).',Sq_shuffle(1:L).',fitfun,'StartPoint',x0);
            coeffvals = coeffvalues(fitted_curve);
            LS_sample(i, :) = lifespan_shuffle;
            Sq_shuffle_sample(i, :) = Sq_shuffle;

            thetas(i) = coeffvals(1);
            kappa(i) = coeffvals(2);
            lamdas(i) = lamda;
        end
        
        WS_theta(1, ind) = mean(thetas);
        WS_kappa(1, ind) = mean(kappa);
        WS_lamda(1, ind) = mean(lamdas);
    end
    

end

