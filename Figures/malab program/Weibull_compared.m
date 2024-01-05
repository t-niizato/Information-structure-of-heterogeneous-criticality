function [W_theta, W_kappa, WS_theta, WS_kappa] = Weibull_compared(complex_bits, Sq, l, type)
    W_theta = zeros(1, 7);
    W_kappa = zeros(1, 7);
    W_test = zeros(1, 7);
    WS_theta = zeros(1, 7);
    WS_kappa = zeros(1, 7);
    WS_test = zeros(1, 7);
    shuffle_data;
    
    for ind = 1: 7
        x0 = [0, 0]; 
        fitfun = fittype( @(theta,k, x) exp(-k*(x/lamda(ind)).^theta));
        Sq_sample = Sq(ind, :);
        L = length(Sq_sample(Sq_sample>0));
        [fitted_curve,gof] = fit((0:L-1).',Sq(ind, 1:L).',fitfun,'StartPoint',x0);
        coeffvals = coeffvalues(fitted_curve);
        W_theta(1, ind) = coeffvals(1);
        W_kappa(1, ind) = coeffvals(2);
        W_test(1, ind) = gof.sse;

        x0 = [0, 0]; 
        [fitted_curve,gof] = fit((0:L-1).',Sq_shuffle(ind, 1:L).',fitfun,'StartPoint',x0);
        coeffvals = coeffvalues(fitted_curve);
        WS_theta(1, ind) = coeffvals(1);
        WS_kappa(1, ind) = coeffvals(2);
        WS_test(1, ind) = gof.sse;
    end
end

