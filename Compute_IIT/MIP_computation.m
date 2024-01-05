clearvars;

%% data_set
data_import;
index = [1, 2, 3, 4, 5, 6, 7];
interval = 400; % 1 sec for 20 steps 

%% parameters for phi computation
params.tau = 3; % time lag
options.type_of_dist = 'Gauss'; % type of probability distributions
options.type_of_phi = 'star'; % type of phi
options.type_of_MIPsearch = 'Exhaustive'; % type of MIP search
options.normalization = 0; % normalization by Entropy
%% compute all phi data
% set paramters
itr = 10; % computional step(0.5 sec)
bits = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

phis_dtheta = cell(1, 7);
phis_dspeed = cell(1, 7);
CUT_dtheta = cell(1, 7);
CUT_dspeed = cell(1, 7);

for ind = 1 : length(index)
    ind
    [cut, phis] = MIP_function_para(dshita{ind}, itr, interval, params, options);
    [cut2, phis2] = MIP_function_para(accel{ind}, itr, interval, params, options);
    phis_dtheta{ind} = phis;
    phis_dspeed{ind} = phis2;
    CUT_dtheta{ind} = sum((cut-1).*bits, 2);
    CUT_dspeed{ind} = sum((cut2-1).*bits, 2);
    
    writematrix(phis_dtheta{ind} , "PHIs_dtheta_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(phis_dspeed{ind} , "PHIs_dspeed_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(CUT_dtheta{ind} , "MIP_CUT_dtheta_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(CUT_dspeed{ind} , "MIP_CUT_dspeed_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
end


