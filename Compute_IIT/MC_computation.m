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
options.type_of_complexsearch = 'Exhaustive'; % type of complex search
options.normalization = 0; % normalization of phi by Entropy

%% compute all phi data
% set paramters
itr = 10; % computional step(0.5 sec)
bits = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

% results
main_phis_dtheta = cell(1, 7);
main_phis_dspeed = cell(1, 7);
Sum_main_phis_dtheta = cell(1, 7);
Sum_main_phis_dspeed = cell(1, 7);
M_complex_dtheta = cell(1, 7);
M_complex_dspeed = cell(1, 7);
M_num_dtheta = cell(1, 7);
M_num_dspeed = cell(1, 7);

for ind = 1 : length(index)
     ind
     tic
     [phis_T,Sum_T, comp_T, comp_num_T] = Main_Complex_function_para(dshita{ind}, itr,interval, params, options);
     [phis_A,Sum_A, comp_A, comp_num_A] = Main_Complex_function_para(accel{ind}, itr, interval, params, options);
     toc
     
     main_phis_dtheta{ind} = phis_T;
     Sum_main_phis_dtheta{ind} = Sum_T;
     M_complex_dtheta{ind} = sum(comp_T.*bits, 2);
     M_num_dtheta{ind} = comp_num_T;
     
     main_phis_dspeed{ind} = phis_A;
     Sum_main_phis_dspeed{ind} = Sum_A;
     M_complex_dspeed{ind} = sum(comp_A.*bits, 2);
     M_num_dspeed{ind} = comp_num_A;
     
    writematrix(main_phis_dtheta{ind} , "MC_PHI_dtheta_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(main_phis_dspeed{ind} , "MC_PHI_dspeed_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(Sum_main_phis_dtheta{ind} , "MC_sumPHI_dtheta_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(Sum_main_phis_dspeed{ind} , "MC_sumPHI_dspeed_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(M_complex_dtheta{ind} , "MC_dtheta_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(M_complex_dspeed{ind} , "MC_dspeed_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(M_num_dtheta{ind} , "MCnum_dtheta_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")
    writematrix(M_num_dspeed{ind} , "MCnum_dspeed_" + num2str(ind) + "_" + num2str(interval) + "_20 (sec).csv")  
     
end

