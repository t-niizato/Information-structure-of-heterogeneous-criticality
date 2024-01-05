function [phis,sum_phis, comp, comp_num] = Main_Complex_function_para(data, itr, interval, params, options)
% data, params -> PHI, Main complex, Main complex number
time_step = round(size(data, 1)/itr);
start = interval/itr;

phis = zeros(1, time_step);
sum_phis = zeros(1, time_step);
comp = zeros( time_step, 10);
comp_num = zeros(1, time_step);

check = cell(1, time_step);

for i = start:time_step-1
    check{i} = data(itr*(i)-interval+1:itr*(i), :).';
end

parfor i = start:time_step-1
   
    [~, ~, main_complexes, phis_main_complexes, ~] = Complex_search(check{i}, params, options);
    
    
    a = zeros(1, 10);
    phis(i) = phis_main_complexes(1);
    sum_phis(i) = sum(phis_main_complexes);
    a(main_complexes{1}) =  1;
    comp(i, :) = a;
    comp_num(i) = size(phis_main_complexes, 1);
    
end
    
     
end

