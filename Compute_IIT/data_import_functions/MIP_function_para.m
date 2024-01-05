function [MIP_cut,phis] = MIP_function_para(data, itr, interval, params, options)
% data -> MIP_phis and MIP_cuts

    time_step = round(size(data, 1)/itr);
    start = interval/itr;
    phis = zeros(1, time_step);
    MIP_cut = zeros(time_step, 10);
    
    check = cell(1, time_step);
    
    for i = start:time_step-1
        check{i} = data(itr*(i)-interval+1:itr*(i), :).';
    end

    parfor i = start:time_step-1
        [MIP_cut(i, :), phis(i)] = MIP_search(check{i}, params, options);
    end


end