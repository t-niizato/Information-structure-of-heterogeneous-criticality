function [complex_bits] = Main_Complex_bit_seq(MC_COM)
%   Main Complexを十進法から二進法に変換する関数
    % 出力形式
    complex_bits = cell(2, 7);
    
    for ind = 1 : 7
        times = length(MC_COM{1,ind});
        cb1 = zeros(times, 10);
        cb2 = zeros(times, 10);
        for t = 1 : times
            cb1(t, :) = de2bi(MC_COM{1,ind}(t), 10);
            cb2(t, :) = de2bi(MC_COM{2,ind}(t), 10);
        end
        complex_bits{1, ind} = cb1;
        complex_bits{2, ind} = cb2;
    end
    
end

