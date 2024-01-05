function [x,y] = fill_blank(x, y, dis)
%   x, y から順番に単にベクトルを出力
%   全く同じ場所に止まっている場合は前回の向きをそのまま受け継ぐとする
    time = size(x, 1);
    for t = 1 : time
        for i = 1 : 10
            if dis(t, i) == 0
                x(t, i) = x(t-1, i);
                y(t, i) = y(t-1, i);
            else
                x(t, i) = x(t, i)/dis(t, i);
                y(t, i) = y(t, i)/dis(t, i);
            end
        end
    end

end

