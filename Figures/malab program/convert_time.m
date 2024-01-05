function [I] = convert_time(t, interval,itr)
    %PHI時間tを位置時間Tに変換
    init = interval/itr;
    
    start = itr*(init+t-1)-interval+3;
    ends = itr*(init+t-1)+2;
    I = start:ends;
end

