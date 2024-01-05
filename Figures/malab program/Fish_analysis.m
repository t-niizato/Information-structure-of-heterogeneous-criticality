fish_data_analysis;

%% 比較用PM
PP = cell(2 , 7);
MM = cell(2 , 7);
KK = cell(2 , 7);
interval = lengths(l);
itr = 10;

for ind = 1 :7
    time_step = round(size(P{ind}, 1)/itr) - interval/itr;
    
    Pm = zeros(1, time_step);
    Pv = zeros(1, time_step);
    Mm = zeros(1, time_step);
    Mv = zeros(1, time_step);
    Km = zeros(1, time_step);
    Kv = zeros(1, time_step);

    a = P{ind};
    b = M{ind};
    c = K{ind};

    for i = 1:time_step
        ranges = 1:interval + itr*(i-1);
        Pm(i) = mean(a(ranges));
        Pv(i) = std(a(ranges));
        Mm(i) = mean(b(ranges));
        Mv(i) = std(b(ranges));
        Km(i) = mean(c(ranges));
        Kv(i) = std(c(ranges));
    end

    PP{1,ind} = Pm;
    PP{2,ind} = Pv;    
    MM{1,ind} = Mm;
    MM{2,ind} = Mv;
    KK{1,ind} = Km;
    KK{2,ind} = Kv;
end

LPP = [];
LMM = [];
LPP2 = [];
LMM2 = [];

for ind = 1 : 7    
    LPP = [LPP; PP{1,ind}.'];
    LMM = [LMM; MM{1,ind}.'];
    LPP2 = [LPP2; PP{2,ind}.'];
    LMM2 = [LMM2; MM{2,ind}.'];
end


%% 比較用: long version

LPP = cell(2 , 7);
LMM = cell(2 , 7);
LKK = cell(2 , 7);
interval = lengths(l);
interval = 100;
itr = 10;

for ind = 1 :7
    time_step = round(size(P{ind}, 1)/itr);
    start = interval/itr;
    
    Pm = zeros(1, time_step);
    Pv = zeros(1, time_step);
    Mm = zeros(1, time_step);
    Mv = zeros(1, time_step);
    Km = zeros(1, time_step);
    Kv = zeros(1, time_step);

    a = P{ind};
    b = M{ind};
    c = K{ind};

    for i = start:time_step-1
            Pm(i) = mean(a(itr*(i)-interval+1:itr*(i)));
            Pv(i) = std(a(itr*(i)-interval+1:itr*(i)));
            Mm(i) = mean(b(itr*(i)-interval+1:itr*(i)));
            Mv(i) = std(b(itr*(i)-interval+1:itr*(i)));
            Km(i) = mean(c(itr*(i)-interval+1:itr*(i)));
            Kv(i) = std(c(itr*(i)-interval+1:itr*(i)));
    end

    LPP{1,ind} = Pm;
    LPP{2,ind} = Pv;    
    LMM{1,ind} = Mm;
    LMM{2,ind} = Mv;
    LKK{1,ind} = Km;
    LKK{2,ind} = Kv;
end

