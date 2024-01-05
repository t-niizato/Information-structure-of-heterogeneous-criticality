time = size(K2,1);
MKs = zeros(1, time);
inter = 20;

for t = 1 : time
    if t < time - inter
        MKs(t) = sqrt(sum(nanmean(K2(t:t+inter, :), 1).^2));
    else
        MKs(t) = sqrt(sum(nanmean(K2(t:end, :), 1).^2));
    end
end