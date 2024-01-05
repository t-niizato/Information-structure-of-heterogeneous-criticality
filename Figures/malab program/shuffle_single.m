lifespan_shuffle = zeros(1, 500);
Sq_shuffle = zeros(1, 500);

test = Shuffle(test);

for n = 1 : 10
    one_test = test(: ,n);
    index = find(one_test==0);
    result = index-circshift(index, 1)-1;
    result = result(result>0);
    for t = 1 : length(result)
        lifespan_shuffle(result(t)) = lifespan_shuffle(result(t)) + 1;
    end
end

for t = 1 :500
    Sq_shuffle(t) = 1 - sum(lifespan_shuffle(1:t-1))/sum(lifespan_shuffle(1:end));
end
    
freq = sum(test,"all");
M_size = 10*length(one_test);
lamda = M_size/freq;
