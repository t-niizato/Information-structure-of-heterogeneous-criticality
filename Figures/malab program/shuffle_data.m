lifespan_shuffle = zeros(7, 500);
Sq_shuffle = zeros(7, 500);
lamda = zeros(7, 1);

for ind = 1 : 7
    test = complex_bits{type, ind}(20*l:end-1, :);
    test = Shuffle(test);
    for n = 1 : 10
        one_test = test(: ,n);
        index = find(one_test==0);
        result = index-circshift(index, 1)-1;
        result = result(result>0);

        for t = 1 : length(result)
            lifespan_shuffle(ind, result(t)) = lifespan_shuffle(ind, result(t)) + 1;
        end
    end

    for t = 1 :500
        Sq_shuffle(ind, t) = 1 - sum(lifespan_shuffle(ind, 1:t-1))/sum(lifespan_shuffle(ind, 1:end));
    end
    
    freq = sum(test,"all");
    M_size = 10*length(one_test);
    lamda(ind) = M_size/freq;
end


