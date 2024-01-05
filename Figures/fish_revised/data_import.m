data_name = ["10-fish-2016-part3.txt", "10-fish-part1-2016.txt", "10-fish-part2-2016.txt", "10-fish-part1.txt", "10-fish-part2.txt",...
    '10-fish-part3.txt', '10-fish-part5.txt'];

data = cell(1, 7);

for i = 1:7
   D = readmatrix(data_name(i));
   x = D(:, 3:12);
   y = D(:, 13:22);
   % smoothing
   for j = 1:10
       x(:, i) = movmean(x(:, i), 2);
       y(:, i) = movmean(y(:, i), 2);
   end
   
   data{i} = [x, y];
end

