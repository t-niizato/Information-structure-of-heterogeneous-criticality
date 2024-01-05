data_name = ["10-fish-2016-part3.txt", "10-fish-part1-2016.txt", "10-fish-part2-2016.txt", "10-fish-part1.txt", "10-fish-part2.txt",...
    '10-fish-part3.txt', '10-fish-part5.txt'];

data = cell(1, 7);

for i = 1:7
   D = readmatrix(data_name(i));
   x = D(:, 3:12);
   y = D(:, 13:22);
   % smoothing
   for j = 1:10
       x(:, i) = movmean(x(:, i), 5);
       y(:, i) = movmean(y(:, i), 5);
   end
   
   data{i} = [x, y];
end

[dshita, accel, speed] = generate_sequence(data);



%{
i=5;
X = data{i};
x = X(:, 1:10);
y = X(:, 11:20);

dx = x - circshift(x,1,1) ;
dy = y - circshift(y,1,1) ;
ddx = dx - circshift(dx,1,1) ;
ddy = dy - circshift(dy,1,1) ;

skip_time = [];
ac_mean = mean(accel{5}, "all");
for i = 1: 10
    skip_time = [skip_time; find(accel{5}(:, i)>ac_mean*50)];
end

skip_time = unique(skip_time);

%}
