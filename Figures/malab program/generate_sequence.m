function [dshita, accel, speed] = generate_sequence(data)
% data_importで作成したデータでIIT用に変換
% 出力はaccelとdshitaのセル
    n = size(data,2);
    accel = cell(1, n);
    dshita = cell(1, n);
    speed = cell(1, n);


    for i = 1 : n
        X = data{i};
        x = X(:, 1:10);
        y = X(:, 11:20);
        dx = x - circshift(x,1,1) ;
        dy = y - circshift(y,1,1) ;
        ddx = dx - circshift(dx,1,1) ;
        ddy = dy - circshift(dy,1,1) ;

        shita = atan2(dy, dx);
        d = atan2(sin(shita-circshift(shita,1,1)), cos(shita-circshift(shita,1,1)));
        s = sqrt(dx.*dx +dy.*dy);
        a = sqrt(ddx.*ddx +ddy.*ddy);

        dshita{i} = d(3:end, :);
        accel{i} = a(3:end, :);
        speed{i} = s(3:end, :);

    end
end

