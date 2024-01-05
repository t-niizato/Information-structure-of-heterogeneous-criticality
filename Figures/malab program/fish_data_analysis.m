data_import_fish;
%% 基本データの生成
P = cell(1, 7);
M = cell(1, 7);
K = cell(1, 7);
MK = cell(1, 7);

for i = 1 : 7
    x = data{i}(:, 1:10);
    y = data{i}(:, 11:20);

    cx = mean(x, 2);
    cy = mean(y, 2);
    norm_cxy = sqrt(cx.*cx + cy.*cy);

    dx = x - circshift(x,1);
    dy = y - circshift(y,1);

    norm_dxy = sqrt(dx.*dx + dy.*dy);

    ddx = dx - circshift(dx,1);
    ddy = dy - circshift(dy,1);

    norm_ddxy = sqrt(ddx.*ddx + ddy.*ddy);


    %% Polality
    [nx, ny] = fill_blank(dx,dy,norm_dxy);
    P{i} = sqrt(sum(nx, 2).^2 + sum(ny, 2).^2)/10;

    %% milling
    M{i} = Torus(x - cx, y - cy,nx,ny);

    %% curvature
    X = [cx,cy];
    [L2,R2,K2] = curvature(X);
    % curvature vector : K
    K{i} = sqrt(K2(:,1).^2 + K2(:,2).^2);
    mean_K_generate;
    MK{i} = MKs;
    
end

