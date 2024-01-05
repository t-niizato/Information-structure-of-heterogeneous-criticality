function [ m ] = Torus( cx,cy,ux,uy )
%   群れのトーラスを計算する

[time,number] = size(ux);
m = zeros(time,1);

for i = 1 : time 
    
    R = zeros(number,3);
    
    for j = 1 : number
        C = zeros(1,3);
        C(1,1) = cx(i,j) / norm([ cx(i,j)  cy(i,j)]);
        C(1,2) = cy(i,j) / norm([ cx(i,j)  cy(i,j)]);
        D = zeros(1,3);
        D(1,1) = ux(i,j);
        D(1,2) = uy(i,j);
        
        R(j,:) = cross(C,D);
        
    end
    
    m(i) = norm(sum(R)) / number;
    
end


end

