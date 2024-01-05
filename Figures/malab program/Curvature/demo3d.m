
x = sin(linspace(0,2*pi));
y = cos(linspace(0,4*pi));
z = cos(linspace(0,6*pi));
XX = [x',y',z'];
[L,R,K] = curvature(XX);
figure;
plot(L,R)
xlabel L
ylabel R
title('Curvature radius vs. cumulative curve length')
figure;
h = plot3(XX(:,1),XX(:,2),XX(:,3)); 
grid on; 
axis equal
set(h,'marker','.');
xlabel x
ylabel y
zlabel z
title('3D curve with curvature vectors')
hold on
quiver3(XX(:,1),XX(:,2),XX(:,3),K(:,1),K(:,2),K(:,3));
hold off