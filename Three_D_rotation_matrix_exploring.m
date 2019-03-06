% 图像坐标系是左手系，z轴朝外
% 绕哪个轴旋转，就用左手螺旋定则，左手握住轴，拇指指向该轴正方向，四指方向为旋转的正方向
% 因此对照片中的人脸：
% roll 的正方向为顺时针（绕z轴旋转）
% yaw 的正方向为从左往右（站在观察者的角度）（绕y轴旋转）
% pitch 的正方向为从下往上 （绕x轴旋转）

syms y p r;
% yaw pitch roll

Ry = [cos(y) 0 sin(y);
    0 1 0;
    -sin(y) 0 cos(y)];

Rp = [1 0 0;
    0 cos(p) -sin(p);
    0 sin(p) cos(p)];

Rr = [cos(r) -sin(r) 0;
    sin(r) cos(r) 0;
    0 0 1];

% first pitch, then yaw, at last roll
total = Rp*Ry*Rr;
x1 = [1;0;0];
x2 = [0;1;0];
x3 = [0;0;1];

% compare the following 3 expressions with plot_pose_cube and draw_axis in
% the following file
% https://github.com/natanielruiz/deep-head-pose/blob/master/code/utils.py

total*x1
total*x2
total*x3