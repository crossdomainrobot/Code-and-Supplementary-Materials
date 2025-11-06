%% ===============================================================
%  计算 & 可视化雪车赛道：坡度、2-D/3-D 曲率、总长、高程曲线
%  （按固定的 height_differences 33 段, 无梯度下降过程）
% ================================================================

clear; clc; close all;

%% ---------------- 0. 数据目录 ----------------
dataDir = 'D:\Aresearch\雪车论文\雪车论文2\雪车\相关程序(1)\fendauncs';  % 修改成你的实际路径

%% ---------------- 1. 读入中心线 ----------------
data = load(fullfile(dataDir,'2-xy-www1.txt'));   % N×2 : [x,y]
x = data(:,1);
y = data(:,2);

segment_points = load(fullfile(dataDir,'segment_lengthswww2.txt'));   % 33×1
nSeg = numel(segment_points);
nPts = numel(x);

assert(sum(segment_points) == nPts, ...
    '总离散点数与 segment_lengths 不匹配！');

%% ---------------- 2. 指定 33 段高度差 ----------------
height_differences = [ ...   % 单位 m
   10.228909,  6.097909,  5.171409,  6.315900,  3.140909, ...
    1.189545,  2.226065,  1.065909,  1.700909,  3.065909, ...
    2.305909,  2.315909,  2.128409,  5.567472,  3.040909, ...
    2.150909,  2.378409,  8.194815,  4.081009,  2.040909, ...
    2.190451,  6.362909,  4.670909,  3.378409,  3.746409, ...
    5.189932,  4.690909,  2.183096, 17.313956,  4.440909, ...
    3.940909,  4.290128,  6.098659 ];

assert(numel(height_differences) == nSeg, ...
    'height_differences 必须是 33 个元素！');

%% ---------------- 3. 插值 Z 坐标 ----------------
startZ   = sum(height_differences);               % 令起点为最高点
cumDrop  = cumsum([0, height_differences]);       % 长度 34
zBreaks  = startZ - cumDrop;                      % 34 个分段高程

z = zeros(nPts,1);
idx = 1;
for s = 1:nSeg
    idxEnd = idx + segment_points(s) - 1;
    z(idx:idxEnd) = linspace(zBreaks(s), zBreaks(s+1), segment_points(s));
    idx = idxEnd + 1;
end

%% ---------------- 4-A. 2-D 距离 & 坡度 ----------------
dx2d   = diff(x);
dy2d   = diff(y);
dist2d = sqrt(dx2d.^2 + dy2d.^2);          % 长度 N-1

cumDist = [0; cumsum(dist2d)];             % 长度 N

% 每条线段坡度（正值=下坡），与 z 对齐：
dz          = diff(z);                     % 长度 N-1
slopeSeg    = dz ./ dist2d;                % 长度 N-1
slopeFull   = [slopeSeg(1); slopeSeg];     % 长度 N，与 z 对齐

% 可选：把 Δz = 0 的段设为 NaN（图中断开而不是 0）
maskZero = (dz == 0);
slopeFull([false; maskZero]) = NaN;

%% ---------------- 4-B. 2-D / 3-D 曲率 ----------------
% gradient 默认步长为 1，对相对尺度足够；如需更精确可用 gradient(var, cumDist)
dx1 = gradient(x);
dy1 = gradient(y);
dz1 = gradient(z);

ddx = gradient(dx1);
ddy = gradient(dy1);
ddz = gradient(dz1);

curv2D = abs(dx1 .* ddy - ddx .* dy1) ./ (dx1.^2 + dy1.^2).^1.5;

num   = sqrt( (ddx.^2 + ddy.^2 + ddz.^2) .* (dx1.^2 + dy1.^2 + dz1.^2) ...
            - (dx1.*ddx + dy1.*ddy + dz1.*ddz).^2 );
den   = (dx1.^2 + dy1.^2 + dz1.^2).^1.5;
curv3D = num ./ den;

%% ---------------- 4-C. 3-D 总长 ----------------
segLen3D = sqrt(diff(x).^2 + diff(y).^2 + diff(z).^2);
totalLen = sum(segLen3D);

%% ---------------- 5. 输出整体指标 ----------------
fprintf('\n==== 赛道整体指标 ====\n');
fprintf('⮞ 赛道总长 (3-D)  : %.3f m\n', totalLen);
fprintf('⮞ 总下降 (起点-终点): %.3f m\n\n', startZ - z(end));

%% ---------------- 6. 绘图 ----------------
% Fig-1 三维中心线
figure(1); clf;
plot3(x, y, z, '-o', 'LineWidth',1.5);
grid on; view(3);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Figure-1  三维赛道中心线');
% 
% % Fig-2 高程-距离
% figure(2); clf;
% plot(cumDist, z, '-o', 'LineWidth',1.5);
% grid on;
% xlabel('累计二维距离 (m)'); ylabel('高程 Z (m)');
% title('Figure-2  高程-距离 曲线');
% 
% % Fig-4 曲率
% figure(4); clf;
% subplot(2,1,1);
% plot(cumDist, curv2D, '-o', 'LineWidth',1.2);
% grid on; ylabel('\kappa_{2D} (1/m)');
% title('Figure-4-a  2-D 曲率');
% 
% subplot(2,1,2);
% plot(cumDist, curv3D, '-o', 'LineWidth',1.2);
% grid on; xlabel('累计二维距离 (m)'); ylabel('\kappa_{3D} (1/m)');
% title('Figure-4-b  3-D 曲率');
% 
% % Fig-6 坡度
% figure(6); clf;
% plot(cumDist, slopeFull, '-o', 'LineWidth',1.2);
% grid on;
% xlabel('累计二维距离 (m)'); ylabel('Slope');
% title('Figure-6  坡度-距离 曲线');
% 
% %% ---------------- 7. 结果表格 ----------------
% results = table(cumDist, z, slopeFull, curv2D, curv3D, ...
%     'VariableNames', {'CumDist_m','Elevation_m','Slope','Curv2D','Curv3D'});
% % writetable(results, fullfile(dataDir,'track_metrics.csv'));
% % disp('已将各指标写出到 track_metrics.csv');
