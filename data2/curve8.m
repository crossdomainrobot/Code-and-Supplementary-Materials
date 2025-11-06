%% ===============================================================
%  计算 & 可视化雪车赛道：坡度、2-D/3-D 曲率、总长、高程曲线
%  —— 3-D 中心线按误差渐变着色 + 2-D 轨迹褐色投影 + 阈值处理
% ===============================================================

clear;  clc;  close all;

%% ---------------- 0. 数据目录 ----------------
dataDir = 'D:\Aresearch\雪车论文\雪车论文2\雪车\相关程序(1)\fendauncs';  % ← 修改为实际路径

%% ---------------- 1. 读入中心线 ----------------
data = load(fullfile(dataDir,'2-xy-www1.txt'));   % N×2 : [x,y]
x = data(:,1);  y = data(:,2);
segment_points = load(fullfile(dataDir,'segment_lengthswww2.txt'));   % 33×1
nSeg = numel(segment_points);   nPts = numel(x);
assert(sum(segment_points)==nPts, "总离散点数与 segment_lengths 不匹配！");

%% ---------------- 2. 高度差 ----------------
height_differences = [ ...   % 33×1，单位 m
   10.228909,  6.097909,  5.171409,  6.315900,  3.140909, ...
    1.189545,  2.226065,  1.065909,  1.700909,  3.065909, ...
    2.305909,  2.315909,  2.128409,  5.567472,  3.040909, ...
    2.150909,  2.378409,  8.194815,  4.081009,  2.040909, ...
    2.190451,  6.362909,  4.670909,  3.378409,  3.746409, ...
    5.189932,  4.690909,  2.183096, 17.313956,  4.440909, ...
    3.940909,  4.290128,  6.098659 ];
assert(numel(height_differences)==nSeg, "height_differences 必须是 33 个元素！");

%% ---------------- 3. 构造 Z 坐标 ----------------
startZ   = sum(height_differences);               % 起点最高
cumDrop  = cumsum([0, height_differences]);       % 长度 34
zBreaks  = startZ - cumDrop;                      % 34 个分段高程

z = zeros(nPts,1);
idx = 1;
for s = 1:nSeg
    idxEnd = idx + segment_points(s) - 1;
    z(idx:idxEnd) = linspace(zBreaks(s), zBreaks(s+1), segment_points(s));
    idx = idxEnd + 1;
end

%% ---------------- 4. 曲率、坡度等 ----------------
dx2d   = diff(x);    dy2d = diff(y);
dist2d = sqrt(dx2d.^2 + dy2d.^2);                 % N-1
cumDist = [0; cumsum(dist2d)];                    % N

dz          = diff(z);
slopeSeg    = dz ./ dist2d;                       % N-1
slopeFull   = [slopeSeg(1); slopeSeg];            % 与 z 对齐
maskZero = (dz==0);                               % Δz==0 段
slopeFull([false; maskZero]) = NaN;

dx1 = gradient(x);  dy1 = gradient(y);  dz1 = gradient(z);
ddx = gradient(dx1); ddy = gradient(dy1); ddz = gradient(dz1);

curv2D = abs(dx1 .* ddy - ddx .* dy1) ./ (dx1.^2 + dy1.^2).^1.5;
num   = sqrt( (ddx.^2 + ddy.^2 + ddz.^2) .* (dx1.^2 + dy1.^2 + dz1.^2) ...
            - (dx1.*ddx + dy1.*ddy + dz1.*ddz).^2 );
den   = (dx1.^2 + dy1.^2 + dz1.^2).^1.5;
curv3D = num ./ den;

%% ---------------- 5. ★ 阈值处理 ----------------
curv2D(curv2D < 0.01) = 0.01;
curv3D(curv3D < 0.01) = 0.01;
slopeFull(slopeFull < -0.204) = -0.204;
slopeFull(isnan(slopeFull)) = 0;
curv2D(isnan(curv2D)) = 0;
curv3D(isnan(curv3D)) = 0;

%% ---------------- 6. 误差数组（33 段） ----------------
errSeg = [-0.024811,  1.044705, -0.034901, -4.119380, -0.034401, ...
           1.648261, -0.167951, -0.497198,  0.012379, -0.217468, ...
           0.017469, -0.115426, -0.048731, -0.051661, -0.069765, ...
          -0.019081,  0.000869,  0.906874, -0.077760, -0.053181, ...
           0.027029,  0.209959, -0.002101, -0.205091,  0.019832, ...
          -1.486671,  0.300674, -0.230081, -6.405431, -0.018481, ...
           0.162980,  0.208460, -0.006691 ];
assert(numel(errSeg)==nSeg, "errSeg 必须是 33 个元素！");

%% ---------------- 7. 颜色映射（4 锚点） ----------------
cNegMax = hex2rgb('#3ca3b0');     % 最大负误差
cNegMid = hex2rgb('#26bc8c');     % 中等负误差
cZero   = hex2rgb('#efc348');     % 误差 0
cPosMax = hex2rgb('#f11c00');     % 最大正误差

minErr  = min(errSeg);
maxErr  = max(errSeg);
negMid  = 0.5 * minErr;           % 中等负误差对应误差值

anchorErr = [minErr,  negMid,  0,  maxErr];
anchorRGB = [cNegMax; cNegMid; cZero; cPosMax];   % 4×3

nColor  = 256;
errSpan = linspace(minErr, maxErr, nColor)';

cmap = [ ...
    interp1(anchorErr, anchorRGB(:,1), errSpan, 'linear', 'extrap'), ...
    interp1(anchorErr, anchorRGB(:,2), errSpan, 'linear', 'extrap'), ...
    interp1(anchorErr, anchorRGB(:,3), errSpan, 'linear', 'extrap') ];

err2rgb = @(e) interp1(errSpan, cmap, e, 'linear', 'extrap');

%% ---------------- 8. 整体指标 ----------------
segLen3D = sqrt(diff(x).^2 + diff(y).^2 + diff(z).^2);
totalLen = sum(segLen3D);
fprintf('\n==== 赛道整体指标 ====\n');
fprintf('⮞ 赛道总长 (3-D)  : %.3f m\n', totalLen);
fprintf('⮞ 总下降 (起点-终点): %.3f m\n\n', startZ - z(end));

%% ---------------- 9. 三维中心线 + 2-D 轨迹 ----------------
figure(1); clf; hold on; grid on; view(3);

% === ★ 新增：统一坐标轴/标签样式 ===
ax            = gca;
ax.LineWidth  = 1.5;                 % 轴线宽
ax.FontName   = 'Times New Roman';   % 刻度字体
ax.FontSize   = 30;                  % 刻度字号

xlabel('X (m)', 'FontName','Times New Roman', 'FontSize',36);
ylabel('Y (m)', 'FontName','Times New Roman', 'FontSize',36);
zlabel('Z (m)', 'FontName','Times New Roman', 'FontSize',36);  % 可保留/可删

% —— 3-D 彩色中心线 —— %
idx = 1;
for s = 1:nSeg
    idxEnd = idx + segment_points(s) - 1;
    plot3(x(idx:idxEnd), y(idx:idxEnd), z(idx:idxEnd), ...
          '-', 'Color', err2rgb(errSeg(s)), 'LineWidth', 4);
    idx = idxEnd + 1;
end

% —— 2-D 轨迹褐色投影 —— %
projZ = min(z) - 0.5;                         % 投影到地面稍下方
brown  = hex2rgb('#000000');                  % 褐色 (示例中给出 #000000)
plot3(x, y, projZ*ones(size(x)), '-', ...
      'Color', brown, 'LineWidth', 2, ...
      'DisplayName', '2-D 轨迹');

% 视角微调
view(45,25);

hold off;

%% ---------------- 10. 工具函数 ----------------
function rgb = hex2rgb(hexstr)
% 将 16 进制颜色 "#rrggbb" 转为 [r g b]（0–1）
    if hexstr(1)=='#', hexstr = hexstr(2:end); end
    rgb = sscanf(hexstr,'%2x%2x%2x',[1 3]) / 255;
end
