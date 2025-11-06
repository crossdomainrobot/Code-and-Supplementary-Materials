%% ===============================================================
%  批量计算 & 可视化雪车赛道（100 组）：
%  - 从 Excel 读取 100×33 高度差
%  - 计算 3-D 中心线、高度、坡度、2-D/3-D 曲率
%  - 误差按真实值着色
% ===============================================================

clear;  clc;  close all;

%% ---------------- 0. 路径参数 ----------------
% ❶ 数据目录（中心线 xy 与分段点数）
dataDir = 'D:\Aresearch\雪车论文\雪车论文2\雪车\相关程序(1)\fendauncs';           % ← 修改
% ❷ Excel 高度差文件（100 × 33）
excelFile = 'D:\Aresearch\雪车论文\雪车论文2\数据\方案对比\LLM对比\数据处理\Ours真实收敛值Whistler_LastValues_100x33.xlsx';
% ❸ PNG 保存目录
saveDir  = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据可视化\Ours无干扰\赛道';
if ~exist(saveDir,'dir'); mkdir(saveDir); end

%% ---------------- 1. 读入中心线 (一次即可) ----------------
dataXY = load(fullfile(dataDir,'2-xy-www1.txt'));         % N×2 : [x,y]
x0 = dataXY(:,1);   y0 = dataXY(:,2);
segment_points = load(fullfile(dataDir,'segment_lengthswww2.txt'));  % 33×1
nSeg = numel(segment_points);     nPts = numel(x0);
assert(sum(segment_points)==nPts, "总离散点数与 segment_lengths 不匹配！");

%% ---------------- 2. 真实高程差 (33) ----------------
realVals = [ ...
   10.71572,  6.14825,  4.97581,  6.31038,  3.12531, ...
    2.55046,  2.20386,  1.68713,  1.76053,  3.40447, ...
    2.26344,  2.30372,  3.10214,  5.55557,  3.00872, ...
    2.10999,  2.34004,  7.16291,  4.07776,  1.99409, ...
    2.66388,  6.46395,  4.66176,  3.39000,  3.71014, ...
    5.67758,  4.79036,  2.41349, 18.34033,  5.10889, ...
    4.02582,  4.85554,  5.79760 ];
assert(numel(realVals)==nSeg, '真实高程差必须为 33 个元素！');

%% ---------------- 3. 读入 Excel (100 × 33) ----------------
rawH = readmatrix(excelFile);       % 默认无表头
assert(size(rawH,2)==nSeg,  'Excel 列数应为 33！');
nCases = size(rawH,1);              % 100

fprintf('>>> 成功读取 %d 组 height_differences，将依次绘图...\n', nCases);

%% ---------------- 4. 主循环：100 组 ----------------
for c = 1:nCases
    height_differences = rawH(c,:);                 % 1×33
    %% === 4.1 计算 Z 坐标 ===
    startZ   = sum(height_differences);             % 起点最高
    cumDrop  = cumsum([0, height_differences]);     % 34
    zBreaks  = startZ - cumDrop;                    % 34 个分段高程
    
    z = zeros(nPts,1);
    idx = 1;
    for s = 1:nSeg
        idxEnd = idx + segment_points(s) - 1;
        z(idx:idxEnd) = linspace(zBreaks(s), zBreaks(s+1), segment_points(s));
        idx = idxEnd + 1;
    end
    
    %% === 4.2 曲率、坡度等 ===
    dx2d   = diff(x0);    dy2d = diff(y0);
    dist2d = sqrt(dx2d.^2 + dy2d.^2);               % N-1
    cumDist = [0; cumsum(dist2d)];                  % N
    
    dz          = diff(z);
    slopeSeg    = dz ./ dist2d;                     % N-1
    slopeFull   = [slopeSeg(1); slopeSeg];          % 与 z 对齐
    maskZero    = (dz==0);    slopeFull([false; maskZero]) = NaN;
    
    dx1 = gradient(x0);  dy1 = gradient(y0);  dz1 = gradient(z);
    ddx = gradient(dx1); ddy = gradient(dy1); ddz = gradient(dz1);
    
    curv2D = abs(dx1 .* ddy - ddx .* dy1) ./ (dx1.^2 + dy1.^2).^1.5;
    num    = sqrt( (ddx.^2 + ddy.^2 + ddz.^2) .* (dx1.^2 + dy1.^2 + dz1.^2) ...
                 - (dx1.*ddx + dy1.*ddy + dz1.*ddz).^2 );
    den    = (dx1.^2 + dy1.^2 + dz1.^2).^1.5;
    curv3D = num ./ den;
    
    %% === 4.3 阈值处理 ===
    curv2D(curv2D < 0.01) = 0.01;
    curv3D(curv3D < 0.01) = 0.01;
    slopeFull(slopeFull < -0.204) = -0.204;
    
    slopeFull(isnan(slopeFull)) = 0;
    curv2D(isnan(curv2D)) = 0;
    curv3D(isnan(curv3D)) = 0;
    
    %% === 4.4 误差计算 & 颜色映射 ===
    errSeg = height_differences - realVals;         % 1×33
    % 4-色渐变锚点
    cNegMax = hex2rgb('#3ca3b0');   cNegMid = hex2rgb('#26bc8c');
    cZero   = hex2rgb('#efc348');   cPosMax = hex2rgb('#f11c00');
    
    minErr  = min(errSeg);    maxErr = max(errSeg);
    negMid  = 0.5 * minErr;   % 负方向中点
    
    anchorErr = [minErr,  negMid,    0,   maxErr];
    anchorRGB = [cNegMax; cNegMid; cZero; cPosMax];
    
    nColor = 256;   errSpan = linspace(minErr, maxErr, nColor)';
    cmap = [interp1(anchorErr, anchorRGB(:,1), errSpan, 'linear','extrap'), ...
            interp1(anchorErr, anchorRGB(:,2), errSpan, 'linear','extrap'), ...
            interp1(anchorErr, anchorRGB(:,3), errSpan, 'linear','extrap')];
    err2rgb = @(e) interp1(errSpan, cmap, e, 'linear', 'extrap');
    
    %% === 4.5 绘图 ===
    fig = figure('Visible', 'off', ...
                 'Units', 'pixels', ...
                 'Position', [100 100 1200 600]);   % [L B W H]
    hold on; grid on; view(3);
    
    ax            = gca;
    ax.LineWidth  = 1.5;
    ax.FontName   = 'Times New Roman';
    ax.FontSize   = 30;
    ax.XTick = linspace(0, 600, 2);
    ax.YTick = linspace(-100, 100, 2);
    ax.ZTick = linspace(0, 100, 2);
    
    xlabel('X (m)', 'FontName','Times New Roman', 'FontSize',36);
    ylabel('Y (m)', 'FontName','Times New Roman', 'FontSize',36);
    zlabel('Z (m)', 'FontName','Times New Roman', 'FontSize',36);
    
    % 调整 XYLabel 位置
    ax.XLabel.Units = 'normalized'; ax.YLabel.Units = 'normalized';
    ax.XLabel.Position = ax.XLabel.Position + [-0.46  0.16  0.2];
    ax.YLabel.Position = ax.YLabel.Position + [ 0.55  0.14  0];
    
    % —— 3-D 分段彩色中心线 —— %
    idx = 1;
    for s = 1:nSeg
        idxEnd = idx + segment_points(s) - 1;
        plot3(x0(idx:idxEnd), y0(idx:idxEnd), z(idx:idxEnd), ...
              '-', 'Color', err2rgb(errSeg(s)), 'LineWidth', 4);
        idx = idxEnd + 1;
    end
    
    % —— 2-D 轨迹褐色投影 —— %
    projZ = min(z) - 0.5;
    brown = hex2rgb('#000000');
    plot3(x0, y0, projZ*ones(size(x0)), '-', ...
          'Color', brown, 'LineWidth', 2);
    
    view(45,25);
    hold off;
    
    %% === 4.6 保存 PNG ===
    pngName = fullfile(saveDir, sprintf('track_visualization_%03d.png', c));
    exportgraphics(fig, pngName, 'Resolution', 360);
    close(fig);
    
    fprintf('√ [%3d/%d] 已保存：%s\n', c, nCases, pngName);
end

fprintf('\n==== 全部完成！100 张图片已生成。====\n');

%% ---------------- 5. 工具函数 ----------------
function rgb = hex2rgb(hexstr)
% 将十六进制颜色 (#rrggbb) 转换为 [r g b] (0-1)
    if hexstr(1)=='#', hexstr = hexstr(2:end); end
    rgb = sscanf(hexstr,'%2x%2x%2x',[1 3]) / 255;
end
