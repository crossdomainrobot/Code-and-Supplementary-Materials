%% =========================================================
%  process_height_sets.m  (含 MeanSlope + 阈值&NaN处理)
% =========================================================
clear; clc;

%% ---------- 0. 路径 ----------
xyDir     = 'D:\Aresearch\雪车论文\雪车论文2\雪车\相关程序(1)\fendauncs';
excelPath = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据处理（cost计算）\Whistler_LastValues_1_1_6.xlsx';
outDir    = 'D:\Aresearch\雪车论文\雪车论文2\结果';
if ~exist(outDir,'dir'); mkdir(outDir); end

%% ---------- 1. 中心线 ----------
xy  = load(fullfile(xyDir,'2-xy-www1.txt'));
x   = xy(:,1);  y = xy(:,2);
segment_pts = load(fullfile(xyDir,'segment_lengthswww2.txt'));
nSeg = numel(segment_pts);  nPts = numel(x);
assert(sum(segment_pts)==nPts);

%% ---------- 2. 预计算二维距离 ----------
dx2d   = diff(x); 
dy2d   = diff(y);
dist2d = sqrt(dx2d.^2 + dy2d.^2);          % 长度 N-1
cumDistFull = [0; cumsum(dist2d)];         % 长度 N

%% ---------- 3. 读 Excel ----------
heightMatrix = readmatrix(excelPath);      % 100 × 33
[nRuns, nCols] = size(heightMatrix);
assert(nCols==nSeg);

%% ---------- 4. 结果容器 ----------
summaryTbl = table('Size',[nRuns 4], ...
    'VariableTypes',{'double','double','double','double'}, ...
    'VariableNames',{'TotalLength_m','TotalDrop_m','MeanCurv3D','MeanSlope'});
runData(nRuns,1) = struct('CumDist',[],'Slope',[],'Curv3D',[]);

%% ---------- 5. 主循环 ----------
for k = 1:nRuns
    hd = heightMatrix(k,:);
    startZ  = sum(hd);
    zBreaks = startZ - cumsum([0, hd]);             % 1×(nSeg+1)
    
    % ----- z -----
    z   = zeros(nPts,1);
    idx = 1;
    for s = 1:nSeg
        idxE = idx + segment_pts(s)-1;
        z(idx:idxE) = linspace(zBreaks(s), zBreaks(s+1), segment_pts(s));
        idx = idxE + 1;
    end
    
    % ----- 坡度 (Slope) -----
    dz         = diff(z);
    slopeSeg   = dz ./ dist2d;                      % 长度 N-1
    slopeFull  = [slopeSeg(1); slopeSeg];          % 长度 N
    slopeFull([false; dz==0]) = NaN;               % Δz = 0 先置 NaN
    
    % ——— 应用坡度上限 0.204 ———
    slopeFull(slopeFull < -0.204) = -0.204;
    
    % ----- 3-D 曲率 (Curv3D) -----
    dx1 = gradient(x);  dy1 = gradient(y);  dz1 = gradient(z);
    ddx = gradient(dx1); ddy = gradient(dy1); ddz = gradient(dz1);
    num = sqrt((ddx.^2+ddy.^2+ddz.^2) .* (dx1.^2+dy1.^2+dz1.^2) ...
              - (dx1.*ddx + dy1.*ddy + dz1.*ddz).^2);
    den = (dx1.^2+dy1.^2+dz1.^2).^1.5;
    curv3D = num ./ den;
    
    % ——— 曲率阈值，小于 0.01 → 0 ———
    curv3D(curv3D < 0.01) = 0.01;
    
    % ----- NaN 统一归 0 -----
    slopeFull(isnan(slopeFull)) = 0;
    curv3D(isnan(curv3D))       = 0;
    
    % ----- 总长 & 下降 -----
    segLen3D  = sqrt(diff(x).^2 + diff(y).^2 + diff(z).^2);
    totalLen  = sum(segLen3D);
    totalDrop = startZ - z(end);
    
    % ----- 汇总 -----
    summaryTbl(k,:) = { totalLen, totalDrop, ...
                        mean(curv3D), ...
                        mean(slopeFull) };
    
    runData(k).CumDist = cumDistFull;
    runData(k).Slope   = slopeFull;
    runData(k).Curv3D  = curv3D;
end

%% ---------- 6. 保存 ----------
csvName = fullfile(outDir,'summary_Whistler_LastValues_1_1_7.csv');
writetable(summaryTbl,csvName);

matName = fullfile(outDir,'runData_Whistler_LastValues_1_1_7.mat');
save(matName,'runData','-v7.3');

fprintf('汇总 CSV => %s\n逐点 MAT => %s\n', csvName, matName);
