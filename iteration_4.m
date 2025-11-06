%% ========== 0. 可在此修改的参数 ==========
dataDir  = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据处理（最大最小平均hd）';   % 数据目录
outFile  = fullfile(dataDir,'Whistler_Iteration_Row47.xlsx');           % 输出文件名

skipList = [8 12 13 14 18 29];   % 不提取的 Track 号；如需 26 列再多加一个

excelTargetRow = 47;             % ⬅️ 明确：就要第 47 行
firstCol = 'A';  lastCol = 'CV'; % ⬅️ 明确：只取前 100 列 (A–CV)
nCols   = 100;                   % 列数恒定

%% ========== 1. 计算有效 Track 并预分配 ==========
validTracks = setdiff(1:33, skipList);   % 需要提取的 Track
nTracks     = numel(validTracks);        % 列数 = 有效 Track 数

iterVals = nan(nCols, nTracks);          % 100×nTracks，空位先填 NaN
varNames = cell(1, nTracks);             % 存列标题

%% ========== 2. 逐文件读取第 47 行 ==========
colIdx = 0;   % 填充 iterVals 的列指针
for X = validTracks
    colIdx  = colIdx + 1;
    file    = sprintf('Whistler_Track%d2_data.xlsx', X);
    fPath   = fullfile(dataDir, file);

    if ~isfile(fPath)
        warning('未找到文件 %s — 整列保持 NaN。', file);
        continue
    end

    % -- 2.1 用 readmatrix 直接按行列范围读取 —
    rangeStr = sprintf('%s%d:%s%d', firstCol, excelTargetRow, lastCol, excelTargetRow);
    rowData  = readmatrix(fPath, 'Range', rangeStr);

    % readmatrix 可能把空格读成 NaN；如果返回空，保持 NaN 占位
    if ~isempty(rowData)
        % 若实际列数 <100，后面已是 NaN；若 >100，则 readmatrix 已截断
        iterVals(1:numel(rowData), colIdx) = rowData(:);
    end

    varNames{colIdx} = sprintf('Track%d', X);
    fprintf('✓ 已提取 Track%-2d → 列 %d\n', X, colIdx);
end

%% ========== 3. 生成 table 并写 Excel ==========
T_out = array2table(iterVals, 'VariableNames', varNames);

try                       % 新版 MATLAB
    writetable(T_out, outFile, 'WriteVariableNames', true);
catch                     % 旧版回退 xlswrite
    warning('writetable 不可用，回退到 xlswrite。');
    xlswrite(outFile, varNames, 1, 'A1');
    xlswrite(outFile, iterVals, 1, 'A2');
end

fprintf('\n🎉 全部完成！文件已保存到：\n%s\n', outFile);
