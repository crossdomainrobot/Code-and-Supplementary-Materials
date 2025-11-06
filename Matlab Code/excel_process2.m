%% ============= 用户可修改的参数 =============
dataDir = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据处理（收敛性分布）';   % 数据目录
outFile = fullfile(dataDir, 'Whistler_LastValues_100x33.xlsx');        % 输出文件
%% ============================================

nFiles = 33;           % 文件数量 (Track1~Track33)
nCols  = 100;          % 每个文件应有的列数 (A~CV)

lastVals = nan(nCols, nFiles);    % 预分配结果矩阵
varNames = strings(1, nFiles);    % 存放列标题

%% -------- 主循环：提取每列最后一个有效数 --------
for X = 1:nFiles
    fileName = sprintf('Whistler_Track%d2_data.xlsx', X);
    filePath = fullfile(dataDir, fileName);

    if ~isfile(filePath)
        warning('文件缺失: %s —— 跳过！', filePath);
        continue
    end

    T = readtable(filePath, 'PreserveVariableNames', true);

    for c = 1:min(nCols, width(T))
        colData = T{:, c};

        % 兼容数值/字符/混合列
        if isnumeric(colData) || islogical(colData)
            idx = find(~isnan(colData), 1, 'last');
        else
            numData = str2double(colData);   % 空字符 => NaN
            idx = find(~isnan(numData), 1, 'last');
            colData = numData;               % 便于取值
        end

        if ~isempty(idx)
            lastVals(c, X) = colData(idx);
        end
    end

    varNames(X) = sprintf('Track%d', X);
    fprintf('✓ 已提取 %s\n', fileName);
end

%% -------- 写入汇总 Excel --------
% 将 string → char 元胞数组，避免 “不支持类型 'string'” 报错
headerCell = cellstr(varNames)';   % 1×33 cell，转置成行向量

try
    writecell(headerCell, outFile, 'Sheet', 1, 'Range', 'A1');
catch ME
    % 旧版 MATLAB (无 writecell) 或依然报错时，用 xlswrite 兜底
    if contains(ME.message, 'writecell') || contains(ME.message, 'No method')
        xlswrite(outFile, headerCell, 1, 'A1');
    else
        rethrow(ME);
    end
end

% 写入 100×33 数据块，从 A2 开始
try
    writematrix(lastVals, outFile, 'Sheet', 1, 'Range', 'A2');
catch
    % 同样兜底到 xlswrite（会自动追加）
    xlswrite(outFile, lastVals, 1, 'A2');
end

fprintf('\n🎉 汇总完成！结果保存在：\n%s\n', outFile);