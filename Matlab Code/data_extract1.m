%% 批量裁剪 Whistler 轨道数据（修正版：连起始行一起删除）
dataDir = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据处理（收敛性分布）';

for X = 1:33
    fileName = sprintf('Whistler_Track%d2_data.xlsx', X);
    filePath = fullfile(dataDir, fileName);

    if ~isfile(filePath)
        warning('未找到文件: %s', filePath);
        continue
    end

    % 读入第一张工作表（默认 Sheet1）
    T = readtable(filePath, 'PreserveVariableNames', true);

    % —— 计算应保留的最后一行 —— %
    switch X
        case 14      % Excel 第 57 行起删
            maxRow = 55;      % 保留到表格行 55 (= Excel 行 56)
        case 18      % Excel 第 78 行起删
            maxRow = 76;      % 保留到表格行 76 (= Excel 行 77)
        case 29      % Excel 第 63 行起删
            maxRow = 61;      % 保留到表格行 61 (= Excel 行 62)
        otherwise    % Excel 第 45 行起删
            maxRow = 43;      % 保留到表格行 43 (= Excel 行 44)
    end

    % —— 行裁剪 —— %
    if height(T) > maxRow
        T = T(1:maxRow, :);
    end

    % —— 列裁剪 —— %
    maxCol = 100;            % CV 列（第 100 列）保留
    if width(T) > maxCol
        T = T(:, 1:maxCol);
    end

    % —— 写回 —— %
    writetable(T, filePath, 'WriteMode', 'overwritesheet');  % R2022b 起可用
    % 对旧版本 MATLAB，可改用：
    % writetable(T, filePath, 'Sheet', 1, 'WriteMode', 'replacefile');

    fprintf('✓ 已处理文件：%s\n', fileName);
end

disp('全部处理完成！');
