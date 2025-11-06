% 设置主目录路径
mainDir = 'D:\Aresearch\雪车论文\雪车论文2\数据';

% 文件夹数量（Whistler Track1~33）
numFolders = 33;

for folderIdx = 1:numFolders
    % 构建当前文件夹路径
    folderName = sprintf('Whistler Track%d2_comparision', folderIdx);
    folderPath = fullfile(mainDir, folderName);
    % 获取所有以 _height.txt 结尾的文件
    fileList = dir(fullfile(folderPath, '*_height.txt'));

    if isempty(fileList)
        fprintf('⚠️ 文件夹 %s 中没有找到 TXT 文件。\n', folderName);
        continue;
    end

    % 初始化数据 cell 和文件名 cell
    dataCell = cell(1, numel(fileList));
    fileNames = cell(1, numel(fileList));
    maxLen = 0;

    % 遍历每个文件，读取数据
    for i = 1:numel(fileList)
        fileName = fileList(i).name;
        filePath = fullfile(folderPath, fileName);

        try
            colData = readmatrix(filePath);  % 读取单列数据
        catch
            fprintf('❌ 无法读取文件: %s\n', filePath);
            continue;
        end

        if isvector(colData)
            colData = colData(:);  % 强制转为列向量
        else
            warning('⚠️ 文件 %s 中包含多列，仅取第一列数据。', fileName);
            colData = colData(:, 1);
        end

        dataCell{i} = colData;
        fileNames{i} = fileName;
        maxLen = max(maxLen, length(colData));
    end

    % 补齐每列数据为等长，NaN填充
    paddedMatrix = NaN(maxLen, numel(dataCell));
    for i = 1:numel(dataCell)
        col = dataCell{i};
        paddedMatrix(1:length(col), i) = col;
    end

    % 写入 Excel
    excelName = sprintf('Whistler_Track%d2_comparision_data.xlsx', folderIdx);
    excelPath = fullfile(mainDir, excelName);

    % 使用 table 写入，方便添加列标题
    T = array2table(paddedMatrix, 'VariableNames', matlab.lang.makeValidName(fileNames));

    try
        writetable(T, excelPath);
        fprintf('✅ 完成文件夹 %s 的整理，输出至 %s\n', folderName, excelName);
    catch ME
        fprintf('❌ 写入 Excel 文件失败: %s\n原因: %s\n', excelName, ME.message);
    end
end

disp('✅ 所有文件夹数据整理完成。');
