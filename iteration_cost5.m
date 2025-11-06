% 设置文件夹路径
clc,clear
folderPath = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据处理（收敛性分布）';

% 初始化结果矩阵，100 行（变量），33 列（文件）
resultMatrix = zeros(100, 33);

% 遍历33个文件
for fileIndex = 1:33
    % 构造文件名
    fileName = sprintf('Whistler_Track%d2_data.xlsx', fileIndex);
    fullPath = fullfile(folderPath, fileName);
    
    % 读取数据（跳过表头）
    [~, ~, raw] = xlsread(fullPath);
    data = raw(2:end, 1:100);  % 第二行到最后，取前100列

    % 转换为数值矩阵（有可能混有空或非数值）
    numericData = nan(size(data)); % 初始化为 NaN
    for col = 1:100
        for row = 1:size(data,1)
            if isnumeric(data{row,col})
                numericData(row,col) = data{row,col};
            elseif ischar(data{row,col})
                num = str2double(data{row,col});
                if ~isnan(num)
                    numericData(row,col) = num;
                end
            end
        end
    end

    % 对每列提取第九个值（第10行），或最后一个非NaN值
    colValues = zeros(100, 1);
    for col = 1:100
        colData = numericData(:, col);
        validData = colData(~isnan(colData));
        if length(validData) >= 3
            colValues(col) = validData(3);
        elseif ~isempty(validData)
            colValues(col) = validData(end);
        else
            colValues(col) = NaN;  % 若整列无有效数据
        end
    end

    % 存入结果矩阵
    resultMatrix(:, fileIndex) = colValues;
end

% 保存结果到 Excel 文件，每列对应一个文件，每行为一个变量
outputFile = fullfile(folderPath, 'Whistler_LastValues_1_-15.xlsx');
xlswrite(outputFile, resultMatrix);
