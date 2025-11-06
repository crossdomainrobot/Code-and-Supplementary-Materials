%% ========== 路径与文件名 ==========
dataDir   = 'D:\Aresearch\雪车论文\雪车论文2\数据\数据处理（收敛性分布）';  % 数据目录
inputFile = fullfile(dataDir, 'Whistler_LastValues_100x33.xlsx');        % 原文件
outputFile = fullfile(dataDir, 'Whistler_LastValues_100x33_flipped.xlsx'); % 输出文件
%% ==================================

% ---------- 1. 读入整张工作表 ----------
% 保留原列名，防止被 MATLAB 自动改名
T = readtable(inputFile, 'PreserveVariableNames', true);

% ---------- 2. 左右倒置列顺序 ----------
T_flipped = T(:, end:-1:1);

% ---------- 3. 写回 Excel ----------
try
    % R2019b 及更新版本
    writetable(T_flipped, outputFile, 'WriteMode', 'overwritefile');
catch
    % 旧版 MATLAB 无 writetable 或 WriteMode 参数：回退到 xlswrite
    warning('writetable 不可用，已回退到 xlswrite（旧版 Excel 写入接口）。');
    % 把 table 转成 cell 再写
    header   = T_flipped.Properties.VariableNames;
    dataBody = table2array(T_flipped);
    xlswrite(outputFile, header,   1, 'A1');
    xlswrite(outputFile, dataBody, 1, 'A2');
end

fprintf('🎉 处理完成！已生成左右倒置后的文件：\n%s\n', outputFile);
