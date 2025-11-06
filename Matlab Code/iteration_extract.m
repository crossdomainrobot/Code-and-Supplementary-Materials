%% ========= 1. 基本参数 =========
mainDir    = 'D:\Aresearch\雪车论文\雪车论文2\数据\DOPRO';   % 主目录
numFolders = 33;                                      % Whistler Track1–33
numZZ      = 50;                                      % run1001–run1030 

%% ========= 2. 预分配结果矩阵 =========
countsMatrix = NaN(numZZ, numFolders);                % 30 × 33
rowNames = arrayfun(@(z) sprintf('run10%02d', z), 1:numZZ, 'uni', false);
colNames = arrayfun(@(k) sprintf('Track%d',  k), 1:numFolders, 'uni', false);

%% ========= 3. 逐文件夹、逐 ZZ 统计数字个数 =========
for k = 1:numFolders
    % ① 生成子文件夹路径（如有后缀，按需调整）
    folderName = sprintf('Whistler Track%d2', k);
    folderPath = fullfile(mainDir, folderName);

    if ~isfolder(folderPath)
        warning('❌ 未找到文件夹：%s', folderName);
        continue
    end

    % ② 对每个 ZZ (01–30) 寻找 run10ZZ_iterYY_height.txt
    for zz = 1:numZZ
        pattern  = sprintf('run10%02d_iter*_height*.txt', zz);
        fileList = dir(fullfile(folderPath, pattern));

        if isempty(fileList)
            % 该 ZZ 在此赛道上缺文件 —— 保持 NaN
            fprintf('⚠️  %s 缺少 %s\n', folderName, pattern);
            continue
        end

        % ③ 若有多个 YY，取最新（修改时间最大）
        [~, idx] = max([fileList.datenum]);
        fPath = fullfile(folderPath, fileList(idx).name);

        try
            data = readmatrix(fPath, 'OutputType', 'double');
            nNumbers = sum(~isnan(data(:)));          % 非 NaN 元素个数
            countsMatrix(zz, k) = nNumbers;           % 存入结果矩阵
        catch ME
            warning('⚠️  读取失败: %s\n原因: %s', fPath, ME.message);
        end
    end
end

%% ========= 4. 写入 Excel =========
T = array2table(countsMatrix, ...
                'VariableNames', colNames, ...
                'RowNames',      rowNames);

excelPath = fullfile(mainDir, 'Whistler_Tracks_run10_counts.xlsx');

try
    writetable(T, excelPath, 'WriteRowNames', true);
    fprintf('\n✅   汇总完成，已写入：%s\n', excelPath);
catch ME
    error('❌ 写入 Excel 失败：%s', ME.message);
end
