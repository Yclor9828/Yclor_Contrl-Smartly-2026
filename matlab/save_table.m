function path = save_table(rows, filename)
%SAVE_TABLE 把结构体数组写成 CSV（UTF-8），用于跨语言结果核对
if isempty(rows)
    error('save_table:empty', '无数据可写');
end
path = fullfile(results_dir(), filename);
T = struct2table(rows, 'AsArray', true);
try
    writetable(T, path, 'Encoding', 'UTF-8');
catch
    writetable(T, path);
end
end
