function d = results_dir()
%RESULTS_DIR 返回仓库根目录下 results/ 的绝对路径（不存在则创建）
here = fileparts(mfilename('fullpath'));
d = fullfile(fileparts(here), 'results');
if ~isfolder(d)
    mkdir(d);
end
end
