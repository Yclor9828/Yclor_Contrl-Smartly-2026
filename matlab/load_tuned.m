function t = load_tuned()
%LOAD_TUNED 读取 Python 侧数值寻优得到的"最优对照组"参数 (results/tuned_params.json)
%
%   结构化整定结果由 python/tuning.py 产生，MATLAB 侧直接复用，
%   从而保证"最优固定 PID"对照组在两种语言中完全相同。
%   若文件不存在，返回空结构体，调用方将退化为配置默认参数。

p = fullfile(results_dir(), 'tuned_params.json');
if isfile(p)
    t = jsondecode(fileread(p));
else
    t = struct();
    warning('未找到 %s，pid_opt / fuzzy_opt 将退化为配置默认参数。', p);
end
end
