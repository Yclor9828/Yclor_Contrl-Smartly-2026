%VERIFY_CONSISTENCY 跨语言一致性核查（MATLAB 侧）
%
%   在标准论域的 61×61 网格上评估模糊推理系统，写出"金标准网格"CSV：
%     results/fis_grid_matlab.csv           由 MATLAB 类（纯代码推理）生成
%     results/fis_grid_official_matlab.csv  由官方 readfis + evalfis 生成（若装有工具箱）
%   随后 python/verify_consistency.py 会读取这些 CSV 与 Python 实现逐点比对，
%   并检查 .fis 文件与代码实现是否等价。
%
%   运行:  verify_consistency

clear; clc;
here = fileparts(mfilename('fullpath'));
cfg = fuzzy_config();
univ = cfg.fuzzy.universe;
n = 61;
axis_ = linspace(univ(1), univ(2), n);

% ---------------- 1. 纯代码实现（MATLAB 类） ----------------
P = cfg.scenarios.thermal.pid;
c = fuzzy_pid_controller(P.kp0, P.ki0, P.kd0, 1, 1, 1, 1, 1, 0, 100, 1);
M = zeros(n * n, 5);
idx = 0;
for i = 1:n
    for j = 1:n
        idx = idx + 1;
        [a, b, d] = c.infer_point(axis_(i), axis_(j));
        M(idx, :) = [axis_(i), axis_(j), a, b, d];
    end
end
p1 = fullfile(results_dir(), 'fis_grid_matlab.csv');
writematrix(M, p1);
fprintf('[OK] MATLAB 类推理网格已写出: %s (%d 点)\n', p1, n * n);

% ---------------- 2. 官方 .fis 推理（若可用） ----------------
fisfile = fullfile(here, 'fuzzy_pid.fis');
p2 = fullfile(results_dir(), 'fis_grid_official_matlab.csv');
if ~isfile(fisfile)
    fprintf('[跳过] 未找到 %s，请先运行 generate_fis。\n', fisfile);
elseif exist('readfis', 'file') ~= 2
    fprintf('[跳过] 本机未安装 Fuzzy Logic Toolbox，无法用官方推理核校验 .fis。\n');
else
    fis = readfis(fisfile);
    G = zeros(n * n, 5);
    idx = 0;
    for i = 1:n
        for j = 1:n
            idx = idx + 1;
            out = evalfis([axis_(i), axis_(j)], fis);
            G(idx, :) = [axis_(i), axis_(j), out(1), out(2), out(3)];
        end
    end
    writematrix(G, p2);
    d = max(abs(M(:, 3:5) - G(:, 3:5)), [], 'all');
    fprintf('[OK] 官方 .fis 推理网格已写出: %s\n', p2);
    fprintf('     MATLAB 类 与 .fis(官方推理) 最大偏差 = %.3e\n', d);
    if d < 1e-6
        fprintf('     [通过] .fis 与代码实现数值等价，Simulink 结果可与脚本结果相互印证。\n');
    else
        fprintf('     [警告] 偏差过大：请检查 .fis 的隶属度函数是否为肩形 trapmf。\n');
    end
end

% ---------------- 3. 规则库自检（点对称性 / 取值范围） ----------------
rules = cfg.fuzzy.rules;
names = fieldnames(rules);
fprintf('\n---- 规则库自检 ----\n');
for k = 1:numel(names)
    R = double(rules.(names{k}));
    ok_shape = isequal(size(R), [7 7]);
    ok_range = all(abs(R(:)) <= 3);
    ok_sym = isequal(R, flip(flip(R, 1), 2));
    fprintf(' %-4s 尺寸7x7=%d  取值∈[-3,3]=%d  点对称=%d\n', names{k}, ok_shape, ok_range, ok_sym);
end
fprintf('\n[提示] 运行 python python/verify_consistency.py 完成 Python 侧比对。\n');
