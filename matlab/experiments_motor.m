%EXPERIMENTS_MOTOR 场景二拓展实验（MATLAB 实现）
%
%   E1M 机电参数摄动蒙特卡洛鲁棒性：J0、Kt、Ke、Ra 同时 ±30% 随机摄动
%   E2M 控制器采样周期（1 ms / 2 ms / 5 ms / 10 ms）对控制品质的影响
%   E3M 转速量测噪声与微分滤波对执行机构抖振的影响
%
%   运行:  experiments_motor
%   结果:  results/exp*_motor_*.csv  +  results/figures/E1M~E3M*.png

clear; clc; close all;
cfg = fuzzy_config();
SCN = cfg.scenarios.motor;   % run_motor / evaluate_motor 接收场景配置结构体
tuned = load_tuned();
figdir = fullfile(results_dir(), 'figures');
if ~isfolder(figdir), mkdir(figdir); end

keys = {'pid_opt', 'fuzzy', 'fuzzy_opt'};
nk = numel(keys);
labels = cell(1, nk);
colors = zeros(nk, 3);
for j = 1:nk
    [~, labels{j}, colors(j, :)] = build_controller('motor', keys{j}, tuned);
end

% =========================================================================
% E1M 参数摄动蒙特卡洛
% =========================================================================
n = 150; mismatch = 0.30; seed = 20240923;
rng(seed, 'twister');
fprintf('====================================================================\n');
fprintf(' E1M 机电参数摄动蒙特卡洛：J0/Kt/Ke/Ra 各自独立 ±%.0f%%，%d 组工况，控制器不重调\n', ...
        mismatch * 100, n);
fprintf('====================================================================\n');

ITAE = zeros(n, nk); SIG = zeros(n, nk); DIP = zeros(n, nk); TS = zeros(n, nk);
rows = struct([]);
for r = 1:n
    f = 1 + (rand(1, 4) * 2 - 1) * mismatch;
    for j = 1:nk
        [ctrl, label] = build_controller('motor', keys{j}, tuned);
        res = run_motor(ctrl, SCN, 'J0', SCN.plant.J0 * f(1), 'Kt', SCN.plant.Kt * f(2), ...
                        'Ke', SCN.plant.Ke * f(3), 'Ra', SCN.plant.Ra * f(4), 'seed', r);
        m = evaluate_motor(res, label);
        ITAE(r, j) = m.itae_abs; SIG(r, j) = m.sigma1_pct;
        DIP(r, j) = m.dip_rad_s; TS(r, j) = m.ts1_s;
        row = m; row.run = r; row.J0 = SCN.plant.J0 * f(1); row.Kt = SCN.plant.Kt * f(2);
        row.Ke = SCN.plant.Ke * f(3); row.Ra = SCN.plant.Ra * f(4); row.key = keys{j};
        if isempty(rows), rows = row; else, rows(end + 1) = row; end %#ok<SAGROW>
    end
    if mod(r, 25) == 0, fprintf('  ... 已完成 %d/%d 组\n', r, n); end
end
save_table(rows, 'exp1_motor_monte_carlo_raw.csv');

fprintf('\n%-24s %10s %10s %10s %10s %10s\n', '控制器', 'ITAE均值', 'ITAE最坏', ...
        'σ1均值/%', 'σ1最坏/%', '跌落均值');
summ = struct([]);
for j = 1:nk
    s = struct('controller', labels{j}, 'itae_mean', mean(ITAE(:, j)), ...
               'itae_worst', max(ITAE(:, j)), 'itae_std', std(ITAE(:, j)), ...
               'sigma_mean', mean(SIG(:, j)), 'sigma_worst', max(SIG(:, j)), ...
               'dip_mean', mean(DIP(:, j)), 'dip_worst', max(DIP(:, j)), ...
               'ts_mean', mean(TS(:, j)));
    if isempty(summ), summ = s; else, summ(end + 1) = s; end %#ok<SAGROW>
    fprintf('%-24s %10.2f %10.2f %10.2f %10.2f %10.3f\n', s.controller, s.itae_mean, ...
            s.itae_worst, s.sigma_mean, s.sigma_worst, s.dip_mean);
end
save_table(summ, 'exp1_motor_monte_carlo_summary.csv');

imp = (ITAE(:, 1) - ITAE(:, nk)) ./ abs(ITAE(:, 1)) * 100;
fprintf('\n配对比较 (%s vs %s): ITAE 改善均值 %.2f%%，中位数 %.2f%%，胜率 %.1f%%\n', ...
        labels{nk}, labels{1}, mean(imp), median(imp), mean(imp > 0) * 100);

fig = figure('Position', [80, 80, 1250, 480], 'Visible', 'off');
subplot(1, 3, 1);
boxplot(ITAE, 'Labels', labels, 'Symbol', '');
ylabel('全程 ITAE'); grid on; title('(a) 参数摄动下 ITAE 分布');
subplot(1, 3, 2);
boxplot(SIG, 'Labels', labels, 'Symbol', '');
ylabel('初始超调 σ1 (%)'); grid on; title('(b) 超调量分布');
subplot(1, 3, 3);
scatter(ITAE(:, 1), ITAE(:, nk), 16, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
lim = [min([ITAE(:, 1); ITAE(:, nk)]) * 0.9, max([ITAE(:, 1); ITAE(:, nk)]) * 1.1];
plot(lim, lim, 'k--'); grid on;
xlabel(labels{1}); ylabel(labels{nk}); title('(c) 配对散点（对角线下方为优势区）');
sgtitle(sprintf('E1M 电机参数 ±%.0f%% 摄动蒙特卡洛（%d 组工况）', mismatch * 100, n));
saveas(fig, fullfile(figdir, 'E1M_motor_monte_carlo.png')); close(fig);

% =========================================================================
% E2M 采样周期
% =========================================================================
fprintf('\n====================================================================\n');
fprintf(' E2M 控制器采样周期实验（对象仍按 %.3f s 积分，控制器零阶保持）\n', SCN.simulation.dt);
fprintf('====================================================================\n');
Ts_list = [0.001, 0.002, 0.005, 0.010];
rows2 = struct([]);
ITAE2 = zeros(nk, numel(Ts_list)); SIG2 = ITAE2; TV2 = ITAE2; TS2 = ITAE2;
for j = 1:nk
    for i = 1:numel(Ts_list)
        [ctrl, label] = build_controller('motor', keys{j}, tuned);
        res = run_motor(ctrl, SCN, 'ctrl_period', Ts_list(i));
        m = evaluate_motor(res, label);
        m.Ts = Ts_list(i); m.key = keys{j};
        ITAE2(j, i) = m.itae_abs; SIG2(j, i) = m.sigma1_pct;
        TV2(j, i) = m.tv; TS2(j, i) = m.ts1_s;
        if isempty(rows2), rows2 = m; else, rows2(end + 1) = m; end %#ok<SAGROW>
        fprintf(' %-24s Ts=%.3f s  σ1=%6.2f%%  ts=%6.3f s  ITAE=%8.2f  TV=%9.1f  饱和=%.1f%%\n', ...
                label, m.Ts, m.sigma1_pct, m.ts1_s, m.itae_abs, m.tv, m.sat_duty_pct);
    end
end
save_table(rows2, 'exp2_motor_control_period.csv');

fig = figure('Position', [80, 80, 1250, 420], 'Visible', 'off');
titles2 = {'(a) 跟踪精度', '(b) 超调量', '(c) 执行机构动作频度'};
ylabs2 = {'ITAE', 'σ1 (%)', 'TV(u)'};
datasets = {ITAE2, SIG2, TV2};
markers = {'o-', 's-', '^-'};
for a = 1:3
    subplot(1, 3, a);
    for j = 1:nk
        semilogx(Ts_list * 1000, datasets{a}(j, :), markers{a}, 'Color', colors(j, :), ...
                 'LineWidth', 1.6, 'DisplayName', labels{j});
        hold on;
    end
    grid on; xlabel('控制器采样周期 Ts (ms)'); ylabel(ylabs2{a});
    title(titles2{a});
    if a == 1, legend('Location', 'northwest', 'FontSize', 8); end
end
sgtitle('E2M 控制器采样周期对电机调速品质的影响');
saveas(fig, fullfile(figdir, 'E2M_motor_control_period.png')); close(fig);

% =========================================================================
% E3M 量测噪声与微分滤波
% =========================================================================
fprintf('\n====================================================================\n');
fprintf(' E3M 转速量测噪声与微分滤波实验\n');
fprintf('====================================================================\n');
noises = [0, 0.5, 1.0, 2.0];
taus = [0.0, 0.005];
rows3 = struct([]);
TV3 = zeros(nk, numel(noises), numel(taus));
IT3 = TV3; STB3 = TV3;
for j = 1:nk
    for a = 1:numel(noises)
        for b = 1:numel(taus)
            [ctrl, label] = build_controller('motor', keys{j}, tuned, ...
                                             'kd_filter_tau', taus(b));
            res = run_motor(ctrl, SCN, 'noise_std', noises(a), 'seed', 11);
            m = evaluate_motor(res, label);
            m.noise_std = noises(a); m.filter_tau = taus(b); m.key = keys{j};
            TV3(j, a, b) = m.tv; IT3(j, a, b) = m.itae_abs; STB3(j, a, b) = m.tail_std2;
            if isempty(rows3), rows3 = m; else, rows3(end + 1) = m; end %#ok<SAGROW>
            fprintf(' %-24s σn=%4.1f rad/s  Tf=%.3f s  TV=%9.1f  稳态波动=%8.4f  ITAE=%8.2f\n', ...
                    label, m.noise_std, m.filter_tau, m.tv, m.tail_std2, m.itae_abs);
        end
    end
end
save_table(rows3, 'exp3_motor_noise_filter.csv');

fig = figure('Position', [80, 80, 1250, 430], 'Visible', 'off');
styles = {'o-', 's--'};
tagnames = {'理想微分', '滤波微分'};
for a = 1:2
    subplot(1, 2, a);
    for j = 1:nk
        for b = 1:numel(taus)
            if a == 1, yv = squeeze(TV3(j, :, b)); else, yv = squeeze(IT3(j, :, b)); end
            plot(noises, yv, styles{b}, 'Color', colors(j, :), 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('%s %s', labels{j}, tagnames{b}));
            hold on;
        end
    end
    grid on; xlabel('量测噪声 σn (rad/s)');
    if a == 1, ylabel('TV(u)'); title('(a) 执行机构动作频度');
    else, ylabel('ITAE'); title('(b) 跟踪精度'); end
    legend('Location', 'northwest', 'FontSize', 7);
end
sgtitle('E3M 量测噪声与微分滤波对电机调速的影响');
saveas(fig, fullfile(figdir, 'E3M_motor_noise_filter.png')); close(fig);

fprintf('\n[完成] 电机拓展实验结束，CSV 与图位于 %s\n', results_dir());
