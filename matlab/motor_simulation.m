%MOTOR_SIMULATION 场景二：直流伺服电机多工况转速跟踪（MATLAB 实现）
%
%   对象: 电枢回路 + 机械转子机电耦合模型
%         La·dia/dt = ua - Ra·ia - Ke·ω ,  J·dω/dt = Kt·ia - B·ω - TL
%   工况: [0, 2.0) s   空载起动 0 -> 100 rad/s
%         [2.0, 3.8) s 突加负载力矩 TL = 0.25 N·m
%         [3.8, 6.0] s 转动惯量增至 2.5 倍 + 二次升速至 130 rad/s
%
%   与旧版本相比的改进：
%     - 参数全部取自 shared/fuzzy_config.json，不再出现"README 与代码参数不一致"；
%     - 对照组扩展为 5 个（无抗饱和 / 抗饱和 / ITAE 最优固定参数 / 模糊默认因子 /
%       模糊寻优因子），其中"ITAE 最优"由 Python 侧等预算寻优给出，两种语言共用；
%     - 指标由 perf_metrics / integration_metrics 统一计算，新增 TV(u) 与饱和占空比。
%
%   运行:  motor_simulation

clear; clc; close all;
cfg = fuzzy_config();
SCN = cfg.scenarios.motor;   % run_motor / evaluate_motor 接收场景配置结构体
tuned = load_tuned();
keys = {'pid_nowindup', 'pid_aw', 'pid_opt', 'fuzzy', 'fuzzy_opt'};
labels_cn = {'无抗饱和 PID', '抗饱和 PID', 'ITAE最优 PID', '模糊自适应PID', '模糊-优化因子'};

fprintf('====================================================================\n');
fprintf(' 直流伺服电机多工况转速跟踪仿真 (MATLAB)\n');
fprintf(' 电机参数: Ra=%.2f Ohm, La=%.3f H, Kt=%.3f, Ke=%.3f, B=%.4f, J0=%.4f\n', ...
        SCN.plant.Ra, SCN.plant.La, SCN.plant.Kt, SCN.plant.Ke, SCN.plant.B, SCN.plant.J0);
fprintf(' 工况: dt=%.3f s, 总时长 %.1f s, 电压限幅 ±%.0f V\n', ...
        SCN.simulation.dt, SCN.simulation.t_end, SCN.plant.u_max);
fprintf('====================================================================\n');

rows = struct([]);
data = struct();
colors = struct();
for i = 1:numel(keys)
    [ctrl, label, col] = build_controller('motor', keys{i}, tuned);
    res = run_motor(ctrl, SCN);
    m = evaluate_motor(res, label);
    data.(keys{i}) = res;
    colors.(keys{i}) = col;
    if isempty(rows), rows = m; else, rows(end + 1) = m; end %#ok<SAGROW>
end

fprintf('\n---------------- 性能指标总表（全部由仿真数据计算） ----------------\n');
fprintf('%-24s %8s %8s %8s %8s %8s %9s\n', '控制器', 'tr/s', 'σ1/%', 'ts1/s', ...
        '跌落/(rad/s)', 'σ2/%', 'ITAE');
for i = 1:numel(rows)
    fprintf('%-24s %8.3f %8.2f %8.3f %8.3f %8.2f %9.1f\n', rows(i).controller, ...
            rows(i).tr_s, rows(i).sigma1_pct, rows(i).ts1_s, rows(i).dip_rad_s, ...
            rows(i).sigma2_pct, rows(i).itae_abs);
end
fprintf('%-24s %8s %8s %8s %8s %9s\n', '控制器', 'ess2/(rad/s)', 'TV(u)', '饱和/%', ...
        '扰动恢复/s', 'IAE');
for i = 1:numel(rows)
    fprintf('%-24s %8.4f %8.1f %8.2f %8.3f %9.1f\n', rows(i).controller, ...
            rows(i).ess2, rows(i).tv, rows(i).sat_duty_pct, ...
            rows(i).dip_recovery_s, rows(i).iae);
end
p = save_table(rows, 'motor_metrics_matlab.csv');
fprintf('\n[OK] 指标 CSV: %s\n', p);

% ---------------- 绘图 ----------------
main_keys = {'pid_nowindup', 'pid_aw', 'pid_opt', 'fuzzy'};
t = data.pid_aw.t;
r = data.pid_aw.r;

fig = figure('Position', [80, 60, 1150, 900], 'Visible', 'off');

subplot(4, 1, 1);
plot(t, r, 'r--', 'LineWidth', 1.5, 'DisplayName', '转速指令'); hold on;
for i = 1:numel(main_keys)
    k = main_keys{i};
    idx = find(strcmp(keys, k), 1);
    plot(t, data.(k).y, 'Color', colors.(k), 'LineWidth', 1.6, ...
         'DisplayName', sprintf('%s (σ1=%.2f%%, σ2=%.2f%%)', labels_cn{idx}, ...
                                rows(idx).sigma1_pct, rows(idx).sigma2_pct));
end
grid on; ylabel('转速 \omega (rad/s)'); xlim([0, SCN.simulation.t_end]);
title('直流伺服电机多工况转速跟踪（含负载突加与 2.5 倍惯量摄动）');
legend('Location', 'southeast', 'FontSize', 8);
xline(SCN.load_torque.t_on, 'k:', sprintf('t=%.1fs 突加负载 %.2f N·m', ...
      SCN.load_torque.t_on, SCN.load_torque.amplitude));
xline(SCN.inertia.t_on, 'm:', sprintf('t=%.1fs J×%.1f + 二次升速', ...
      SCN.inertia.t_on, SCN.inertia.scale));

subplot(4, 1, 2);
for i = 1:numel(main_keys)
    k = main_keys{i};
    plot(t, data.(k).u, 'Color', colors.(k), 'LineWidth', 1.2, ...
         'DisplayName', [labels_cn{strcmp(keys, k)} ' 电压']);
end
grid on; ylabel('电枢电压 (V)'); xlim([0, SCN.simulation.t_end]);
ylim([-40, 40]); legend('Location', 'northeast', 'FontSize', 8);
title('控制器输出电枢电压 u_a(t)');

subplot(4, 1, 3);
% rows 顺序与 keys 一致: 1=pid_nowindup, 2=pid_aw, 3=pid_opt, 4=fuzzy, 5=fuzzy_opt
semilogy(t, max(abs(data.pid_aw.e), 1e-3), 'Color', colors.pid_aw, 'LineWidth', 1.4, ...
         'DisplayName', sprintf('抗饱和 PID (ITAE=%.0f)', rows(2).itae_abs)); hold on;
semilogy(t, max(abs(data.fuzzy.e), 1e-3), 'Color', colors.fuzzy, 'LineWidth', 1.6, ...
         'DisplayName', sprintf('模糊自适应 PID (ITAE=%.0f)', rows(4).itae_abs));
grid on; ylabel('|e| (rad/s, 对数轴)'); xlim([0, SCN.simulation.t_end]);
legend('Location', 'northeast', 'FontSize', 8); title('跟踪误差衰减');

subplot(4, 1, 4);
yyaxis left;
plot(t, data.fuzzy.kp, 'm-', 'LineWidth', 1.5, 'DisplayName', 'K_p(t)'); hold on;
plot(t, data.fuzzy.kd * 100, 'c--', 'LineWidth', 1.4, 'DisplayName', 'K_d(t)×100');
ylabel('K_p / K_d×100');
yyaxis right;
plot(t, data.fuzzy.ki, '-', 'Color', [0.85 0.45 0.1], 'LineWidth', 1.4, ...
     'DisplayName', 'K_i(t)');
ylabel('K_i'); grid on; xlabel('时间 t (s)'); xlim([0, SCN.simulation.t_end]);
legend('Location', 'east', 'FontSize', 8);
title('模糊在线增益整定轨迹（真实仿真数据）');

out_img = fullfile(fileparts(mfilename('fullpath')), 'motor_simulation_result.png');
saveas(fig, out_img);
close(fig);
fprintf('[OK] 结果图已保存: %s\n', out_img);

% 参数自适应轨迹 CSV（便于报告引用）
gp = fullfile(results_dir(), 'motor_gain_trajectories.csv');
writematrix([t, data.fuzzy.kp, data.fuzzy.ki, data.fuzzy.kd], gp);
fprintf('[OK] 增益轨迹 CSV: %s\n', gp);
