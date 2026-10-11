function m = integration_metrics(t, y, r, u, u_min, u_max)
%INTEGRATION_METRICS 全程误差积分指标与执行机构代价指标
%
%   与 Python metrics.whole_run_metrics 完全同定义：
%     itae_abs = ∫ t·|e| dt  （采用绝对时间，与工业惯例一致）
%     iae/ise  = ∫|e|dt / ∫e²dt
%     tv       = Σ|u(k)-u(k-1)|  （控制量总变差，衡量执行器动作频度）
%     sat_duty = 控制量处于限幅边界的时间比例 (%)

err = r(:) - y(:);
t = t(:); u = u(:);
m = struct();
m.itae_abs = trapz(t, t .* abs(err));
m.iae = trapz(t, abs(err));
m.ise = trapz(t, err .^ 2);
m.max_abs_err = max(abs(err));

du = abs(diff(u));
span = max(abs(u_max - u_min), 1e-12);
at_limit = (abs(u - u_min) < 1e-6 * span) | (abs(u - u_max) < 1e-6 * span);
m.u_mean = mean(u);
m.u_std = std(u);
if isempty(du), m.tv = 0; m.max_du = 0; else, m.tv = sum(du); m.max_du = max(du); end
m.sat_duty_pct = mean(at_limit) * 100;
end
