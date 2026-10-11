function m = perf_metrics(t, y, r, t0, t1, y_base, scale, band, closed_end)
%PERF_METRICS 窗口时域性能指标（与 Python metrics.response_metrics 完全同定义）
%
%   m = perf_metrics(t, y, r, t0, t1)
%   m = perf_metrics(t, y, r, t0, t1, y_base, scale, band, closed_end)
%
%   y_base     : 阶跃前初值；给 NaN 时不计算上升时间/超调量
%   scale      : 归一化用"设定增量"；给 NaN 时取 |r_end - y_base|
%   band       : 误差带宽度（相对 scale），默认 0.02（±2%）
%   closed_end : true 时窗口包含 t1 样本（末段工况用）
%
%   返回字段与 Python 侧一一对应，便于跨语言核对 CSV。

if nargin < 6 || isempty(y_base), y_base = NaN; end
if nargin < 7 || isempty(scale), scale = NaN; end
if nargin < 8 || isempty(band), band = 0.02; end
if nargin < 9 || isempty(closed_end), closed_end = false; end

% 半开窗口 [t0, t1)：设定值在 t1 处跳变，t1 样本属于下一阶段
if closed_end
    mask = (t >= t0 - 1e-9) & (t <= t1 + 1e-9);
else
    mask = (t >= t0 - 1e-9) & (t < t1 - 1e-9);
end
tt = t(mask); yy = y(mask); rr = r(mask);
if numel(tt) < 3
    error('perf_metrics:window', '窗口 [%g, %g] 内样本不足', t0, t1);
end

r_ss = rr(end);
if isnan(y_base), y_base_eff = yy(1); else, y_base_eff = y_base; end
if isnan(scale), scale = abs(r_ss - y_base_eff); end
if abs(scale) < 1e-12, scale = 1; end

err = rr - yy;
sgn = 1; if r_ss < y_base_eff, sgn = -1; end

tail = tt >= tt(end) - 0.1 * (tt(end) - tt(1));
[~, i_pk] = max(sgn * yy);
[~, i_dev] = max(abs(err));

m = struct();
m.r_final   = r_ss;
m.y_final   = yy(end);
m.peak      = yy(i_pk);
m.t_peak    = tt(i_pk) - t0;
m.overshoot = max(0, sgn * (m.peak - r_ss) / scale * 100);
m.ess       = mean(err(tail));
m.tail_std  = std(yy(tail));
m.max_dev   = max(abs(err));
m.t_max_dev = tt(i_dev) - t0;
m.iae       = trapz(tt, abs(err));
m.ise       = trapz(tt, err .^ 2);
m.itae      = trapz(tt, (tt - t0) .* abs(err));
m.rmse      = sqrt(mean(err .^ 2));

if isnan(y_base)
    m.rise_time = NaN;
else
    i10 = find(sgn * yy >= sgn * (y_base_eff + 0.1 * sgn * scale), 1, 'first');
    i90 = find(sgn * yy >= sgn * (y_base_eff + 0.9 * sgn * scale), 1, 'first');
    if isempty(i10) || isempty(i90)
        m.rise_time = NaN;
    else
        m.rise_time = tt(i90) - tt(i10);
    end
end

tol = band * scale;
outside = abs(err) > tol;
if any(outside)
    i_last = find(outside, 1, 'last');
    m.settling_time = tt(i_last) - t0;
else
    m.settling_time = 0;
end
m.recovery_time = m.settling_time;
end
