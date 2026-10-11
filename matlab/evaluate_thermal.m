function m = evaluate_thermal(res, label)
%EVALUATE_THERMAL 加热炉仿真结果 -> 标准指标结构体（与 Python evaluate_thermal 一致）

if nargin < 2, label = ''; end
t = res.t; y = res.y; r = res.r; u = res.u;
cfg = res.cfg;
ph = cfg.phases;
p1 = ph.step1; p2 = ph.dist; p3 = ph.step2;
y0 = cfg.plant.y0;
span1 = cfg.reference.segments(1, 2) - y0;

m1 = perf_metrics(t, y, r, p1(1), p1(2), y0, span1);
m2 = perf_metrics(t, y, r, p2(1), p2(2), NaN, span1);
y_b2 = interp1(t, y, p3(1));
span2 = cfg.reference.segments(end, 2) - y_b2;
m3 = perf_metrics(t, y, r, p3(1), p3(2), y_b2, span2, [], true);
allm = integration_metrics(t, y, r, u, res.u_min, res.u_max);

m = struct();
m.controller = label;
m.tr_s = m1.rise_time;
m.sigma1_pct = m1.overshoot;
m.ts1_s = m1.settling_time;
m.ess1 = m1.ess;
m.peak1 = m1.peak;
m.dist_drop = m2.max_dev;
m.dist_recovery_s = m2.recovery_time;
m.dist_iae = m2.iae;
m.sigma2_pct = m3.overshoot;
m.ts2_s = m3.settling_time;
m.ess2 = m3.ess;
m.tail_std2 = m3.tail_std;
m.itae_abs = allm.itae_abs;
m.iae = allm.iae;
m.tv = allm.tv;
m.sat_duty_pct = allm.sat_duty_pct;
end
