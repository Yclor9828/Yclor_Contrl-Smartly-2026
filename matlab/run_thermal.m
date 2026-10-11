function res = run_thermal(ctrl, cfg, varargin)
%RUN_THERMAL 工业加热炉 FOPDT 对象闭环仿真（与 Python simcore.run_thermal 等价）
%
%   res = run_thermal(ctrl, cfg)
%   res = run_thermal(ctrl, cfg, 'K', 1.2, 'T', 20, 'tau', 4, ...
%                     'noise_std', 0.5, 'seed', 3, 'disturbance', false, ...
%                     'ctrl_period', 1.0, 't_end', 280, 'ref_segments', [0 40; 60 80])
%
%   时间推进顺序与 Python 完全一致：
%     k 时刻控制器读取量测 -> 计算 u -> 对象用 u 积分一步 -> 得到 y(k+1)
%   数值积分：显式欧拉，步长取共享配置中的 dt。

p = inputParser;
addParameter(p, 'K', NaN);
addParameter(p, 'T', NaN);
addParameter(p, 'tau', NaN);
addParameter(p, 'noise_std', 0.0);
addParameter(p, 'seed', 0);
addParameter(p, 't_end', NaN);
addParameter(p, 'disturbance', true, @islogical);
addParameter(p, 'ctrl_period', NaN);
addParameter(p, 'ref_segments', []);
addParameter(p, 'u_limits', []);
parse(p, varargin{:});
o = p.Results;

dt = cfg.simulation.dt;
t_end = cfg.simulation.t_end;
if ~isnan(o.t_end), t_end = o.t_end; end
n = round(t_end / dt) + 1;
t = (0:n-1)' * dt;

if isempty(o.ref_segments)
    segs = cfg.reference.segments;
else
    segs = o.ref_segments;
end
r = segs(1, 2) * ones(n, 1);
for i = 2:size(segs, 1)
    r(t >= segs(i, 1) - 1e-12) = segs(i, 2);
end

P = cfg.plant;
K = P.K; TT = P.T; tau = P.tau;
if ~isnan(o.K), K = o.K; end
if ~isnan(o.T), TT = o.T; end
if ~isnan(o.tau), tau = o.tau; end
y0 = P.y0;
u_min = P.u_min; u_max = P.u_max;
if ~isempty(o.u_limits), u_min = o.u_limits(1); u_max = o.u_limits(2); end

if o.disturbance
    dc = cfg.disturbance;
    d = zeros(n, 1);
    mask = (t >= dc.t_on - 1e-12) & (t < dc.t_off - 1e-12);
    d(mask) = dc.amplitude;
else
    d = zeros(n, 1);
end

% 多速率（零阶保持）：控制器每 ctrl_period 秒动作一次，对象仍按 dt 积分
if isnan(o.ctrl_period)
    n_sub = 1;
else
    ctrl.dt = o.ctrl_period;
    n_sub = max(1, round(o.ctrl_period / dt));
end

% 纯滞后缓冲区。注意长度必须为 round(tau/dt)+1：
% 本循环采用"先移位、再取 u_buf(1)"的写法，缓冲区里除历史控制量外还要容纳本步
% 刚写入的控制量，故需额外一个单元；否则等效滞后会比 tau 少一个采样周期
% （实测：漏掉这个 +1 会使 5.0 s 的纯滞后变成 4.8 s，进而使 MATLAB 与 Python
%   的闭环轨迹出现 ~1.2 ℃ 的系统性偏差——跨语言比对脚本正是靠这一点发现的）。
n_delay = max(1, round(tau / dt)) + 1;
u_buf = zeros(n_delay, 1);
y = y0; y_meas = y0;
ctrl.reset();

rng(o.seed, 'twister');
Y = zeros(n, 1); YM = zeros(n, 1); U = zeros(n, 1); E = zeros(n, 1); EC = zeros(n, 1);
KP = zeros(n, 1); KI = zeros(n, 1); KD = zeros(n, 1);
prev_e = 0; u = 0;
for k = 1:n
    if mod(k - 1, n_sub) == 0
        y_meas = y + o.noise_std * randn();
        u = ctrl.compute(r(k), y_meas);
    end
    u_buf = [u_buf(2:end); u];        % 先移位写入本步控制量
    u_delayed = u_buf(1);             % 再取最老的一个 —— 纯滞后 tau
    y = y + (-(y - y0) + K * u_delayed + d(k)) / TT * dt;

    e = r(k) - y;
    Y(k) = y; YM(k) = y_meas; U(k) = u; E(k) = e;
    if k > 1, EC(k) = (e - prev_e) / dt; end
    prev_e = e;
    KP(k) = ctrl.kp; KI(k) = ctrl.ki; KD(k) = ctrl.kd;
end

res = struct('t', t, 'r', r, 'd', d, 'y', Y, 'ym', YM, 'u', U, 'e', E, 'ec', EC, ...
             'kp', KP, 'ki', KI, 'kd', KD, 'cfg', cfg, 'u_min', u_min, 'u_max', u_max);
end
