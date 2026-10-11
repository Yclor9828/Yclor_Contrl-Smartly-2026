function res = run_motor(ctrl, cfg, varargin)
%RUN_MOTOR 直流伺服电机机电耦合模型闭环仿真（与 Python simcore.run_motor 等价）
%
%   res = run_motor(ctrl, cfg)
%   res = run_motor(ctrl, cfg, 'J_scale', 2.5, 'Ra', 1.5, 'load', false, ...
%                   'inertia', false, 'noise_std', 0.2, 'seed', 1, 'ctrl_period', 0.002)
%
%   多工况时序（与共享配置一致）：
%     [0, 2.0) s    空载起动 0 -> 100 rad/s
%     [2.0, 3.8) s  突加负载力矩 TL = 0.25 N·m
%     [3.8, 6.0] s  转动惯量增至 2.5 倍，同时指令二次升速至 130 rad/s

p = inputParser;
addParameter(p, 'J_scale', 1.0);
addParameter(p, 'Ra', NaN); addParameter(p, 'La', NaN);
addParameter(p, 'Kt', NaN); addParameter(p, 'Ke', NaN);
addParameter(p, 'B', NaN); addParameter(p, 'J0', NaN);
addParameter(p, 'noise_std', 0.0);
addParameter(p, 'seed', 0);
addParameter(p, 't_end', NaN);
addParameter(p, 'load', true, @islogical);
addParameter(p, 'inertia', true, @islogical);
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

if isempty(o.ref_segments), segs = cfg.reference.segments; else, segs = o.ref_segments; end
r = segs(1, 2) * ones(n, 1);
for i = 2:size(segs, 1)
    r(t >= segs(i, 1) - 1e-12) = segs(i, 2);
end

P = cfg.plant;
Ra = P.Ra; La = P.La; Kt = P.Kt; Ke = P.Ke; B = P.B; J0 = P.J0;
if ~isnan(o.Ra), Ra = o.Ra; end
if ~isnan(o.La), La = o.La; end
if ~isnan(o.Kt), Kt = o.Kt; end
if ~isnan(o.Ke), Ke = o.Ke; end
if ~isnan(o.B),  B  = o.B;  end
if ~isnan(o.J0), J0 = o.J0; end
u_min = P.u_min; u_max = P.u_max;
if ~isempty(o.u_limits), u_min = o.u_limits(1); u_max = o.u_limits(2); end

if o.load
    TL = zeros(n, 1);
    TL(t >= cfg.load_torque.t_on - 1e-12) = cfg.load_torque.amplitude;
else
    TL = zeros(n, 1);
end
if o.inertia
    Jscale = ones(n, 1);
    Jscale(t >= cfg.inertia.t_on - 1e-12) = cfg.inertia.scale;
else
    Jscale = ones(n, 1);
end
Jscale = Jscale * o.J_scale;

if isnan(o.ctrl_period)
    n_sub = 1;
else
    ctrl.dt = o.ctrl_period;
    n_sub = max(1, round(o.ctrl_period / dt));
end

ctrl.reset();
rng(o.seed, 'twister');
ia = 0; w = 0; w_meas = 0;
Y = zeros(n, 1); YM = zeros(n, 1); U = zeros(n, 1); E = zeros(n, 1); EC = zeros(n, 1);
KP = zeros(n, 1); KI = zeros(n, 1); KD = zeros(n, 1);
prev_e = 0; u = 0;
for k = 1:n
    if mod(k - 1, n_sub) == 0
        w_meas = w + o.noise_std * randn();
        u = ctrl.compute(r(k), w_meas);
    end
    J = J0 * Jscale(k);
    dia = (u - Ra * ia - Ke * w) / La * dt;      % 显式欧拉：两个微分同时由当前状态计算
    dw  = (Kt * ia - B * w - TL(k)) / J * dt;    % （与 Python plants.DCMotorPlant 逐行一致）
    ia = ia + dia;
    w  = w  + dw;

    e = r(k) - w;
    Y(k) = w; YM(k) = w_meas; U(k) = u; E(k) = e;
    if k > 1, EC(k) = (e - prev_e) / dt; end
    prev_e = e;
    KP(k) = ctrl.kp; KI(k) = ctrl.ki; KD(k) = ctrl.kd;
end

res = struct('t', t, 'r', r, 'd', TL, 'jscale', Jscale, 'y', Y, 'ym', YM, 'u', U, ...
             'e', E, 'ec', EC, 'kp', KP, 'ki', KI, 'kd', KD, 'cfg', cfg, ...
             'u_min', u_min, 'u_max', u_max);
end
