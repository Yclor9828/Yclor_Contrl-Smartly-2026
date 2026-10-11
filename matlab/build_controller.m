function [ctrl, label, color] = build_controller(scen_name, key, tuned, varargin)
%BUILD_CONTROLLER 按对照组 key 构造控制器（与 Python controllers.build 一一对应）
%
%   key 取值:
%     'pid_nowindup' 常规固定 PID，无抗积分饱和（复现积分饱和）
%     'pid_aw'       常规固定 PID + 条件积分抗饱和
%     'pid_opt'      常规固定 PID，标称工况 ITAE 寻优参数
%     'fuzzy'        模糊自适应 PID（配置默认缩放因子）
%     'fuzzy_opt'    模糊自适应 PID（寻优缩放因子）
%
%   可选参数: 'anti_windup', 'kd_filter_tau', 'scaling' (struct), 'u_limits'

p = inputParser;
addParameter(p, 'anti_windup', [], @(x) isempty(x) || islogical(x));
addParameter(p, 'kd_filter_tau', 0.0, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'scaling', struct(), @isstruct);
addParameter(p, 'u_limits', []);
parse(p, varargin{:});
o = p.Results;

cfg = fuzzy_config();
C = cfg.scenarios.(scen_name);
P = C.pid;
dt = C.simulation.dt;
umin = C.plant.u_min; umax = C.plant.u_max;
if ~isempty(o.u_limits), umin = o.u_limits(1); umax = o.u_limits(2); end

switch key
    case 'pid_nowindup', kind = 'pid'; aw = false; tk = ''; label = 'PID-无抗饱和';
        color = [0.12 0.47 0.71];
    case 'pid_aw',       kind = 'pid'; aw = true;  tk = ''; label = 'PID-抗饱和';
        color = [1.00 0.50 0.05];
    case 'pid_opt',      kind = 'pid'; aw = true;  tk = 'pid'; label = 'PID-ITAE最优';
        color = [0.58 0.40 0.74];
    case 'fuzzy',        kind = 'fuzzy'; aw = true; tk = ''; label = '模糊自适应PID';
        color = [0.17 0.63 0.17];
    case 'fuzzy_opt',    kind = 'fuzzy'; aw = true; tk = 'fuzzy'; label = '模糊自适应PID-优化因子';
        color = [0.84 0.15 0.16];
    otherwise
        error('build_controller:key', '未知对照组 key: %s', key);
end
if ~isempty(o.anti_windup), aw = o.anti_windup; end

T = struct();
field = [scen_name '_' tk];
if ~isempty(tk) && isfield(tuned, field)
    T = tuned.(field);
end

if strcmp(kind, 'pid')
    kp = getfield_def(T, 'kp', P.kp0);
    ki = getfield_def(T, 'ki', P.ki0);
    kd = getfield_def(T, 'kd', P.kd0);
    ctrl = fixed_pid_controller(kp, ki, kd, umin, umax, dt, ...
        'anti_windup', aw, 'kd_filter_tau', o.kd_filter_tau);
else
    kup = getfield_def(T, 'kup', P.kup);
    kui = getfield_def(T, 'kui', P.kui);
    kud = getfield_def(T, 'kud', P.kud);
    kp0 = getfield_def(T, 'kp0', P.kp0);
    ki0 = getfield_def(T, 'ki0', P.ki0);
    kd0 = getfield_def(T, 'kd0', P.kd0);
    if isfield(o.scaling, 'kup'), kup = o.scaling.kup; end
    if isfield(o.scaling, 'kui'), kui = o.scaling.kui; end
    if isfield(o.scaling, 'kud'), kud = o.scaling.kud; end
    ctrl = fuzzy_pid_controller(kp0, ki0, kd0, P.ke, P.kec, kup, kui, kud, ...
        umin, umax, dt, 'anti_windup', aw, 'kd_filter_tau', o.kd_filter_tau);
end
end

function v = getfield_def(s, name, default)
if isfield(s, name), v = s.(name); else, v = default; end
end
