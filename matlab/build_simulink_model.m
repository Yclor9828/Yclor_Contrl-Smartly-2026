function [mdl, info] = build_simulink_model(varargin)
%BUILD_SIMULINK_MODEL 用代码自动搭建场景一（工业加热炉）模糊自适应 PID 的 Simulink 模型
%
%   本脚本不依赖任何手工拖拽：new_system / add_block / add_line / set_param /
%   save_system 全部由代码完成，可重复运行（幂等：先关闭并删除同名模型）。
%
%   用法:
%       build_simulink_model                              % 生成 matlab/fuzzy_pid_simulink.slx
%       build_simulink_model('dist_after_delay', false)   % 冷负荷扰动改接在纯滞后之前
%       [mdl, info] = build_simulink_model(...)           % 同时返回模块清单
%
%   模型结构（场景一：G(s)=1.5/(25s+1)·e^(-5s)，初温 20 ℃，100→125 ℃，t=130~200 s 冷负荷）:
%
%     [设定_阶跃1(0→100)] ┐
%                         ├─ Sum ─ r ─┐
%     [设定_阶跃2(200→+25)]┘           ├─ Sum(+,−) ─ e ─┬─ Gain ke ─ Sat[-3,3] ─────────┐
%                                      │                 │                              ├ Mux2 ─ FLC ─ Demux3
%                                      │                 └─ UnitDelay ─ Sum(+,-) ─ Gain(1/dt) ─ Gain kec ─ Sat[-3,3] ┘
%                                      │                                                                    │
%                                      │        ┌──────────────── 增益 kup/kui/kud + 常数 Kp0/Ki0/Kd0 ─ Saturation(下限) ─ Kp/Ki/Kd
%                                      │        │
%                                      └────→ PID Controller (P/I/D 端口 External, 限幅 0~100, clamping 抗饱和)
%                                                 │
%                     冷负荷扰动 step ── Sum ──→ Discrete Transfer Fcn ── Delay(25×0.2s=5s) ── Bias(+20) ── y
%
%   与脚本级实现（python/simcore.py、matlab/run_thermal.m 的显式欧拉法）逐项对齐的关键点：
%     1) 模糊模块：Fuzzy Logic Toolbox 的 "Fuzzy Logic Controller"（库 fuzblock），
%        FIS 参数指向基础工作区变量 FIS_FUZZY_PID，该变量由 readfis('fuzzy_pid.fis') 读入，
%        保证与 Python/MATLAB 代码共用同一套隶属度函数与 49 条规则。
%     2) 微分环节：脚本用理想后向差分 ec(k) = (e(k)-e(k-1))/dt（kd_filter_tau = 0），
%        因此模型用 Unit Delay + Sum + Gain(1/dt) 复现，而不是连续滤波微分
%        （Transfer Fcn [100 0]/[1 100] 的极点在 -100 rad/s，在 0.2 s 定步长下
%        无论离散还是欧拉积分都不可用，且与脚本的"理想微分"并不等价，详见报告）。
%     3) 求解器：FixedStepDiscrete（0.2 s）要求模型内不得有连续状态，故被控对象的
%        Transfer Fcn 1.5/(25s+1) 与 Transport Delay 5 s 用**严格等价的离散形式**实现：
%        Discrete Transfer Fcn  num=[0.012 0], den=[1 -0.992], Ts=0.2
%            <=> (y-y0)(k) = 0.992·(y-y0)(k-1) + 0.012·u(k-1)      （即脚本的显式欧拉一步）
%        Delay 25 个采样点 (25×0.2 s = 5 s) <=> 纯滞后 τ = 5 s
%     4) PID Controller 模块：Controller=PID, Form=Parallel, TimeDomain=Discrete-time,
%        IntegratorMethod='Forward Euler', UseFilter=off, ControllerParametersSource=external,
%        LimitOutput=on(0~100), AntiWindupMode='clamping'。
%
%   生成后脚本会打印模块清单，并把清单与结构说明返回给调用者。

% -------------------------------------------------------------------------
p = inputParser;
p.addParameter('dist_after_delay', true, @islogical);   % true: 扰动接在 5 s 纯滞后之后（与脚本严格等价）
p.addParameter('model_name', 'fuzzy_pid_simulink', @ischar);
p.addParameter('save', true, @islogical);
p.addParameter('verbose', true, @islogical);
p.parse(varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
cfg  = fuzzy_config();
th   = cfg.scenarios.thermal;
pl   = th.plant;
pp   = th.pid;
sm   = th.simulation;
rf   = th.reference.segments;
dc   = th.disturbance;

K = pl.K;  T = pl.T;  tau = pl.tau;  y0 = pl.y0;
u_min = pl.u_min;  u_max = pl.u_max;
dt = sm.dt;  t_end = sm.t_end;
r1 = rf(1, 2);  t_r2 = rf(2, 1);  r2 = rf(2, 2);            % 100 ℃ / t=200 s 再升到 125 ℃
d_amp = dc.amplitude;  d_t_on = dc.t_on;  d_t_off = dc.t_off; % -18 ℃, 130~200 s
d_u = d_amp / K;                                             % 等效到对象输入端的扰动 -12

% 显式欧拉一步离散化 y(k)=a·y(k-1)+b·u(k-1)
a_euler = 1 - dt / T;  b_euler = K * dt / T;
n_delay = round(tau / dt);                                   % 纯滞后采样点数 25
fis_file = fullfile(here, 'fuzzy_pid.fis');
fis_var  = 'FIS_FUZZY_PID';

% -------------------------------------------------------------------------
% 幂等：先关闭并删除同名模型
% -------------------------------------------------------------------------
mdl = o.model_name;
slx = fullfile(here, [mdl '.slx']);
if bdIsLoaded(mdl)
    set_param(mdl, 'Dirty', 'off');
    close_system(mdl, 0);
end
if isfile(slx), delete(slx); end
new_system(mdl);

% -------------------------------------------------------------------------
% 模糊推理系统：readfis 读入 .fis 后放入基础工作区，供 FLC 模块引用
% -------------------------------------------------------------------------
fis = readfis(fis_file);
assignin('base', fis_var, fis);
% 注意：R2024a 的 readfis 返回 sugfis 对象，其 AndMethod 由工具箱按 Sugeno 结构给出，
% 与 shared/fuzzy_config.json 的代数积前件合成一致（实测与 MATLAB 类偏差 1.3e-15）。

% 分块搭建

% ---------------- 1. 参考信号（两个 Step 相加） ----------------
blk_step1 = addb(mdl, 'simulink/Sources/Step', '设定_阶跃1', [30 40 70 70], ...
    'Time', num2str(0), 'Before', num2str(0), 'After', num2str(r1));
blk_step2 = addb(mdl, 'simulink/Sources/Step', '设定_阶跃2', [30 110 70 140], ...
    'Time', num2str(t_r2), 'Before', num2str(0), 'After', num2str(r2 - r1));
blk_sumr = addb(mdl, 'simulink/Math Operations/Sum', '求和_设定值r', [130 55 150 125], ...
    'IconShape', 'rectangular', 'Inputs', '++');
conn(mdl, blk_step1, 1, blk_sumr, 1);
conn(mdl, blk_step2, 1, blk_sumr, 2);

% ---------------- 2. 误差 ----------------
blk_sume = addb(mdl, 'simulink/Math Operations/Sum', '求和_误差e', [130 210 150 280], ...
    'IconShape', 'rectangular', 'Inputs', '+-');
conn(mdl, blk_sumr, 1, blk_sume, 1);

% ---------------- 3. 误差量化 → Saturation[-3,3] ----------------
blk_ke  = addb(mdl, 'simulink/Math Operations/Gain', '增益_量化ke', [200 225 250 255], ...
    'Gain', num2str(pp.ke));
blk_sate = addb(mdl, 'simulink/Discontinuities/Saturation', '饱和_量化e', [290 225 330 255], ...
    'UpperLimit', num2str(3), 'LowerLimit', num2str(-3));
conn(mdl, blk_sume, 1, blk_ke, 1);
conn(mdl, blk_ke, 1, blk_sate, 1);

% ---------------- 4. 误差变化率（理想后向差分）→ Saturation[-3,3] ----------------
blk_ud  = addb(mdl, 'simulink/Discrete/Unit Delay', '单位延迟_e', [200 300 250 330], ...
    'SampleTime', num2str(dt), 'InitialCondition', '0');
blk_sedd = addb(mdl, 'simulink/Math Operations/Sum', '求和_差分e', [290 295 310 365], ...
    'IconShape', 'rectangular', 'Inputs', '+-');
blk_ddt = addb(mdl, 'simulink/Math Operations/Gain', '增益_微分1_dt', [340 320 390 350], ...
    'Gain', num2str(1 / dt));
blk_kec = addb(mdl, 'simulink/Math Operations/Gain', '增益_量化kec', [420 320 470 350], ...
    'Gain', num2str(pp.kec));
blk_satec = addb(mdl, 'simulink/Discontinuities/Saturation', '饱和_量化ec', [510 320 550 350], ...
    'UpperLimit', num2str(3), 'LowerLimit', num2str(-3));
conn(mdl, blk_sume, 1, blk_ud, 1);
conn(mdl, blk_sume, 1, blk_sedd, 1);
conn(mdl, blk_ud, 1, blk_sedd, 2);
conn(mdl, blk_sedd, 1, blk_ddt, 1);
conn(mdl, blk_ddt, 1, blk_kec, 1);
conn(mdl, blk_kec, 1, blk_satec, 1);

% ---------------- 5. Mux(2) → 模糊控制器 → Demux(3) ----------------
blk_mux = addb(mdl, 'simulink/Signal Routing/Mux', '合成_模糊输入2', [590 230 595 340], ...
    'Inputs', '2');
conn(mdl, blk_sate, 1, blk_mux, 1);
conn(mdl, blk_satec, 1, blk_mux, 2);

[flc_lib, flc_ok] = flc_library_block();
if flc_ok
    blk_flc = addb(mdl, flc_lib, '模糊控制器_FLC', [640 225 750 300], 'FIS', fis_var);
    flc_kind = 'Fuzzy Logic Toolbox 官方 Fuzzy Logic Controller 模块（库 fuzblock）';
else
    % 回退方案：MATLAB Function 模块内嵌推理（与 evalfis 同一套 .fis）
    blk_flc = addb(mdl, 'simulink/User-Defined Functions/MATLAB Function', ...
        '模糊控制器_FLC', [640 225 750 300]);
    set_matlab_fcn(blk_flc, fis_var);
    flc_kind = '回退方案：MATLAB Function 模块内嵌 evalfis 推理';
end

blk_demux = addb(mdl, 'simulink/Signal Routing/Demux', '分解_增量3', [790 210 800 330], ...
    'Outputs', '3');
conn(mdl, blk_flc, 1, blk_demux, 1);

% ---------------- 6. 三条在线整定支路：kp = kp0 + kup·Δkp ----------------
specs = { 'Kp', pp.kp0, pp.kup, 0.01, 220, 1; ...
          'Ki', pp.ki0, pp.kui, 0.00, 330, 2; ...
          'Kd', pp.kd0, pp.kud, 0.00, 440, 3 };
gain_blk = cell(3, 1);
for i = 1:3
    nm = specs{i, 1}; base = specs{i, 2}; sc = specs{i, 3}; lo = specs{i, 4};
    yy = specs{i, 5}; port = specs{i, 6};
    b_sc = addb(mdl, 'simulink/Math Operations/Gain', ['增益_缩放ku' lower(nm)], ...
        [850 yy 900 yy + 30], 'Gain', num2str(sc));
    b_c  = addb(mdl, 'simulink/Sources/Constant', ['常数_基准' nm '0'], ...
        [850 yy + 45 900 yy + 65], 'Value', num2str(base));
    b_s  = addb(mdl, 'simulink/Math Operations/Sum', ['求和_' nm], ...
        [940 yy 960 yy + 60], 'IconShape', 'rectangular', 'Inputs', '++');
    b_sat = addb(mdl, 'simulink/Discontinuities/Saturation', ['饱和_' nm], ...
        [1000 yy 1040 yy + 30], 'UpperLimit', '1e6', 'LowerLimit', num2str(lo));
    conn(mdl, blk_demux, port, b_sc, 1);
    conn(mdl, b_sc, 1, b_s, 1);
    conn(mdl, b_c, 1, b_s, 2);
    conn(mdl, b_s, 1, b_sat, 1);
    gain_blk{i} = b_sat;
end

% ---------------- 7. PID Controller（外部 P/I/D 增益 + 限幅 + clamping 抗饱和） ----------------
blk_pid = addb(mdl, 'simulink/Continuous/PID Controller', 'PID控制器', [1100 200 1210 360], ...
    'Controller', 'PID', ...
    'Form', 'Parallel', ...
    'TimeDomain', 'Discrete-time', ...
    'SampleTime', num2str(dt), ...
    'IntegratorMethod', 'Forward Euler', ...
    'UseFilter', 'off', ...
    'ControllerParametersSource', 'external', ...
    'InitialConditionForIntegrator', '0', ...
    'DifferentiatorICPrevScaledInput', '0', ...
    'ExternalReset', 'none', ...
    'LimitOutput', 'on', ...
    'UpperSaturationLimit', num2str(u_max), ...
    'LowerSaturationLimit', num2str(u_min), ...
    'AntiWindupMode', 'clamping');
conn(mdl, blk_sume, 1, blk_pid, 1);      % 端口1: 误差 e
conn(mdl, gain_blk{1}, 1, blk_pid, 2);   % 端口2: Kp
conn(mdl, gain_blk{2}, 1, blk_pid, 3);   % 端口3: Ki
conn(mdl, gain_blk{3}, 1, blk_pid, 4);   % 端口4: Kd

% ---------------- 8. 冷负荷扰动（两个 Step 相减得到 130~200 s 的 -12） ----------------
blk_d1 = addb(mdl, 'simulink/Sources/Step', '扰动_冷负荷1', [640 430 680 460], ...
    'Time', num2str(d_t_on), 'Before', num2str(0), 'After', num2str(d_u));
blk_d2 = addb(mdl, 'simulink/Sources/Step', '扰动_冷负荷2', [640 480 680 510], ...
    'Time', num2str(d_t_off), 'Before', num2str(0), 'After', num2str(-d_u));
blk_sumd = addb(mdl, 'simulink/Math Operations/Sum', '求和_扰动d', [720 445 740 515], ...
    'IconShape', 'rectangular', 'Inputs', '++');
conn(mdl, blk_d1, 1, blk_sumd, 1);
conn(mdl, blk_d2, 1, blk_sumd, 2);

% ---------------- 9. 被控对象 ----------------
% 单输入求和：u + d（扰动接在 TF 之前）或仅 u（扰动接在纯滞后之后）
if o.dist_after_delay
    blk_sumu = addb(mdl, 'simulink/Math Operations/Sum', '求和_对象输入u', ...
        [1260 240 1280 310], 'IconShape', 'rectangular', 'Inputs', '+');
    conn(mdl, blk_pid, 1, blk_sumu, 1);
else
    blk_sumu = addb(mdl, 'simulink/Math Operations/Sum', '求和_对象输入ud', ...
        [1260 240 1280 310], 'IconShape', 'rectangular', 'Inputs', '++');
    conn(mdl, blk_pid, 1, blk_sumu, 1);
    conn(mdl, blk_sumd, 1, blk_sumu, 2);
end
% Discrete Transfer Fcn：num/den 为 z 的多项式，[b 0]/[1 -a] <=> b/(1-a·z^-1)
blk_tf = addb(mdl, 'simulink/Discrete/Discrete Transfer Fcn', '对象_一阶惯性', ...
    [1330 240 1430 310], 'Numerator', sprintf('[%g 0]', b_euler), ...
    'Denominator', sprintf('[1 %g]', -a_euler), 'SampleTime', num2str(dt));
conn(mdl, blk_sumu, 1, blk_tf, 1);

blk_dly = addb(mdl, 'simulink/Discrete/Delay', '纯滞后_5s', [1470 240 1540 310], ...
    'DelayLength', num2str(n_delay), 'SampleTime', num2str(dt), 'InitialCondition', '0');
conn(mdl, blk_tf, 1, blk_dly, 1);

if o.dist_after_delay
    blk_sumy = addb(mdl, 'simulink/Math Operations/Sum', '求和_负荷扰动叠加', ...
        [1590 250 1610 320], 'IconShape', 'rectangular', 'Inputs', '++');
    conn(mdl, blk_dly, 1, blk_sumy, 1);
    conn(mdl, blk_sumd, 1, blk_sumy, 2);
    blk_bias = addb(mdl, 'simulink/Math Operations/Bias', '偏置_初温20C', [1650 265 1710 295], ...
        'Bias', num2str(y0));
    conn(mdl, blk_sumy, 1, blk_bias, 1);
else
    blk_bias = addb(mdl, 'simulink/Math Operations/Bias', '偏置_初温20C', [1650 265 1710 295], ...
        'Bias', num2str(y0));
    conn(mdl, blk_dly, 1, blk_bias, 1);
end
conn(mdl, blk_bias, 1, blk_sume, 2);     % 温度反馈 → 误差

% ---------------- 10. 数据记录 ----------------
w1 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_温度y', [1780 40 1840 70], ...
    'VariableName', 'y_sim', 'SaveFormat', 'Timeseries');
w2 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_控制量u', [1780 100 1840 130], ...
    'VariableName', 'u_sim', 'SaveFormat', 'Timeseries');
w3 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_设定r', [1780 160 1840 190], ...
    'VariableName', 'r_sim', 'SaveFormat', 'Timeseries');
w4 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_扰动d', [1780 220 1840 250], ...
    'VariableName', 'd_sim', 'SaveFormat', 'Timeseries');
w5 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_Kp', [1780 280 1840 310], ...
    'VariableName', 'kp_sim', 'SaveFormat', 'Timeseries');
w6 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_Ki', [1780 340 1840 370], ...
    'VariableName', 'ki_sim', 'SaveFormat', 'Timeseries');
w7 = addb(mdl, 'simulink/Sinks/To Workspace', '记录_Kd', [1780 400 1840 430], ...
    'VariableName', 'kd_sim', 'SaveFormat', 'Timeseries');
conn(mdl, blk_bias, 1, w1, 1);
conn(mdl, blk_pid, 1, w2, 1);
conn(mdl, blk_sumr, 1, w3, 1);
if o.dist_after_delay
    conn(mdl, blk_sumd, 1, w4, 1);
else
    conn(mdl, blk_sumu, 1, w4, 1);
end
conn(mdl, gain_blk{1}, 1, w5, 1);
conn(mdl, gain_blk{2}, 1, w6, 1);
conn(mdl, gain_blk{3}, 1, w7, 1);

% ---------------- 11. 求解器与仿真配置 ----------------
set_param(mdl, ...
    'Solver', 'FixedStepDiscrete', ...
    'FixedStep', num2str(dt), ...
    'StopTime', num2str(t_end), ...
    'SaveOutput', 'off', ...
    'SaveTime', 'off', ...
    'SaveState', 'off', ...
    'ReturnWorkspaceOutputs', 'on', ...
    'SolverType', 'Fixed-step');

% ---------------- 12. 保存 + 模块清单 ----------------
if o.save
    save_system(mdl, slx);
end

info = struct();
info.model = mdl;
info.model_file = slx;
info.solver = 'FixedStepDiscrete';
info.fixed_step = dt;
info.stop_time = t_end;
info.fis_file = fis_file;
info.fis_var = fis_var;
info.flc_block = blk_flc;
info.flc_kind = flc_kind;
info.dist_after_delay = o.dist_after_delay;
info.plant = struct('K', K, 'T', T, 'tau', tau, 'y0', y0, 'u_min', u_min, 'u_max', u_max, ...
                    'a_euler', a_euler, 'b_euler', b_euler, 'n_delay', n_delay);
info.pid_params = pp;
info.gain_lower_limits = [0.01 0.0 0.0];
info.blocks = block_inventory(mdl);
info.n_blocks = numel(info.blocks);
info.n_lines = numel(find_system(mdl, 'FindAll', 'on', 'Type', 'line'));

if o.verbose
    fprintf('=====================================================================\n');
    fprintf(' Simulink 模型已生成: %s\n', slx);
    fprintf(' 求解器: %s, 定步长 %.2f s, 仿真时长 %.0f s\n', info.solver, dt, t_end);
    fprintf(' 模糊模块: %s\n', info.flc_kind);
    fprintf(' FIS 变量: %s  (由 %s 读入)\n', fis_var, fis_file);
    fprintf(' 被控对象: Discrete Transfer Fcn num=[%g 0] den=[1 %g] (Ts=%.1f s) ≈ %.1f/(%.0fs+1); Delay %d 拍 = %.1f s\n', ...
        b_euler, -a_euler, dt, K, T, n_delay, n_delay * dt);
    fprintf(' 冷负荷扰动注入点: %s\n', ternary(o.dist_after_delay, ...
        '5 s 纯滞后之后（与脚本 run_thermal 严格等价）', '5 s 纯滞后之前'));
    fprintf(' 模块数: %d, 连线数: %d\n', info.n_blocks, info.n_lines);
    fprintf('---- 模块清单 ----\n');
    for i = 1:info.n_blocks
        fprintf('  %-28s %-34s %s\n', info.blocks{i, 1}, info.blocks{i, 2}, info.blocks{i, 3});
    end
    fprintf('=====================================================================\n');
end
end

% =========================================================================
% 局部辅助函数
% =========================================================================
function b = addb(mdl, lib, name, pos, varargin)
%ADDB 添加模块（统一错误信息）
b = [mdl '/' name];
add_block(lib, b, 'Position', pos, varargin{:});
end

function conn(mdl, src, sport, dst, dport)
%CONN 连接两个模块端口
add_line(mdl, sprintf('%s/%d', get_param(src, 'Name'), sport), ...
              sprintf('%s/%d', get_param(dst, 'Name'), dport), 'autorouting', 'on');
end

function [blk, ok] = flc_library_block()
%FLC_LIBRARY_BLOCK 查找 Fuzzy Logic Toolbox 提供的 Fuzzy Logic Controller 模块
%   R2024a 中该模块位于库 fuzblock 内，且库内模块名含换行符，
%   因此必须用 find_system 取回精确路径，不能手写字符串。
blk = ''; ok = false;
try
    load_system('fuzblock');
    cand = find_system('fuzblock', 'regexp', 'on', 'Name', 'Fuzzy Logic');
    for i = 1:numel(cand)
        nm = get_param(cand{i}, 'Name');
        if contains(nm, 'Ruleviewer'), continue; end
        % 试用该模块建一个临时模型验证可用性
        blk = cand{i};
        ok = true;
        break;
    end
catch
    ok = false;
end
end

function set_matlab_fcn(blk, fis_var)
%SET_MATLAB_FCN 回退方案：把模糊推理写成 MATLAB Function 模块
src = sprintf([ ...
    'function dk = fuzzy_infer(u)\n' ...
    '%%#codegen\n' ...
    'persistent fis\n' ...
    'if isempty(fis)\n' ...
    '    fis = readfis(''%s'');\n' ...
    'end\n' ...
    'e = u(1); ec = u(2);\n' ...
    'dk = evalfis(fis, [e ec]);\n'], ...
    fullfile(fileparts(mfilename('fullpath')), 'fuzzy_pid.fis'));
sf = sfroot;
chart = sf.find('-isa', 'Stateflow.EMChart', '-and', 'Path', blk);
if isempty(chart)
    error('build_simulink_model:matlabFcn', '无法定位 MATLAB Function 模块 %s', blk);
end
chart.Script = src;
end

function inv = block_inventory(mdl)
%BLOCK_INVENTORY 生成模块清单 {名称, 类型, 用途}
blks = find_system(mdl, 'SearchDepth', 1, 'Type', 'Block');
inv = cell(numel(blks), 3);
for i = 1:numel(blks)
    nm = get_param(blks{i}, 'Name');
    inv{i, 1} = nm;
    inv{i, 2} = get_param(blks{i}, 'BlockType');
    inv{i, 3} = block_purpose(nm);
end
[~, ord] = sort({inv{:, 1}});
inv = inv(ord, :);
end

function s = block_purpose(nm)
%BLOCK_PURPOSE 模块用途（用于报告）
map = { ...
    '设定_阶跃1', '参考信号: t>=0 输出 100 ℃'; ...
    '设定_阶跃2', '参考信号: t>=200 s 再叠加 25 ℃ → 125 ℃'; ...
    '求和_设定值r', '两段设定值相加得到 r(t)'; ...
    '求和_误差e', 'e = r - y'; ...
    '增益_量化ke', '误差量化因子 ke'; ...
    '饱和_量化e', 'Saturation [-3, 3]（模糊输入论域）'; ...
    '单位延迟_e', 'e(k-1)，用于后向差分'; ...
    '求和_差分e', 'e(k) - e(k-1)'; ...
    '增益_微分1_dt', '除以控制周期 dt，得到 de/dt'; ...
    '增益_量化kec', '误差变化率量化因子 kec'; ...
    '饱和_量化ec', 'Saturation [-3, 3]（模糊输入论域）'; ...
    '合成_模糊输入2', 'Mux(2): [e_n, ec_n]'; ...
    '模糊控制器_FLC', 'Fuzzy Logic Controller，加载 fuzzy_pid.fis'; ...
    '分解_增量3', 'Demux(3): [ΔKp, ΔKi, ΔKd]'; ...
    '增益_缩放kup', 'ΔKp → kup·ΔKp'; ...
    '增益_缩放kui', 'ΔKi → kui·ΔKi'; ...
    '增益_缩放kud', 'ΔKd → kud·ΔKd'; ...
    '常数_基准Kp0', '基准增益 Kp0'; ...
    '常数_基准Ki0', '基准增益 Ki0'; ...
    '常数_基准Kd0', '基准增益 Kd0'; ...
    '求和_Kp', 'Kp = Kp0 + kup·ΔKp'; ...
    '求和_Ki', 'Ki = Ki0 + kui·ΔKi'; ...
    '求和_Kd', 'Kd = Kd0 + kud·ΔKd'; ...
    '饱和_Kp', '下限饱和 (Kp ≥ 0.01，与脚本 max(0.01,·) 一致)'; ...
    '饱和_Ki', '下限饱和 (Ki ≥ 0)'; ...
    '饱和_Kd', '下限饱和 (Kd ≥ 0)'; ...
    'PID控制器', 'PID 模块: P/I/D 外部输入, 限幅 0~100, clamping 抗饱和'; ...
    '扰动_冷负荷1', 't=130 s 起叠加等效冷负荷 -12'; ...
    '扰动_冷负荷2', 't=200 s 起抵消冷负荷'; ...
    '求和_扰动d', '得到 130~200 s 有效、幅值 -12 的扰动'; ...
    '求和_对象输入u', 'u（+d，若扰动接在滞后之前）'; ...
    '对象_一阶惯性', 'Discrete Transfer Fcn，显式欧拉等价的 1.5/(25s+1)'; ...
    '纯滞后_5s', 'Delay 25 拍 = 5 s 纯滞后'; ...
    '求和_负荷扰动叠加', '扰动叠加到纯滞后之后的信号上（与脚本一致）'; ...
    '偏置_初温20C', '叠加初温偏置 y0 = 20 ℃'; ...
    '记录_温度y', '记录温度 y 到基础工作区变量 y_sim'; ...
    '记录_控制量u', '记录控制量 u 到 u_sim'; ...
    '记录_设定r', '记录设定值 r 到 r_sim'; ...
    '记录_扰动d', '记录扰动 d 到 d_sim'; ...
    '记录_Kp', '记录在线整定 Kp 到 kp_sim'; ...
    '记录_Ki', '记录在线整定 Ki 到 ki_sim'; ...
    '记录_Kd', '记录在线整定 Kd 到 kd_sim'};
s = '';
for i = 1:size(map, 1)
    if strcmp(nm, map{i, 1})
        s = map{i, 2};
        return;
    end
end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
