classdef fuzzy_pid_controller < handle
    %FUZZY_PID_CONTROLLER 模糊自适应 PID 控制器（配置驱动，无需任何工具箱）
    %
    %   与 Python 版 python/fuzzy_pid.py 严格同源：
    %     - 隶属度函数与 49 条规则来自 shared/fuzzy_config.json
    %     - 相同的条件积分抗饱和 (conditional integration) 逻辑
    %     - 相同的可选一阶滤波微分
    %
    %   构造:
    %     c = fuzzy_pid_controller(kp0, ki0, kd0, ke, kec, kup, kui, kud, ...
    %                              out_min, out_max, dt)
    %     c = fuzzy_pid_controller(..., 'anti_windup', true, 'kd_filter_tau', 5.0)
    %
    %   调用:
    %     u = c.compute(target, current)

    properties
        kp0; ki0; kd0;          % 基准增益
        ke; kec;                % 误差 / 误差变化率量化因子
        kup; kui; kud;          % 增量比例缩放因子
        out_min; out_max; dt;   % 限幅与控制周期
        anti_windup = true;     % 抗积分饱和开关
        kd_filter_tau = 0.0;    % 微分一阶滤波时间常数 (s)
        % 运行状态
        prev_error = 0; ec_filt = 0; integral_term = 0;
        kp; ki; kd;             % 当前（自整定后）增益
        limit_kp; limit_ki; limit_kd;   % 在线整定的安全边界（由自身参数自动推导）
        last_norm = [0 0 0];    % 模糊推理输出的归一化增量
        saturated = false;
    end

    properties (Access = private)
        fis                     % 隶属度函数与规则库（来自共享配置）
    end

    methods
        function obj = fuzzy_pid_controller(kp0, ki0, kd0, ke, kec, kup, kui, kud, ...
                                            out_min, out_max, dt, varargin)
            p = inputParser;
            addParameter(p, 'anti_windup', true, @islogical);
            addParameter(p, 'kd_filter_tau', 0.0, @(x) isnumeric(x) && isscalar(x));
            parse(p, varargin{:});

            obj.kp0 = kp0; obj.ki0 = ki0; obj.kd0 = kd0;
            obj.ke = ke; obj.kec = kec;
            obj.kup = kup; obj.kui = kui; obj.kud = kud;
            obj.out_min = out_min; obj.out_max = out_max; obj.dt = dt;
            obj.anti_windup = p.Results.anti_windup;
            obj.kd_filter_tau = p.Results.kd_filter_tau;

            cfg = fuzzy_config();
            obj.fis = cfg.fuzzy;

            % 增益安全边界：由自身基准增益与缩放因子推导（与 Python auto_gain_limits 同规则）
            % 取"基准±3倍缩放量"与"3倍基准值"中较宽松者，避免正常整定区间被裁剪，
            % 同时保证 Python / MATLAB 两侧行为一致。
            obj.limit_kp = [0.01, max(4 * kp0, kp0 + 3 * abs(kup))];
            obj.limit_ki = [0.0,  max(3 * ki0, ki0 + 3 * abs(kui))];
            obj.limit_kd = [0.0,  max(3 * kd0, kd0 + 3 * abs(kud))];

            obj.reset();
        end

        function reset(obj)
            obj.prev_error = 0; obj.ec_filt = 0; obj.integral_term = 0;
            obj.kp = obj.kp0; obj.ki = obj.ki0; obj.kd = obj.kd0;
            obj.last_norm = [0 0 0];
            obj.saturated = false;
        end

        function u = compute(obj, target, current)
            error_ = target - current;
            raw_rate = (error_ - obj.prev_error) / obj.dt;

            % 一阶滤波微分
            if obj.kd_filter_tau > 0
                a = obj.dt / (obj.kd_filter_tau + obj.dt);
                obj.ec_filt = a * raw_rate + (1 - a) * obj.ec_filt;
            else
                obj.ec_filt = raw_rate;
            end
            ec = obj.ec_filt;

            % ---- 模糊在线整定 ----
            lo = obj.fis.universe(1); hi = obj.fis.universe(2);
            e_n = min(hi, max(lo, error_ * obj.ke));
            ec_n = min(hi, max(lo, ec * obj.kec));
            [d_kp, d_ki, d_kd] = obj.infer(e_n, ec_n);
            obj.last_norm = [d_kp, d_ki, d_kd];

            obj.kp = min(obj.limit_kp(2), max(obj.limit_kp(1), obj.kp0 + d_kp * obj.kup));
            obj.ki = min(obj.limit_ki(2), max(obj.limit_ki(1), obj.ki0 + d_ki * obj.kui));
            obj.kd = min(obj.limit_kd(2), max(obj.limit_kd(1), obj.kd0 + d_kd * obj.kud));

            % ---- 增量式积分 + 条件积分抗饱和 ----
            p_term = obj.kp * error_;
            d_term = obj.kd * ec;
            delta_i = obj.ki * error_ * obj.dt;
            u_tent = p_term + obj.integral_term + delta_i + d_term;

            if obj.anti_windup
                blocked_high = (u_tent > obj.out_max) && (error_ > 0);
                blocked_low  = (u_tent < obj.out_min) && (error_ < 0);
                if ~(blocked_high || blocked_low)
                    obj.integral_term = obj.integral_term + delta_i;
                end
            else
                obj.integral_term = obj.integral_term + delta_i;
            end

            u_unsat = p_term + obj.integral_term + d_term;
            u = min(obj.out_max, max(obj.out_min, u_unsat));
            obj.saturated = (u ~= u_unsat);
            obj.prev_error = error_;
        end

        function [d_kp, d_ki, d_kd] = infer_point(obj, e_n, ec_n)
            %INFER_POINT 公开的论域输入 -> 归一化输出接口（用于交叉验证/绘图）
            [d_kp, d_ki, d_kd] = obj.infer(e_n, ec_n);
        end

        function [e_axis, ec_axis, surf] = surface(obj, n)
            %SURFACE 在整个论域上计算三张决策曲面（用于绘图/交叉验证）
            if nargin < 2, n = 61; end
            lo = obj.fis.universe(1); hi = obj.fis.universe(2);
            e_axis = linspace(lo, hi, n);
            ec_axis = linspace(lo, hi, n);
            surf = zeros(n, n, 3);
            for i = 1:n
                for j = 1:n
                    [a, b, c] = obj.infer(e_axis(i), ec_axis(j));
                    surf(i, j, :) = [a, b, c];
                end
            end
        end
    end

    methods (Access = private)
        function mu = membership(obj, x)
            %MEMBERSHIP 按配置计算 7 个模糊子集的隶属度向量
            labels = obj.fis.labels;
            mu = zeros(1, numel(labels));
            for k = 1:numel(labels)
                mfk = obj.fis.mf.(labels{k});
                prm = mfk.params(:)';
                switch lower(mfk.type)
                    case 'trimf'
                        mu(k) = local_trimf(x, prm);
                    case 'trapmf'
                        mu(k) = local_trapmf(x, prm);
                    otherwise
                        error('未知隶属度函数类型: %s', mfk.type);
                end
            end
        end

        function [d_kp, d_ki, d_kd] = infer(obj, e_n, ec_n)
            %INFER 零阶 Takagi-Sugeno 推理：代数积前件合成 + 单点输出加权平均
            %      与 fuzzy_pid.fis（AndMethod='prod', DefuzzMethod='wtaver'）
            %      及 Python 实现逐项对应，实测最大偏差 < 1e-12。
            ue = obj.membership(e_n);
            uec = obj.membership(ec_n);
            alpha = ue' * uec;                 % 前件合成: α_ij = μe(i)·μec(j)（代数积 T-范数）
            w = sum(alpha(:));
            if w < 1e-12
                d_kp = 0; d_ki = 0; d_kd = 0;
                return;
            end
            rules = obj.fis.rules;
            d_kp = sum(sum(alpha .* double(rules.dkp))) / w;
            d_ki = sum(sum(alpha .* double(rules.dki))) / w;
            d_kd = sum(sum(alpha .* double(rules.dkd))) / w;
        end
    end
end

% ---------------------------------------------------------------------------
function y = local_trimf(x, p)
a = p(1); b = p(2); c = p(3);
if x <= a || x >= c
    y = 0;
elseif x == b
    y = 1;
elseif x < b
    y = (x - a) / (b - a);
else
    y = (c - x) / (c - b);
end
end

function y = local_trapmf(x, p)
a = p(1); b = p(2); c = p(3); d = p(4);
if x <= a || x >= d
    y = 0;
elseif x >= b && x <= c
    y = 1;
elseif x < b
    y = (x - a) / (b - a);
else
    y = (d - x) / (d - c);
end
end
