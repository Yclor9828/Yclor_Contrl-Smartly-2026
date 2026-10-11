classdef fixed_pid_controller < handle
    %FIXED_PID_CONTROLLER 常规固定参数 PID 控制器（对照组）
    %
    %   与 fuzzy_pid_controller 共用完全相同的积分/抗饱和/滤波微分骨架，
    %   唯一差别是增益恒定不变——从而保证对比实验的唯一变量是"自适应机制"。
    %
    %   构造:
    %     c = fixed_pid_controller(kp, ki, kd, out_min, out_max, dt)
    %     c = fixed_pid_controller(..., 'anti_windup', false)   % 复现积分饱和

    properties
        kp; ki; kd;
        out_min; out_max; dt;
        anti_windup = true;
        kd_filter_tau = 0.0;
        prev_error = 0; ec_filt = 0; integral_term = 0;
        saturated = false;
    end

    methods
        function obj = fixed_pid_controller(kp, ki, kd, out_min, out_max, dt, varargin)
            p = inputParser;
            addParameter(p, 'anti_windup', true, @islogical);
            addParameter(p, 'kd_filter_tau', 0.0, @(x) isnumeric(x) && isscalar(x));
            parse(p, varargin{:});

            obj.kp = kp; obj.ki = ki; obj.kd = kd;
            obj.out_min = out_min; obj.out_max = out_max; obj.dt = dt;
            obj.anti_windup = p.Results.anti_windup;
            obj.kd_filter_tau = p.Results.kd_filter_tau;
            obj.reset();
        end

        function reset(obj)
            obj.prev_error = 0; obj.ec_filt = 0; obj.integral_term = 0;
            obj.saturated = false;
        end

        function u = compute(obj, target, current)
            error_ = target - current;
            raw_rate = (error_ - obj.prev_error) / obj.dt;
            if obj.kd_filter_tau > 0
                a = obj.dt / (obj.kd_filter_tau + obj.dt);
                obj.ec_filt = a * raw_rate + (1 - a) * obj.ec_filt;
            else
                obj.ec_filt = raw_rate;
            end

            p_term = obj.kp * error_;
            d_term = obj.kd * obj.ec_filt;
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
    end
end
