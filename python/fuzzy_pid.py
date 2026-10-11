"""
模糊自适应 PID 控制器核心模块 (Fuzzy Adaptive PID Controller)
============================================================
纯 NumPy 实现的双输入-三输出 Mamdani 模糊推理系统，特点：

1. **配置驱动**：隶属度函数与 49 条规则全部来自 ``shared/fuzzy_config.json``，
   与 MATLAB 类、``fuzzy_pid.fis`` 严格同源，杜绝"同名不同算法"。
2. **统一控制器骨架**：``PIDCore`` 定义了积分累加、抗积分饱和、微分滤波等
   公共环节；固定参数 PID 与模糊自适应 PID 只在"增益如何取值"上不同，
   从而保证对比实验的**唯一变量**就是自适应机制本身。
3. **工程化抗积分饱和**：采用标准条件积分法 (conditional integration)，
   输出饱和且误差继续推向饱和方向时冻结积分，反方向时允许回退解饱和。
"""

from __future__ import annotations

import numpy as np

from fuzzy_config import fuzzy_block


# ---------------------------------------------------------------------------
# 隶属度函数
# ---------------------------------------------------------------------------
def trimf(x, params):
    """标准三角形隶属度函数 (a, b, c)，b 为峰值点。"""
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf(x, params):
    """标准梯形隶属度函数 (a, b, c, d)，[b, c] 为平顶区。"""
    a, b, c, d = params
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


_MF_FUNC = {"trimf": trimf, "trapmf": trapmf}


class FuzzyInferenceSystem:
    """
    双输入-三输出 Mamdani 模糊推理系统。

    输入 : e, ec —— 已量化到标准论域 [-3, 3]
    输出 : ΔKp, ΔKi, ΔKd —— 同样位于 [-3, 3]

    推理链（零阶 Takagi-Sugeno，与 fuzzy_pid.fis / Simulink 官方推理核严格一致）：
      1. 模糊化 (trapmf/trimf)
      2. 前件合成：代数积 T-范数  α_ij = μe(i) · μec(j)   （MATLAB: AndMethod='prod'）
      3. 反模糊化：单点输出加权平均  y* = Σα_k·C_k / Σα_k （MATLAB: sugeno + wtaver）

    说明：若改用"极小运算 + Mamdani 面积重心"，α_ij 与清晰化方式都会改变，
    实测与官方 evalfis 的最大偏差分别可达 0.52 与 1.09（论域宽度仅 6.0），
    会使脚本与 Simulink 给出两套不同结果，故本项目统一采用上述零阶 TS 结构。
    """

    def __init__(self, cfg=None):
        cfg = fuzzy_block() if cfg is None else cfg
        self.labels = list(cfg["labels"])
        self.index_map = {name: i for i, name in enumerate(self.labels)}
        self.centers = np.asarray(cfg["centers"], dtype=float)
        self.lo, self.hi = float(cfg["universe"][0]), float(cfg["universe"][1])
        self.mf_type = [cfg["mf"][lab]["type"] for lab in self.labels]
        self.mf_params = [list(cfg["mf"][lab]["params"]) for lab in self.labels]

        rules = cfg["rules"]
        # 规则表中的整数即为该语言变量对应的论域中心值，故可直接参与重心加权。
        self.rule_delta_kp = np.asarray(rules["dkp"], dtype=float)
        self.rule_delta_ki = np.asarray(rules["dki"], dtype=float)
        self.rule_delta_kd = np.asarray(rules["dkd"], dtype=float)
        for name, mat in (("dkp", self.rule_delta_kp), ("dki", self.rule_delta_ki), ("dkd", self.rule_delta_kd)):
            if mat.shape != (7, 7):
                raise ValueError(f"规则矩阵 {name} 形状应为 (7,7)，实际为 {mat.shape}")

    # -- 模糊化 --------------------------------------------------------------
    def calc_membership(self, val: float) -> np.ndarray:
        """计算输入在 7 个模糊子集上的隶属度向量。"""
        v = float(np.clip(val, self.lo, self.hi))
        return np.array([
            _MF_FUNC[t](v, p) for t, p in zip(self.mf_type, self.mf_params)
        ])

    # -- 推理 + 清晰化 ------------------------------------------------------
    def infer(self, e_norm: float, ec_norm: float):
        """返回 (ΔKp, ΔKi, ΔKd) 的归一化论域值。"""
        ue = self.calc_membership(e_norm)
        uec = self.calc_membership(ec_norm)
        alpha = np.outer(ue, uec)          # 前件合成: α_ij = μe(i)·μec(j)（代数积）
        w = alpha.sum()
        if w < 1e-12:
            return 0.0, 0.0, 0.0
        return (
            float((alpha * self.rule_delta_kp).sum() / w),
            float((alpha * self.rule_delta_ki).sum() / w),
            float((alpha * self.rule_delta_kd).sum() / w),
        )

    def surface(self, n: int = 61):
        """在整个论域上计算三张决策曲面，用于绘图与一致性校验。"""
        e = np.linspace(self.lo, self.hi, n)
        ec = np.linspace(self.lo, self.hi, n)
        E, EC = np.meshgrid(e, ec)
        out = np.zeros((3, n, n))
        for i in range(n):
            for j in range(n):
                out[0, i, j], out[1, i, j], out[2, i, j] = self.infer(E[i, j], EC[i, j])
        return e, ec, out


# ---------------------------------------------------------------------------
# 控制器骨架
# ---------------------------------------------------------------------------
class PIDCore:
    """
    增量式积分 + 条件积分抗饱和 + 可选一阶滤波微分的 PID 骨架。

    参数
    ----
    dt            : 控制周期 (s)
    out_min/out_max: 执行机构物理限幅
    anti_windup   : False 时退化为工业界"朴素积分"，用于复现积分饱和现象
    kd_filter_tau : 微分一阶滤波时间常数 (s)，0 表示理想微分
    """

    def __init__(self, out_min=-np.inf, out_max=np.inf, dt=0.01,
                 anti_windup=True, kd_filter_tau=0.0):
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.dt = float(dt)
        self.anti_windup = bool(anti_windup)
        self.kd_filter_tau = float(kd_filter_tau)
        self.name = "PID"
        self.reset()

    # -- 状态复位 ----------------------------------------------------------
    def reset(self):
        self.prev_error = 0.0
        self.ec_filt = 0.0
        self.integral_term = 0.0
        self.kp = float("nan")
        self.ki = float("nan")
        self.kd = float("nan")
        self.last_delta = (0.0, 0.0, 0.0)
        self.saturated = False

    # -- 子类实现：根据状态给出本步增益 ------------------------------------
    def _gains(self, error, error_rate):
        raise NotImplementedError

    @staticmethod
    def _lowpass(x_new, x_old, dt, tau):
        if tau <= 0.0:
            return x_new
        a = dt / (tau + dt)
        return a * x_new + (1.0 - a) * x_old

    # -- 主计算 ------------------------------------------------------------
    def compute(self, target, current):
        error = float(target) - float(current)
        raw_rate = (error - self.prev_error) / self.dt

        # 一阶滤波微分：抑制测量噪声被微分环节放大，同时避免理想微分引起的代数环
        self.ec_filt = self._lowpass(raw_rate, self.ec_filt, self.dt, self.kd_filter_tau)
        error_rate = self.ec_filt

        self.kp, self.ki, self.kd = self._gains(error, error_rate)

        p_term = self.kp * error
        d_term = self.kd * error_rate
        delta_i = self.ki * error * self.dt

        u_tentative = p_term + self.integral_term + delta_i + d_term

        # 条件积分抗饱和：饱和且误差继续推向饱和方向则冻结积分；反向则允许回退
        if self.anti_windup:
            push_up = error > 0.0
            push_down = error < 0.0
            blocked_high = u_tentative > self.out_max and push_up
            blocked_low = u_tentative < self.out_min and push_down
            if not (blocked_high or blocked_low):
                self.integral_term += delta_i
        else:
            self.integral_term += delta_i

        u_unsat = p_term + self.integral_term + d_term
        u = float(np.clip(u_unsat, self.out_min, self.out_max))
        self.saturated = bool(u != u_unsat)
        self.prev_error = error
        return u


class ConventionalPID(PIDCore):
    """常规固定参数 PID 控制器（工业基准组）。"""

    def __init__(self, kp, ki, kd, **kwargs):
        super().__init__(**kwargs)
        self.kp_const = float(kp)
        self.ki_const = float(ki)
        self.kd_const = float(kd)
        self.name = "PID"

    def reset(self):
        super().reset()
        self.kp = getattr(self, "kp_const", float("nan"))
        self.ki = getattr(self, "ki_const", float("nan"))
        self.kd = getattr(self, "kd_const", float("nan"))

    def _gains(self, error, error_rate):
        return self.kp_const, self.ki_const, self.kd_const


def auto_gain_limits(kp0, ki0, kd0, kup, kui, kud):
    """
    由控制器自身的基准增益与缩放因子推导在线整定的安全边界。

    意义：模糊推理输出 ΔK* 的理论范围为 [-3, 3]，但当缩放因子被整定得很大时，
    增益可能被推到物理上荒谬的区间。这里以"基准值 ±3 倍缩放量"与"3 倍基准值"
    中较宽松者为上限，既保证安全又不会在正常整定区间内裁剪有效控制作用。

    注意：本函数只依赖控制器自身参数，因此 Python 与 MATLAB 两侧、
    以及不同参数组（默认因子 / 寻优因子）都能自动得到一致的边界——
    否则会出现"某一种实现偷偷把增益裁剪掉、另一种没有"的隐蔽不一致。
    """
    return {
        "kp_min": 0.01, "kp_max": max(4.0 * kp0, kp0 + 3.0 * abs(kup)),
        "ki_min": 0.0, "ki_max": max(3.0 * ki0, ki0 + 3.0 * abs(kui)),
        "kd_min": 0.0, "kd_max": max(3.0 * kd0, kd0 + 3.0 * abs(kud)),
    }


class FuzzyAdaptivePID(PIDCore):
    """
    模糊自适应 PID 控制器。

    Kp = Kp0 + kup·ΔKp*,  Ki = Ki0 + kui·ΔKi*,  Kd = Kd0 + kud·ΔKd*
    其中 ΔK* 为模糊推理输出的归一化量（论域 [-3, 3]），量化因子 ke/kec 负责
    把物理量纲的 e、ec 映射到论域。
    """

    def __init__(self, kp0, ki0, kd0, ke=1.0, kec=1.0,
                 kup=1.0, kui=0.1, kud=0.1, gain_limits=None, **kwargs):
        # 注意：先写入基准参数，再调用父类构造函数（父类会调用 reset()）
        self.kp0 = float(kp0)
        self.ki0 = float(ki0)
        self.kd0 = float(kd0)
        self.ke = float(ke)
        self.kec = float(kec)
        self.kup = float(kup)
        self.kui = float(kui)
        self.kud = float(kud)
        # 增益上下限：保证在线整定后的增益不会跑到物理上荒谬的区间
        # （缺省由自身参数自动推导，见 auto_gain_limits）
        self.gain_limits = dict(gain_limits) if gain_limits else auto_gain_limits(
            self.kp0, self.ki0, self.kd0, self.kup, self.kui, self.kud)
        self.fis = FuzzyInferenceSystem()
        self.name = "FuzzyPID"
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.kp = self.kp0
        self.ki = self.ki0
        self.kd = self.kd0
        self.last_norm = (0.0, 0.0, 0.0)
        self.last_mu_e = np.zeros(7)
        self.last_mu_ec = np.zeros(7)

    def _gains(self, error, error_rate):
        lo, hi = self.fis.lo, self.fis.hi
        e_norm = float(np.clip(error * self.ke, lo, hi))
        ec_norm = float(np.clip(error_rate * self.kec, lo, hi))

        d_kp, d_ki, d_kd = self.fis.infer(e_norm, ec_norm)
        self.last_norm = (d_kp, d_ki, d_kd)
        self.last_mu_e = self.fis.calc_membership(e_norm)
        self.last_mu_ec = self.fis.calc_membership(ec_norm)

        kp = self.kp0 + d_kp * self.kup
        ki = self.ki0 + d_ki * self.kui
        kd = self.kd0 + d_kd * self.kud

        lim = self.gain_limits
        kp = min(max(kp, lim.get("kp_min", 1e-3)), lim.get("kp_max", np.inf))
        ki = min(max(ki, lim.get("ki_min", 0.0)), lim.get("ki_max", np.inf))
        kd = min(max(kd, lim.get("kd_min", 0.0)), lim.get("kd_max", np.inf))

        self.last_delta = (kp - self.kp0, ki - self.ki0, kd - self.kd0)
        return kp, ki, kd


def build_controller(kind, cfg, **overrides):
    """
    依据配置构造控制器，保证 Python 与 MATLAB 使用完全相同的默认参数。

    kind: 'pid' | 'pid_opt' | 'fuzzy' | 'fuzzy_opt'
    """
    p = dict(cfg["pid"])
    p.update({k: v for k, v in overrides.items() if v is not None})
    dt = float(cfg["simulation"]["dt"])
    common = dict(out_min=cfg["plant"]["u_min"], out_max=cfg["plant"]["u_max"],
                  dt=dt, anti_windup=overrides.get("anti_windup", True),
                  kd_filter_tau=overrides.get("kd_filter_tau", 0.0))

    if kind in ("pid", "pid_opt"):
        ctrl = ConventionalPID(p["kp0"], p["ki0"], p["kd0"], **common)
        ctrl.name = "PID-抗饱和" if common["anti_windup"] else "PID-无抗饱和"
        return ctrl
    if kind in ("fuzzy", "fuzzy_opt"):
        ctrl = FuzzyAdaptivePID(
            p["kp0"], p["ki0"], p["kd0"],
            ke=p["ke"], kec=p["kec"], kup=p["kup"], kui=p["kui"], kud=p["kud"],
            **common)
        ctrl.name = "模糊自适应PID"
        return ctrl
    raise ValueError(f"未知控制器类型: {kind}")
