"""
被控对象模型库 (Plant models)
============================
包含两个经过参数辨识/文献取值、量纲自洽的工业对象：

1. ``FOPDTPlant``  —— 一阶惯性加纯滞后模型，描述工业电阻加热炉温度过程
   ``T·dy/dt = -(y - y0) + K·u(t - τ) + d(t)``
2. ``DCMotorPlant`` —— 直流伺服电机机电耦合模型（电枢回路 + 机械运动方程）
   ``La·dia/dt = ua - Ra·ia - Ke·ω``，``J·dω/dt = Kt·ia - B·ω - TL``

两者都支持：
- 参数摄动（用于鲁棒性蒙特卡洛实验）：直接传入覆盖参数
- 量测噪声（用于微分噪声实验）：``measure()`` 返回叠加白噪声的量测值，
  而 ``y`` 始终保存真实物理量，指标计算一律基于真值，避免用噪声自欺欺人
"""

from __future__ import annotations

import numpy as np


class FOPDTPlant:
    """一阶惯性 + 纯滞后 (First Order Plus Dead Time) 温度对象。"""

    def __init__(self, K=1.5, T=25.0, tau=5.0, dt=0.2, y0=20.0,
                 u_min=0.0, u_max=100.0, noise_std=0.0, seed=0):
        self.K = float(K)
        self.T = float(T)
        self.tau = float(tau)
        self.dt = float(dt)
        self.y0 = float(y0)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.y = self.y0
        self.y_meas = self.y0
        self.n_delay = max(1, int(round(self.tau / self.dt)))
        self.u_buffer = [0.0] * self.n_delay

    def measure(self):
        """传感器量测（含噪声），控制器只能看到这个信号。"""
        self.y_meas = self.y + (self.rng.normal(0.0, self.noise_std)
                                if self.noise_std > 0 else 0.0)
        return self.y_meas

    def step(self, u, disturbance=0.0):
        """推进一步：先经过纯滞后环节，再积分一阶惯性环节。"""
        self.u_buffer.append(float(u))
        u_delayed = self.u_buffer.pop(0)
        dy = (-(self.y - self.y0) + self.K * u_delayed + disturbance) / self.T * self.dt
        self.y += dy
        return self.y


class DCMotorPlant:
    """直流伺服电机：电枢回路 + 机械转子两状态机电耦合模型。"""

    def __init__(self, Ra=1.2, La=0.03, Kt=0.08, Ke=0.08, B=0.005, J0=0.008,
                 dt=0.001, u_min=-36.0, u_max=36.0, noise_std=0.0, seed=0):
        self.Ra = float(Ra)
        self.La = float(La)
        self.Kt = float(Kt)
        self.Ke = float(Ke)
        self.B = float(B)
        self.J0 = float(J0)
        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.ia = 0.0
        self.omega = 0.0
        self.w_meas = 0.0

    def measure(self):
        self.w_meas = self.omega + (self.rng.normal(0.0, self.noise_std)
                                    if self.noise_std > 0 else 0.0)
        return self.w_meas

    def step(self, u, load_torque=0.0, J=None):
        J = self.J0 if J is None else float(J)
        dia = (u - self.Ra * self.ia - self.Ke * self.omega) / self.La * self.dt
        dw = (self.Kt * self.ia - self.B * self.omega - load_torque) / J * self.dt
        self.ia += dia
        self.omega += dw
        return self.omega


def thermal_plant_from_config(cfg, K=None, T=None, tau=None, noise_std=0.0, seed=0):
    """依据共享配置构造加热炉对象，可对 K/T/tau 做参数摄动。"""
    p = cfg["plant"]
    return FOPDTPlant(
        K=p["K"] if K is None else K,
        T=p["T"] if T is None else T,
        tau=p["tau"] if tau is None else tau,
        y0=p["y0"], dt=cfg["simulation"]["dt"],
        u_min=p["u_min"], u_max=p["u_max"],
        noise_std=noise_std, seed=seed)


def motor_plant_from_config(cfg, J0=None, Ra=None, Kt=None, Ke=None,
                            La=None, B=None, noise_std=0.0, seed=0):
    """依据共享配置构造直流电机对象，可对机电参数做参数摄动。"""
    p = cfg["plant"]
    return DCMotorPlant(
        Ra=p["Ra"] if Ra is None else Ra,
        La=p["La"] if La is None else La,
        Kt=p["Kt"] if Kt is None else Kt,
        Ke=p["Ke"] if Ke is None else Ke,
        B=p["B"] if B is None else B,
        J0=p["J0"] if J0 is None else J0,
        dt=cfg["simulation"]["dt"], u_min=p["u_min"], u_max=p["u_max"],
        noise_std=noise_std, seed=seed)


class SOPDTPlant:
    """
    二阶惯性 + 纯滞后 (Second Order Plus Dead Time) 液位过程。

    G(s) = K / [(T1·s+1)(T2·s+1)] · e^{-τs}

    状态方程（串联两个一阶惯性）：
        T1·dx1/dt = -(x1 - y0) + K·u(t-τ)
        T2·dx2/dt = -(x2 - x1)       ← 第二容腔跟随第一容腔
    输出 y = x2；扰动 d(t) 直接叠加到 x2（流入量等效）。

    典型工业背景：双容串联液位槽——上游槽液位（x1）通过调节阀驱动下游槽液位（x2=输出）。
    """

    def __init__(self, K=0.96, T1=30.0, T2=15.0, tau=3.0, dt=0.5, y0=20.0,
                 u_min=0.0, u_max=100.0, noise_std=0.0, seed=0):
        self.K = float(K)
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.tau = float(tau)
        self.dt = float(dt)
        self.y0 = float(y0)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.x1 = self.y0   # 第一容腔状态（跟随参考值初始化为 y0）
        self.x2 = self.y0   # 第二容腔状态（输出）
        self.y = self.y0
        self.y_meas = self.y0
        self.n_delay = max(1, int(round(self.tau / self.dt)))
        self.u_buffer = [0.0] * self.n_delay

    def measure(self):
        """传感器量测（含量化噪声），控制器只能读这个信号。"""
        self.y_meas = self.y + (self.rng.normal(0.0, self.noise_std)
                                if self.noise_std > 0 else 0.0)
        return self.y_meas

    def step(self, u, disturbance=0.0):
        """推进一步：u → 滞后队列 → 积分两阶惯性。"""
        self.u_buffer.append(float(u))
        u_delayed = self.u_buffer.pop(0)
        # 第一容腔：受控输入
        dx1 = (-(self.x1 - self.y0) + self.K * u_delayed) / self.T1 * self.dt
        self.x1 += dx1
        # 第二容腔：接收第一容腔输出 + 进液扰动
        dx2 = (-(self.x2 - self.x1) + disturbance) / self.T2 * self.dt
        self.x2 += dx2
        self.y = self.x2
        return self.y


def level_plant_from_config(cfg, K=None, T1=None, T2=None, tau=None,
                             noise_std=0.0, seed=0):
    """依据共享配置构造液位双容过程对象，可对参数做摄动。"""
    p = cfg["plant"]
    return SOPDTPlant(
        K=p["K"] if K is None else K,
        T1=p["T1"] if T1 is None else T1,
        T2=p["T2"] if T2 is None else T2,
        tau=p["tau"] if tau is None else tau,
        y0=p["y0"], dt=cfg["simulation"]["dt"],
        u_min=p["u_min"], u_max=p["u_max"],
        noise_std=noise_std, seed=seed)

