"""
仿真运行核心 (Simulation core)
=============================
把"被控对象 + 控制器 + 工况时序 + 指标计算"统一封装，供
``temperature_simulation.py``（单场景对比）与 ``experiments.py``（批量实验）复用，
保证两种语言、所有实验脚本的时间推进顺序完全一致：

    k 时刻：控制器读取量测 y(k)  →  计算 u(k)  →  对象按 u(k) 积分一步 → 得到 y(k+1)

数值积分统一采用显式欧拉法（与 Simulink 定步长离散求解器行为一致），
步长取自共享配置，所有实验的时间轴均为 ``t = 0, dt, 2dt, ...``（不含浮点漂移）。
"""

from __future__ import annotations

import numpy as np

from plants import thermal_plant_from_config, motor_plant_from_config, level_plant_from_config
from metrics import (response_metrics, whole_run_metrics, control_effort, _window)


# ---------------------------------------------------------------------------
# 信号发生
# ---------------------------------------------------------------------------
def time_vector(cfg, t_end=None):
    dt = float(cfg["simulation"]["dt"])
    t_end = float(cfg["simulation"]["t_end"] if t_end is None else t_end)
    n = int(round(t_end / dt)) + 1
    return np.arange(n) * dt


def piecewise_reference(t, segments):
    """分段常值设定信号：segments = [[t_start, value], ...]"""
    r = np.full_like(t, float(segments[0][1]), dtype=float)
    for t_start, value in segments[1:]:
        r[t >= float(t_start) - 1e-12] = float(value)
    return r


def step_signal(t, t_on, amplitude, t_off=None):
    d = np.zeros_like(t)
    m = t >= float(t_on) - 1e-12
    if t_off is not None:
        m &= t < float(t_off) - 1e-12
    d[m] = float(amplitude)
    return d


def scaled_step(t, t_on, scale):
    """阶跃式参数调度（如转动惯量突然放大 2.5 倍）。"""
    s = np.ones_like(t)
    s[t >= float(t_on) - 1e-12] = float(scale)
    return s


# ---------------------------------------------------------------------------
# 场景 1：工业电加热炉（FOPDT）
# ---------------------------------------------------------------------------
def run_thermal(controller, cfg, K=None, T=None, tau=None, noise_std=0.0,
                seed=0, t_end=None, disturbance_enable=True, kd_filter_tau=None,
                ctrl_period=None):
    """
    运行一次加热炉闭环仿真，返回全部时间序列。

    ctrl_period : 控制器采样/执行周期 (s)。给出时采用"零阶保持"多速率仿真：
                  对象仍以 dt 积分，控制器每 ctrl_period 秒才更新一次控制量，
                  用于研究 PLC 扫描周期对控制品质的影响。
    """
    t = time_vector(cfg, t_end)
    dt = float(cfg["simulation"]["dt"])
    r = piecewise_reference(t, cfg["reference"]["segments"])
    if disturbance_enable:
        dc = cfg["disturbance"]
        d = step_signal(t, dc["t_on"], dc["amplitude"], dc.get("t_off"))
    else:
        d = np.zeros_like(t)

    plant = thermal_plant_from_config(cfg, K=K, T=T, tau=tau,
                                      noise_std=noise_std, seed=seed)
    if kd_filter_tau is not None:
        controller.kd_filter_tau = float(kd_filter_tau)
    if ctrl_period is not None:
        controller.dt = float(ctrl_period)
        n_sub = max(1, int(round(float(ctrl_period) / dt)))
    else:
        n_sub = 1
    controller.reset()

    n = t.size
    res = {k: np.zeros(n) for k in ("y", "ym", "u", "e", "ec", "kp", "ki", "kd")}
    prev_e = 0.0
    u = 0.0
    for k in range(n):
        if k % n_sub == 0:
            ym = plant.measure()
            u = controller.compute(r[k], ym)
        plant.step(u, d[k])
        e = r[k] - plant.y
        res["y"][k] = plant.y
        res["ym"][k] = plant.y_meas
        res["u"][k] = u
        res["e"][k] = e
        res["ec"][k] = (e - prev_e) / dt if k else 0.0
        prev_e = e
        res["kp"][k] = controller.kp
        res["ki"][k] = controller.ki
        res["kd"][k] = controller.kd

    res.update(t=t, r=r, d=d, plan=plant)
    return res


def evaluate_thermal(res, cfg, label=""):
    """把加热炉仿真结果折算为一组标准性能指标（可直接写入 CSV）。"""
    t, y, r, u = res["t"], res["y"], res["r"], res["u"]
    p = cfg["phases"]
    p1, p2, p3 = p["step1"], p["dist"], p["step2"]
    y0 = float(cfg["plant"]["y0"])
    span1 = float(cfg["reference"]["segments"][0][1]) - y0

    m1 = response_metrics(t, y, r, p1[0], p1[1], y_base=y0, scale=span1)
    m2 = response_metrics(t, y, r, p2[0], p2[1], y_base=None, scale=span1)
    y_b2 = float(np.interp(p3[0], t, y))
    span2 = float(cfg["reference"]["segments"][-1][1]) - y_b2
    m3 = response_metrics(t, y, r, p3[0], p3[1], y_base=y_b2, scale=span2,
                          closed_end=True)
    allm = whole_run_metrics(t, y, r, u, cfg["plant"]["u_min"], cfg["plant"]["u_max"])

    return {
        "controller": label,
        "tr_s": m1["rise_time"],
        "sigma1_pct": m1["overshoot"],
        "ts1_s": m1["settling_time"],
        "ess1": m1["ess"],
        "peak1": m1["peak"],
        "dist_drop": m2["max_dev"],
        "dist_recovery_s": m2["recovery_time"],
        "dist_iae": m2["iae"],
        "sigma2_pct": m3["overshoot"],
        "ts2_s": m3["settling_time"],
        "ess2": m3["ess"],
        "tail_std2": m3["tail_std"],
        "itae_abs": allm["itae_abs"],
        "iae": allm["iae"],
        "tv": allm["tv"],
        "sat_duty_pct": allm["sat_duty"],
    }


# ---------------------------------------------------------------------------
# 场景 2：直流伺服电机
# ---------------------------------------------------------------------------
def run_motor(controller, cfg, J_scale_nominal=1.0, Ra=None, La=None, Kt=None,
              Ke=None, B=None, J0=None, noise_std=0.0, seed=0, t_end=None,
              load_enable=True, inertia_enable=True, kd_filter_tau=None):
    """运行一次直流伺服电机闭环仿真，返回全部时间序列。"""
    t = time_vector(cfg, t_end)
    dt = float(cfg["simulation"]["dt"])
    r = piecewise_reference(t, cfg["reference"]["segments"])
    if load_enable:
        lc = cfg["load_torque"]
        TL = step_signal(t, lc["t_on"], lc["amplitude"])
    else:
        TL = np.zeros_like(t)
    if inertia_enable:
        ic = cfg["inertia"]
        Jscale = scaled_step(t, ic["t_on"], ic["scale"]) * float(J_scale_nominal)
    else:
        Jscale = np.full_like(t, float(J_scale_nominal))

    plant = motor_plant_from_config(cfg, J0=J0, Ra=Ra, La=La, Kt=Kt, Ke=Ke, B=B,
                                    noise_std=noise_std, seed=seed)
    if kd_filter_tau is not None:
        controller.kd_filter_tau = float(kd_filter_tau)
    controller.reset()

    n = t.size
    res = {k: np.zeros(n) for k in ("y", "ym", "u", "e", "ec", "kp", "ki", "kd")}
    prev_e = 0.0
    for k in range(n):
        ym = plant.measure()
        u = controller.compute(r[k], ym)
        plant.step(u, load_torque=TL[k], J=plant.J0 * Jscale[k])
        e = r[k] - plant.omega
        res["y"][k] = plant.omega
        res["ym"][k] = ym
        res["u"][k] = u
        res["e"][k] = e
        res["ec"][k] = (e - prev_e) / dt if k else 0.0
        prev_e = e
        res["kp"][k] = controller.kp
        res["ki"][k] = controller.ki
        res["kd"][k] = controller.kd

    res.update(t=t, r=r, d=TL, jscale=Jscale, plan=plant)
    return res


def evaluate_motor(res, cfg, label=""):
    """把电机仿真结果折算为标准性能指标。"""
    t, y, r, u = res["t"], res["y"], res["r"], res["u"]
    p = cfg["phases"]
    p1, p2, p3 = p["step1"], p["load"], p["inertia_step"]
    span1 = float(cfg["reference"]["segments"][0][1])

    m1 = response_metrics(t, y, r, p1[0], p1[1], y_base=0.0, scale=span1)
    m2 = response_metrics(t, y, r, p2[0], p2[1], y_base=None, scale=span1)
    y_b3 = float(np.interp(p3[0], t, y))
    span3 = float(cfg["reference"]["segments"][-1][1]) - y_b3
    m3 = response_metrics(t, y, r, p3[0], p3[1], y_base=y_b3, scale=span3,
                          closed_end=True)
    allm = whole_run_metrics(t, y, r, u, cfg["plant"]["u_min"], cfg["plant"]["u_max"])

    return {
        "controller": label,
        "tr_s": m1["rise_time"],
        "sigma1_pct": m1["overshoot"],
        "ts1_s": m1["settling_time"],
        "ess1": m1["ess"],
        "dip_rad_s": m2["max_dev"],
        "dip_recovery_s": m2["recovery_time"],
        "dip_iae": m2["iae"],
        "sigma2_pct": m3["overshoot"],
        "ts2_s": m3["settling_time"],
        "ess2": m3["ess"],
        "tail_std2": m3["tail_std"],
        "itae_abs": allm["itae_abs"],
        "iae": allm["iae"],
        "tv": allm["tv"],
        "sat_duty_pct": allm["sat_duty"],
    }



# ---------------------------------------------------------------------------
# 场景 3：双容串联液位过程（SOPDT）
# ---------------------------------------------------------------------------
def run_level(controller, cfg, K=None, T1=None, T2=None, tau=None,
              noise_std=0.0, seed=0, t_end=None, disturbance_enable=True):
    """
    运行一次液位双容过程闭环仿真，返回全部时间序列。

    被控对象：G(s) = K / [(T1s+1)(T2s+1)] · e^{-τs}
    工况：20%→60% 阶跃，t=120 s 突加 +15% 进液扰动（30 s），t=230 s 降至 40%。
    """
    t = time_vector(cfg, t_end)
    r = piecewise_reference(t, cfg["reference"]["segments"])
    if disturbance_enable:
        dc = cfg["disturbance"]
        d = step_signal(t, dc["t_on"], dc["amplitude"], dc.get("t_off"))
    else:
        d = np.zeros_like(t)

    plant = level_plant_from_config(cfg, K=K, T1=T1, T2=T2, tau=tau,
                                    noise_std=noise_std, seed=seed)
    controller.reset()

    n = t.size
    res = {k: np.zeros(n) for k in ("y", "ym", "u", "e", "ec", "kp", "ki", "kd", "x1")}
    prev_e = 0.0
    dt = float(cfg["simulation"]["dt"])
    for k in range(n):
        ym = plant.measure()
        u = controller.compute(r[k], ym)
        plant.step(u, d[k])
        e = r[k] - plant.y
        res["y"][k] = plant.y
        res["x1"][k] = plant.x1        # 第一容腔液位（中间状态）
        res["ym"][k] = plant.y_meas
        res["u"][k] = u
        res["e"][k] = e
        res["ec"][k] = (e - prev_e) / dt if k else 0.0
        prev_e = e
        res["kp"][k] = controller.kp
        res["ki"][k] = controller.ki
        res["kd"][k] = controller.kd

    res.update(t=t, r=r, d=d, plan=plant)
    return res


def evaluate_level(res, cfg, label=""):
    """把液位仿真结果折算为标准性能指标。"""
    t, y, r, u = res["t"], res["y"], res["r"], res["u"]
    p = cfg["phases"]
    p1, p2, p3 = p["step1"], p["dist"], p["step2"]
    y0 = float(cfg["plant"]["y0"])
    span1 = float(cfg["reference"]["segments"][0][1]) - y0

    m1 = response_metrics(t, y, r, p1[0], p1[1], y_base=y0, scale=span1)
    m2 = response_metrics(t, y, r, p2[0], p2[1], y_base=None, scale=span1)
    y_b2 = float(np.interp(p3[0], t, y))
    span2 = float(cfg["reference"]["segments"][-1][1]) - y_b2
    m3 = response_metrics(t, y, r, p3[0], p3[1], y_base=y_b2, scale=span2,
                          closed_end=True)
    allm = whole_run_metrics(t, y, r, u, cfg["plant"]["u_min"], cfg["plant"]["u_max"])

    return {
        "controller": label,
        "tr_s": m1["rise_time"],
        "sigma1_pct": m1["overshoot"],
        "ts1_s": m1["settling_time"],
        "ess1": m1["ess"],
        "dist_drop": m2["max_dev"],
        "dist_recovery_s": m2["recovery_time"],
        "dist_iae": m2["iae"],
        "sigma2_pct": m3["overshoot"],
        "ts2_s": m3["settling_time"],
        "ess2": m3["ess"],
        "tail_std2": m3["tail_std"],
        "itae_abs": allm["itae_abs"],
        "iae": allm["iae"],
        "tv": allm["tv"],
        "sat_duty_pct": allm["sat_duty"],
    }


# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------
def segment(res, t0, t1):
    m = _window(res["t"], t0, t1)
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape == res["t"].shape else v)
            for k, v in res.items()}


def gain_bounds(cfg):
    """由基准增益与缩放因子推导在线整定的物理边界（防止增益跑飞）。"""
    p = cfg["pid"]
    return {
        "kp_min": 0.01, "kp_max": max(4.0 * p["kp0"], p["kp0"] + 3.0 * abs(p["kup"])),
        "ki_min": 0.0, "ki_max": max(3.0 * p["ki0"], p["ki0"] + 3.0 * abs(p["kui"])),
        "kd_min": 0.0, "kd_max": max(3.0 * p["kd0"], p["kd0"] + 3.0 * abs(p["kud"])),
    }
