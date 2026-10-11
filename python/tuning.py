"""
参数自动整定 (Automatic tuning)
==============================
为了让"模糊自适应 PID 优于常规 PID"这一结论**站得住脚**，必须避免"拿调好的
模糊控制器去比一个随手写的固定 PID"。本模块对两种控制器采用同一目标函数、
同一优化器（Nelder-Mead 单纯形）、同一迭代预算，各自在自己的**自然参数集**上寻优：

- 固定参数 PID：3 个参数 (Kp, Ki, Kd)；
- 模糊自适应 PID：6 个参数 (Kp0, Ki0, Kd0, kup, kui, kud)。

关于"公平性"的两点说明（必须在报告中披露）：
1. 参数集大小不同看似对模糊控制不利，实则相反——当缩放因子取 0 时模糊控制器
   恰好退化为固定增益 PID，因此它的可行域**包含**固定 PID；模糊侧多出的 3 个
   参数正是"自适应权限"本身。为此对缩放因子设置非零下界，要求它必须保留最低限度的
   在线整定能力，不能靠"退化"来偷懒。
2. 目标函数是**标称工况**的全程 ITAE。在标称工况上调参必然存在过拟合风险，
   因此任何"模糊更优"的结论都必须以 experiments.py 中的蒙特卡洛鲁棒性实验
   （参数 ±30% 随机摄动、300 组工况）为主要证据，标称工况只作为必要条件。

寻优结果写入 ``results/tuned_params.json``，供 Python 与 MATLAB 两侧共用。
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize

from fuzzy_config import scenario
from fuzzy_pid import ConventionalPID, FuzzyAdaptivePID
from simcore import run_thermal, evaluate_thermal, run_motor, evaluate_motor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TUNED_PATH = os.path.join(ROOT, "results", "tuned_params.json")

# 搜索空间（依据配置中基准参数的量级设定，均为物理可实现的增益范围）
#
# 模糊控制器参数顺序: (Kp0, Ki0, Kd0, kup, kui, kud)
# 其中缩放因子设有**非零下界**：若允许其取 0，则模糊控制器会退化为固定增益 PID，
# 也就失去了"自适应"这一被考察的对象本身。下界即"最低自适应权限"约束。
BOUNDS = {
    "thermal": {
        "pid":   [(0.2, 30.0), (0.002, 1.0), (0.0, 60.0)],
        "fuzzy": [(0.5, 12.0), (0.005, 0.50), (0.0, 30.0),
                  (0.4, 6.0), (0.010, 0.30), (0.5, 20.0)],
    },
    "motor": {
        "pid":   [(0.05, 20.0), (1.0, 200.0), (0.0, 0.10)],
        "fuzzy": [(0.10, 8.0), (2.0, 120.0), (0.0, 0.05),
                  (0.10, 3.0), (0.5, 60.0), (0.001, 0.06)],
    },
}

PARAM_NAMES = {"pid": ["kp", "ki", "kd"],
               "fuzzy": ["kp0", "ki0", "kd0", "kup", "kui", "kud"]}

_RUNNERS = {
    "thermal": (run_thermal, evaluate_thermal),
    "motor": (run_motor, evaluate_motor),
}


def load_tuned(path=None):
    path = path or TUNED_PATH
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_tuned(data, path=None):
    path = path or TUNED_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _make(scenario_name, kind, params):
    cfg = scenario(scenario_name)
    p = cfg["pid"]
    dt = cfg["simulation"]["dt"]
    common = dict(out_min=cfg["plant"]["u_min"], out_max=cfg["plant"]["u_max"],
                  dt=dt, anti_windup=True)
    if kind == "pid":
        kp, ki, kd = params
        return ConventionalPID(kp, ki, kd, **common)
    kp0, ki0, kd0, kup, kui, kud = params
    return FuzzyAdaptivePID(kp0, ki0, kd0, ke=p["ke"], kec=p["kec"],
                            kup=kup, kui=kui, kud=kud, **common)


def cost(scenario_name, kind, params, verbose=False):
    """优化目标：标称工况全程 ITAE。发散的参数组合返回极大惩罚值。"""
    cfg = scenario(scenario_name)
    run, evaluate = _RUNNERS[scenario_name]
    lo_hi = BOUNDS[scenario_name][kind]
    for x, (lo, hi) in zip(params, lo_hi):
        if not (lo <= x <= hi):
            return 1e12
    try:
        ctrl = _make(scenario_name, kind, params)
        res = run(ctrl, cfg)
        m = evaluate(res, cfg)
        if not np.isfinite(m["itae_abs"]) or m["itae_abs"] > 1e9:
            return 1e12
        if verbose:
            print(f"    ITAE={m['itae_abs']:.1f} σ1={m['sigma1_pct']:.2f}% "
                  f"ts1={m['ts1_s']:.1f}s 参数={np.round(params, 4)}")
        return float(m["itae_abs"])
    except Exception as exc:  # 数值发散
        if verbose:
            print("    失败:", exc)
        return 1e12


def tune(scenario_name, kind, x0, maxiter=90, seed=0, restarts=2):
    """Nelder-Mead 单纯形寻优（含随机重启，避免陷入局部极小）。"""
    bounds = BOUNDS[scenario_name][kind]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    total = maxiter * (restarts + 1)
    print(f"    [{scenario_name}/{kind}] {len(x0)} 参数, 单纯形上限 {total} 次仿真评估, "
          f"起点 {np.round(x0, 4)}")
    rng = np.random.default_rng(seed)
    best_x, best_f = np.asarray(x0, dtype=float), cost(scenario_name, kind, x0)

    starts = [np.asarray(x0, dtype=float)]
    for _ in range(restarts):
        starts.append(np.clip(rng.uniform(lo, hi), lo, hi))

    for si, s in enumerate(starts):
        if cost(scenario_name, kind, s) >= 1e12:
            continue

        def objective(x, _s=scenario_name, _k=kind):
            return cost(_s, _k, x)

        res = minimize(objective, s, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2))
        x = np.clip(res.x, lo, hi)
        f = cost(scenario_name, kind, x)
        print(f"      重启 {si+1}/{len(starts)}: 迭代 {res.nit} 次, 目标 {f:.1f}")
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f


def tune_all(verbose=True):
    out = load_tuned()
    plan = {"thermal": 250, "motor": 200}
    for scen in ("thermal", "motor"):
        cfg = scenario(scen)
        p = cfg["pid"]
        jobs = [
            ("pid", [p["kp0"], p["ki0"], p["kd0"]]),
            ("fuzzy", [p["kp0"], p["ki0"], p["kd0"], p["kup"], p["kui"], p["kud"]]),
        ]
        for kind, x0 in jobs:
            key = f"{scen}_{kind}"
            x, f = tune(scen, kind, x0, maxiter=plan[scen])
            value = {name: float(v) for name, v in zip(PARAM_NAMES[kind], x)}
            value["itae"] = float(f)
            value["start"] = [float(v) for v in x0]
            out[key] = value
            if verbose:
                print(f"[整定] {key:14s} {np.round(x, 4)}  ITAE={f:.1f}")
    save_tuned(out)
    return out


if __name__ == "__main__":
    print("=" * 74)
    print(" 基于标称工况 ITAE 的等预算参数自动整定（固定PID 3参数 / 模糊3缩放因子）")
    print("=" * 74)
    tuned = tune_all()
    print("\n结果已写入:", TUNED_PATH)
