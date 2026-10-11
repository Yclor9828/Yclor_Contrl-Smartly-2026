"""
控制器装配表 (Controller registry)
=================================
把"实验里要对比哪几个控制器"这件事集中定义，保证主场景脚本、批量实验脚本、
MATLAB 侧脚本使用同一套对照组命名与配色：

======================  ==========================================================
对照组                  说明
======================  ==========================================================
``pid_nowindup``        常规固定 PID，朴素积分（无抗积分饱和）——复现积分饱和现象
``pid_aw``              常规固定 PID + 条件积分抗饱和——与模糊控制"同等工程条件"
``pid_opt``             常规固定 PID，在标称工况按 ITAE 数值寻优后的最强固定参数组
``fuzzy``               模糊自适应 PID（缩放因子取配置默认值）
``fuzzy_opt``           模糊自适应 PID（缩放因子按同一目标、同等预算寻优）
======================  ==========================================================

只有把"最强固定 PID"作为对照，模糊自适应的价值（尤其是在参数摄动下的价值）
才有说服力。
"""

from __future__ import annotations

from fuzzy_config import scenario
from fuzzy_pid import ConventionalPID, FuzzyAdaptivePID

SPECS = {
    "pid_nowindup": dict(
        label="PID-无抗饱和", kind="pid", anti_windup=False,
        color="#1f77b4", linestyle="--", note="朴素积分，饱和后积分持续累加"),
    "pid_aw": dict(
        label="PID-抗饱和", kind="pid", anti_windup=True,
        color="#ff7f0e", linestyle="-.", note="条件积分抗饱和"),
    "pid_opt": dict(
        label="PID-ITAE最优", kind="pid", anti_windup=True, tuned="pid",
        color="#9467bd", linestyle=":", note="标称工况数值寻优"),
    "fuzzy": dict(
        label="模糊自适应PID", kind="fuzzy", anti_windup=True,
        color="#2ca02c", linestyle="-", note="配置默认缩放因子"),
    "fuzzy_opt": dict(
        label="模糊自适应PID-优化因子", kind="fuzzy", anti_windup=True, tuned="fuzzy",
        color="#d62728", linestyle="-", note="同一目标寻优缩放因子"),
}

DEFAULT_ORDER = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy", "fuzzy_opt"]
MAIN_ORDER = ["pid_aw", "pid_opt", "fuzzy"]


def build(scen_name, key, tuned=None, anti_windup=None, kd_filter_tau=0.0,
          scaling=None, out_limits=None):
    """
    按 key 构造控制器实例。

    参数
    ----
    tuned        : results/tuned_params.json 的内容（提供"最优对照组"参数）
    anti_windup  : 覆盖默认的抗积分饱和开关
    kd_filter_tau: 微分一阶滤波时间常数 (s)
    scaling      : 仅对模糊控制器生效，覆盖 {kup, kui, kud}（用于敏感性扫描）
    out_limits   : 覆盖执行机构限幅 (u_min, u_max)
    """
    cfg = scenario(scen_name)
    spec = SPECS[key]
    p = cfg["pid"]
    dt = cfg["simulation"]["dt"]
    t = (tuned or {}).get(f"{scen_name}_{spec['tuned']}", {}) if spec.get("tuned") else {}

    u_min, u_max = cfg["plant"]["u_min"], cfg["plant"]["u_max"]
    if out_limits is not None:
        u_min, u_max = out_limits
    common = dict(
        out_min=u_min, out_max=u_max, dt=dt,
        anti_windup=spec["anti_windup"] if anti_windup is None else bool(anti_windup),
        kd_filter_tau=float(kd_filter_tau),
    )

    if spec["kind"] == "pid":
        ctrl = ConventionalPID(
            t.get("kp", p["kp0"]), t.get("ki", p["ki0"]), t.get("kd", p["kd0"]), **common)
    else:
        sc = dict(kup=t.get("kup", p["kup"]), kui=t.get("kui", p["kui"]),
                  kud=t.get("kud", p["kud"]))
        if scaling:
            sc.update({k: float(v) for k, v in scaling.items()})
        ctrl = FuzzyAdaptivePID(
            t.get("kp0", p["kp0"]), t.get("ki0", p["ki0"]), t.get("kd0", p["kd0"]),
            ke=p["ke"], kec=p["kec"],
            kup=sc["kup"], kui=sc["kui"], kud=sc["kud"], **common)
    ctrl.key = key
    ctrl.label = spec["label"]
    ctrl.note = spec["note"]
    return ctrl


def build_set(scen_name, keys, tuned=None, **overrides):
    """按顺序构造一组控制器，返回 [(key, label, controller), ...]。"""
    out = []
    for key in keys:
        ctrl = build(scen_name, key, tuned=tuned, **overrides)
        out.append((key, SPECS[key]["label"], ctrl))
    return out


def tuned_missing(tuned, scen_name, keys):
    """检查本组实验需要的寻优参数是否齐全。"""
    miss = []
    for key in keys:
        need = SPECS[key].get("tuned")
        if need and f"{scen_name}_{need}" not in (tuned or {}):
            miss.append(key)
    return miss
