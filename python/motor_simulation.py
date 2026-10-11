"""
场景二：直流伺服电机转速控制系统 —— Python 侧仿真与可视化
===========================================================
对象：直流电机机电耦合两状态模型（电枢回路 + 机械方程）
    La·dia/dt = ua - Ra·ia - Ke·ω
    J·dω/dt   = Kt·ia - B·ω - TL
    参数：Ra=1.2Ω, La=0.03H, Kt=Ke=0.08, B=0.005, J0=0.008 kg·m²

工况：
    0 ~ 2.0 s    → 转速 0→100 rad/s 起动阶段
    2.0~3.8 s    → 突加 0.25 N·m 负载力矩
    3.8~6.0 s    → 惯量×2.5 变惯量工况 + 升至 130 rad/s

本脚本输出 4 张图：
    M1  转速响应 / 负载力矩 / 电枢电流 / 跟踪误差（四联图）
    M2  相平面轨迹对比（超调区阴影）
    M3  模糊自适应增益整定轨迹
    M4  蒙特卡洛鲁棒性箱线图（机电参数 ±20% 50 组工况）

运行：  python python/motor_simulation.py
"""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
ROOT = os.path.dirname(CURRENT_DIR)
RESULTS = os.path.join(ROOT, "results")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from fuzzy_config import scenario
from simcore import run_motor, evaluate_motor, time_vector
from metrics import print_table, write_csv, relative_gain
from controllers import build_set, SPECS, DEFAULT_ORDER, build
from tuning import load_tuned

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 300

FIG_MAIN = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy"]
FIG_FUZZY = ["fuzzy", "fuzzy_opt"]


# ---------------------------------------------------------------------------
def run_all_motor(cfg, tuned, keys=DEFAULT_ORDER):
    data = {}
    for key, label, ctrl in build_set("motor", keys, tuned=tuned):
        res = run_motor(ctrl, cfg)
        data[key] = dict(res=res, metrics=evaluate_motor(res, cfg, label=label),
                         label=label, ctrl=ctrl)
    return data


def report_motor(data, cfg):
    rows, table = [], []
    for key in DEFAULT_ORDER:
        if key not in data:
            continue
        m = data[key]["metrics"]
        rows.append(dict(m))
        table.append([
            m["controller"],
            f"{m['tr_s']:.3f}", f"{m['sigma1_pct']:.2f}", f"{m['ts1_s']:.3f}",
            f"{m['dip_rad_s']:.3f}", f"{m['dip_recovery_s']:.3f}",
            f"{m['sigma2_pct']:.2f}", f"{m['ts2_s']:.3f}", f"{m['itae_abs']:.1f}",
            f"{m['tv']:.1f}", f"{m['sat_duty_pct']:.1f}",
        ])

    print("\n" + "=" * 118)
    print(" 表 M1  直流伺服电机多工况闭环性能指标总表（Python侧仿真）")
    print("=" * 118)
    print_table("", ["控制器", "tr/s", "σ1/%", "ts1/s", "跌落/(rad/s)",
                     "恢复/s", "σ2/%", "ts2/s", "ITAE", "TV(u)", "饱和/%"], table)

    ref = data.get("pid_opt") or data["pid_aw"]
    fz  = data.get("fuzzy_opt") or data["fuzzy"]
    a, b = ref["metrics"], fz["metrics"]

    def fmt_rel(x, y):
        v = relative_gain(x, y)
        return "—" if not np.isfinite(v) else f"{v:+.1f}%"

    print("\n 表 M2  综合自适应收益（对照 = %s，实验 = %s）" % (ref["label"], fz["label"]))
    print_table("", ["指标", "对照组", "实验组", "改善"],
                [["起动超调 σ1 (%)", f"{a['sigma1_pct']:.2f}", f"{b['sigma1_pct']:.2f}",
                  fmt_rel(a['sigma1_pct'], b['sigma1_pct'])],
                 ["起动调节时间 ts1 (s)", f"{a['ts1_s']:.3f}", f"{b['ts1_s']:.3f}",
                  fmt_rel(a['ts1_s'], b['ts1_s'])],
                 ["负载跌落 (rad/s)", f"{a['dip_rad_s']:.3f}", f"{b['dip_rad_s']:.3f}",
                  fmt_rel(a['dip_rad_s'], b['dip_rad_s'])],
                 ["变惯量超调 σ2 (%)", f"{a['sigma2_pct']:.2f}", f"{b['sigma2_pct']:.2f}",
                  fmt_rel(a['sigma2_pct'], b['sigma2_pct'])],
                 ["全程 ITAE", f"{a['itae_abs']:.1f}", f"{b['itae_abs']:.1f}",
                  fmt_rel(a['itae_abs'], b['itae_abs'])]])

    write_csv(os.path.join(RESULTS, "motor_metrics_python.csv"), rows)
    return rows


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def figure_M1_main(data, cfg):
    """电机转速四联主图"""
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    r = data[DEFAULT_ORDER[0]]["res"]["r"]
    TL = data[DEFAULT_ORDER[0]]["res"]["d"]
    ph = cfg["phases"]
    fig, axs = plt.subplots(4, 1, figsize=(11.5, 13), sharex=True)

    for ax in axs:
        ax.axvspan(*ph["step1"],          color="#f4f6f9", zorder=0)
        ax.axvspan(*ph["load"],           color="#fff2f2", zorder=0)
        ax.axvspan(*ph["inertia_step"],   color="#f2f8fc", zorder=0)
        ax.axvline(ph["load"][0],         color="crimson",  ls="--", lw=1.2, alpha=0.85)
        ax.axvline(ph["inertia_step"][0], color="darkblue", ls="--", lw=1.2, alpha=0.85)
        ax.grid(True, ls="--", alpha=0.45)

    # 子图1：转速响应
    ax = axs[0]
    ax.plot(t, r, "r--", lw=1.8, label="转速设定 r(t)")
    for key in FIG_MAIN:
        if key not in data:
            continue
        m = data[key]["metrics"]
        ax.plot(t, data[key]["res"]["y"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=2.0,
                label=f"{data[key]['label']}（σ1={m['sigma1_pct']:.2f}%, ts1={m['ts1_s']:.3f}s）")
    ax.set_ylabel("转速 ω (rad/s)", fontweight="bold")
    ax.set_title("图 M1  直流伺服电机多工况转速响应对比（起动+负载扰动+变惯量工况）",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-20, 165)
    ax.text(1.0,  157, "阶段1：起动 0→100 rad/s", ha="center", fontsize=9, fontweight="bold")
    ax.text(2.9,  157, "阶段2：负载扰动", ha="center", fontsize=9, color="crimson", fontweight="bold")
    ax.text(4.9,  157, "阶段3：变惯量 →130 rad/s", ha="center", fontsize=9, color="darkblue", fontweight="bold")

    # 子图2：负载力矩
    ax = axs[1]
    ax.plot(t, TL, color="crimson", lw=2.2, label="负载力矩 TL(t)")
    ax.fill_between(t, TL, 0, where=(TL > 0), color="crimson", alpha=0.25)
    ax.set_ylabel("负载力矩 (N·m)", fontweight="bold")
    ax.set_ylim(-0.02, 0.35)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"外部扰动：t={cfg['load_torque']['t_on']:.1f} s 突加 "
                 f"{cfg['load_torque']['amplitude']:.2f} N·m 负载，t={cfg['inertia']['t_on']:.1f} s 惯量×{cfg['inertia']['scale']:.1f}",
                 fontsize=11)

    # 子图3：电枢电流（仅电机有这个状态量）
    ax = axs[2]
    for key in FIG_MAIN:
        if key not in data:
            continue
        # 通过植物对象间接拿到电流时序——重新计算
        pass
    # 电流需要从控制量估算 or 直接使用控制量（电压ua）替代显示
    for key in FIG_MAIN:
        if key not in data:
            continue
        ax.plot(t, data[key]["res"]["u"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=1.8,
                label=f"{data[key]['label']}（TV={data[key]['metrics']['tv']:.1f}）")
    ax.axhline(cfg["plant"]["u_max"], color="k", lw=1.0, ls=":", alpha=0.6)
    ax.axhline(cfg["plant"]["u_min"], color="k", lw=1.0, ls=":", alpha=0.6)
    ax.set_ylabel("电枢电压 ua (V)", fontweight="bold")
    ax.set_ylim(-38, 40)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("控制量（电枢电压）u(t)，含正负限幅 ±36 V", fontsize=11)

    # 子图4：跟踪误差
    ax = axs[3]
    for key in FIG_MAIN:
        if key not in data:
            continue
        ax.semilogy(t, np.maximum(np.abs(data[key]["res"]["e"]), 1e-3),
                    color=SPECS[key]["color"], ls=SPECS[key]["linestyle"], lw=1.8,
                    label=f"{data[key]['label']}（ITAE={data[key]['metrics']['itae_abs']:.1f}）")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("时间 t (s)", fontweight="bold")
    ax.set_ylabel("|e(t)| (rad/s, 对数轴)", fontweight="bold")
    ax.set_title("跟踪误差衰减（对数坐标）", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, cfg["simulation"]["t_end"])

    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "M1_motor_response_main.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 M1 已保存: {path}")
    return path


def figure_M2_phase(data, cfg):
    """相平面轨迹（超调区阴影）"""
    keys = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, key in zip(axs.ravel(), keys):
        if key not in data:
            ax.set_visible(False)
            continue
        res = data[key]["res"]
        # 超调区（e < 0）
        ax.axhspan(-30, 0, color="red", alpha=0.06, zorder=0, label="超调区 (e<0)")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.plot(res["e"], res["ec"], color=SPECS[key]["color"], lw=1.6,
                label=f"{data[key]['label']} 相轨迹")
        ax.scatter([res["e"][0]], [res["ec"][0]], color="darkblue", s=80, zorder=5,
                   label="起点 (e=100)")
        ax.scatter([0], [0], color="red", marker="*", s=170, zorder=5, label="稳态原点")
        m = data[key]["metrics"]
        ax.set_title(f"({key}) {data[key]['label']}：σ1={m['sigma1_pct']:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("误差 e (rad/s)", fontweight="bold")
        ax.set_ylabel("误差变化率 ec (rad/s²)", fontweight="bold")
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("图 M2  电机相平面轨迹对比（红色区域为超调区，轨迹进入说明转速过冲）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "M2_motor_phase_plane.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 M2 已保存: {path}")
    return path


def figure_M3_gains(data, cfg):
    """模糊自适应增益整定轨迹"""
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    ph = cfg["phases"]
    names = ["比例增益 Kp(t)", "积分增益 Ki(t)", "微分增益 Kd(t)"]
    keys_map = [("kp", "kp0"), ("ki", "ki0"), ("kd", "kd0")]
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
    spans = [(ph["step1"], "#f4f6f9"), (ph["load"], "#fff2f2"), (ph["inertia_step"], "#f2f8fc")]
    for ax in axs:
        for (t0, t1), color in spans:
            ax.axvspan(t0, t1, color=color, zorder=0)
        ax.axvline(ph["load"][0],         color="crimson",  ls="--", lw=1.1, alpha=0.8)
        ax.axvline(ph["inertia_step"][0], color="darkblue", ls="--", lw=1.1, alpha=0.8)
        ax.grid(True, ls="--", alpha=0.45)

    for i, (gkey, base_key) in enumerate(keys_map):
        for key in FIG_FUZZY:
            if key not in data:
                continue
            c = data[key]["ctrl"]
            axs[i].plot(t, data[key]["res"][gkey], color=SPECS[key]["color"], lw=1.9,
                        label=f"{data[key]['label']}")
            axs[i].axhline(getattr(c, base_key), color=SPECS[key]["color"], ls=":",
                           lw=1.1, alpha=0.8,
                           label=f"基准值 {base_key}={getattr(c, base_key):g}")
        axs[i].set_ylabel(names[i], fontweight="bold")
    axs[0].set_title("图 M3  电机系统模糊自适应 PID 在线增益整定轨迹（Python 侧仿真）",
                     fontsize=13, fontweight="bold")
    axs[2].set_xlabel("时间 t (s)", fontweight="bold")
    axs[0].legend(loc="upper right", fontsize=8.5, ncol=2)
    for ax in axs:
        ax.set_xlim(0, cfg["simulation"]["t_end"])
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "M3_motor_gain_adaptation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 M3 已保存: {path}")
    return path


def figure_M4_mc_boxplot(cfg, tuned, n=80, seed=20240923, mismatch=0.20):
    """机电参数 ±20% 蒙特卡洛鲁棒性箱线图"""
    rng = np.random.default_rng(seed)
    p = cfg["plant"]
    keys_mc = ["pid_opt", "fuzzy", "fuzzy_opt"]
    rows = []
    print(f"\n[E1M-PY] 电机蒙特卡洛：{n} 组工况 (Ra/La/Kt/Ke/B/J0 各 ±{mismatch*100:.0f}%) ...")
    for i in range(n):
        f = 1.0 + rng.uniform(-mismatch, mismatch, size=6)
        Ra, La, Kt, Ke, B, J0 = (p["Ra"] * f[0], p["La"] * f[1], p["Kt"] * f[2],
                                   p["Ke"] * f[3], p["B"] * f[4], p["J0"] * f[5])
        for key in keys_mc:
            ctrl = build("motor", key, tuned=tuned)
            res = run_motor(ctrl, cfg, Ra=Ra, La=La, Kt=Kt, Ke=Ke, B=B, J0=J0)
            m = evaluate_motor(res, cfg, label=SPECS[key]["label"])
            rows.append(dict(run=i, key=key, **m))
        if (i + 1) % 20 == 0:
            print(f"  已完成 {i+1}/{n} 组")

    # 画箱线图
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.5))
    for ax, metric, title in zip(axs,
                                  ["sigma1_pct", "dip_rad_s", "itae_abs"],
                                  ["起动超调 σ1 (%)", "负载跌落 (rad/s)", "全程 ITAE"]):
        box_data = []
        labels_mc = []
        for key in keys_mc:
            vals = [r[metric] for r in rows if r["key"] == key]
            box_data.append(vals)
            labels_mc.append(SPECS[key]["label"].replace("模糊自适应PID", "模糊").replace("PID-ITAE最优", "PID-ITAE"))
        bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                        medianprops=dict(color="red", lw=2.0))
        for patch, key in zip(bp["boxes"], keys_mc):
            patch.set_facecolor(SPECS[key]["color"])
            patch.set_alpha(0.6)
        ax.set_xticklabels(labels_mc, fontsize=9, rotation=15, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        ax.set_ylabel(title, fontweight="bold")

    fig.suptitle(f"图 M4  电机机电参数 ±{mismatch*100:.0f}% 摄动蒙特卡洛（{n} 组）鲁棒性箱线图\n"
                 f"（箱体=四分位区间，红线=中位数，须=1.5×IQR）",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "M4_motor_mc_boxplot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图 M4 已保存: {path}")
    return path


# ---------------------------------------------------------------------------
def main():
    cfg = scenario("motor")
    tuned = load_tuned()

    print("=" * 118)
    print(" 场景二：直流伺服电机（Ra=1.2Ω, La=0.03H, Kt=Ke=0.08, J0=0.008 kg*m^2）Python侧仿真")
    print(f" 工况: {cfg['simulation']['t_end']:.1f} s, dt={cfg['simulation']['dt']} s, "
          f"0→100 rad/s 起动, TL=0.25 N·m @{cfg['load_torque']['t_on']:.1f} s, "
          f"J×{cfg['inertia']['scale']:.1f} @{cfg['inertia']['t_on']:.1f} s → 130 rad/s")
    print("=" * 118)

    data = run_all_motor(cfg, tuned)
    report_motor(data, cfg)

    figure_M1_main(data, cfg)
    figure_M2_phase(data, cfg)
    figure_M3_gains(data, cfg)
    figure_M4_mc_boxplot(cfg, tuned)

    print("\n[完成] 电机场景全部图表已保存。\n")


if __name__ == "__main__":
    main()
