"""
场景三：双容串联液位过程控制系统 —— 多维度仿真与实验分析
==========================================================
对象：二阶惯性 + 纯滞后 (SOPDT)
    G(s) = K / [(T1·s+1)(T2·s+1)] · e^{-τs}
    参数：K=0.96, T1=30 s, T2=15 s, τ=3 s
    物理背景：工业双容串联液位槽（上下游两个容器），控制量为进液阀门开度。

工况：
    0 ~ 120 s   → 液位 20%→60% 阶跃升液
    120~150 s   → 突加 +15% 进液扰动（额外进液流量冲击）
    230~350 s   → 液位设定降至 40%（回液阶段）

本脚本输出 5 张图：
    L1  液位响应 / 扰动 / 控制量 / 跟踪误差对比（四联图）
    L2  相平面轨迹对比（e-ec 平面，超调区阴影）
    L3  模糊自适应增益整定轨迹
    L4  第一容腔中间液位 x1(t) 对比（验证串级特性）
    L5  多控制器综合性能雷达图

运行：  python python/level_simulation.py
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
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

from fuzzy_config import scenario, fuzzy_block
from fuzzy_pid import FuzzyInferenceSystem
from simcore import run_level, evaluate_level, time_vector
from metrics import print_table, write_csv, relative_gain
from controllers import build_set, SPECS, DEFAULT_ORDER, tuned_missing
from tuning import load_tuned

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 300

FIG_MAIN = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy"]
FIG_FUZZY = ["fuzzy", "fuzzy_opt"]


# ---------------------------------------------------------------------------
def run_all_level(cfg, tuned, keys=DEFAULT_ORDER):
    data = {}
    for key, label, ctrl in build_set("level", keys, tuned=None):  # 液位场景暂不加载寻优参数
        res = run_level(ctrl, cfg)
        data[key] = dict(res=res, metrics=evaluate_level(res, cfg, label=label),
                         label=label, ctrl=ctrl)
    return data


def report_level(data, cfg):
    rows, table = [], []
    for key in DEFAULT_ORDER:
        if key not in data:
            continue
        m = data[key]["metrics"]
        rows.append(dict(m))
        table.append([
            m["controller"],
            f"{m['tr_s']:.2f}", f"{m['sigma1_pct']:.2f}", f"{m['ts1_s']:.1f}",
            f"{m['ess1']:+.3f}", f"{m['dist_drop']:.2f}", f"{m['dist_recovery_s']:.1f}",
            f"{m['sigma2_pct']:.2f}", f"{m['ts2_s']:.1f}", f"{m['itae_abs']:.0f}",
            f"{m['tv']:.0f}", f"{m['sat_duty_pct']:.1f}",
        ])

    print("\n" + "=" * 118)
    print(" 表 L1  双容液位过程多工况闭环性能指标总表")
    print("=" * 118)
    print_table("", ["控制器", "tr/s", "σ1/%", "ts1/s", "ess1/%", "扰动冲高/%",
                     "恢复/s", "σ2/%", "ts2/s", "ITAE", "TV(u)", "饱和/%"], table)

    ref = data.get("pid_opt") or data["pid_aw"]
    fz = data.get("fuzzy_opt") or data["fuzzy"]
    a, b = ref["metrics"], fz["metrics"]

    def fmt_rel(x, y):
        v = relative_gain(x, y)
        return "—（基准为 0）" if not np.isfinite(v) else f"{v:+.1f}%"

    print("\n 表 L2  综合自适应收益（对照组 = %s，实验组 = %s）" % (ref["label"], fz["label"]))
    print_table("", ["指标", "对照组", "实验组", "改善"],
                [["初始超调 σ1 (%)", f"{a['sigma1_pct']:.2f}", f"{b['sigma1_pct']:.2f}",
                  fmt_rel(a['sigma1_pct'], b['sigma1_pct'])],
                 ["初始调节时间 ts1 (s)", f"{a['ts1_s']:.1f}", f"{b['ts1_s']:.1f}",
                  fmt_rel(a['ts1_s'], b['ts1_s'])],
                 ["扰动冲高 (%液位)", f"{a['dist_drop']:.2f}", f"{b['dist_drop']:.2f}",
                  fmt_rel(a['dist_drop'], b['dist_drop'])],
                 ["二次跌落超调 σ2 (%)", f"{a['sigma2_pct']:.2f}", f"{b['sigma2_pct']:.2f}",
                  fmt_rel(a['sigma2_pct'], b['sigma2_pct'])],
                 ["全程 ITAE", f"{a['itae_abs']:.0f}", f"{b['itae_abs']:.0f}",
                  fmt_rel(a['itae_abs'], b['itae_abs'])]])

    write_csv(os.path.join(RESULTS, "level_metrics.csv"), rows)
    return rows


def dump_level_timeseries(data, cfg):
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    cols = {"t": t,
            "r": data[DEFAULT_ORDER[0]]["res"]["r"],
            "d": data[DEFAULT_ORDER[0]]["res"]["d"]}
    for key in DEFAULT_ORDER:
        if key in data:
            cols[f"y_{key}"] = data[key]["res"]["y"]
            cols[f"u_{key}"] = data[key]["res"]["u"]
    path = os.path.join(RESULTS, "level_timeseries_python.csv")
    header = ",".join(cols)
    arr = np.column_stack([cols[c] for c in cols])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.10g")
    return path


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def figure_L1_main(data, cfg):
    """液位响应四联图"""
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    r = data[DEFAULT_ORDER[0]]["res"]["r"]
    d = data[DEFAULT_ORDER[0]]["res"]["d"]
    ph = cfg["phases"]
    fig, axs = plt.subplots(4, 1, figsize=(11.5, 13), sharex=True)

    for ax in axs:
        ax.axvspan(*ph["step1"], color="#f4f6f9", zorder=0)
        ax.axvspan(*ph["dist"],  color="#fff8f2", zorder=0)
        ax.axvspan(*ph["step2"], color="#f2f8fc", zorder=0)
        ax.axvline(ph["dist"][0],  color="#c05020", ls="--", lw=1.2, alpha=0.85)
        ax.axvline(ph["step2"][0], color="darkblue", ls="--", lw=1.2, alpha=0.85)
        ax.grid(True, ls="--", alpha=0.45)

    # 子图1：液位响应
    ax = axs[0]
    ax.plot(t, r, "r--", lw=1.8, label="目标液位设定 r(t)")
    ax.fill_between(t, r - 1.2, r + 1.2, color="gray", alpha=0.18, label="±1.2% 误差带")
    for key in FIG_MAIN:
        if key not in data:
            continue
        m = data[key]["metrics"]
        ax.plot(t, data[key]["res"]["y"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=2.0,
                label=f"{data[key]['label']}（σ1={m['sigma1_pct']:.2f}%, ts1={m['ts1_s']:.1f}s）")
    ax.set_ylabel("液位 (%)", fontweight="bold")
    ax.set_title("图 L1  双容液位过程多工况响应对比（SOPDT: K=0.96, T1=30 s, T2=15 s, τ=3 s）",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(10, 80)
    ax.text(60,  75, "阶段1：20→60% 升液", ha="center", fontsize=9, fontweight="bold")
    ax.text(175, 75, "阶段2：进液冲击扰动", ha="center", fontsize=9, color="#c05020", fontweight="bold")
    ax.text(290, 75, "阶段3：60→40% 降液", ha="center", fontsize=9, color="darkblue", fontweight="bold")

    # 子图2：扰动信号
    ax = axs[1]
    ax.plot(t, d, color="#c05020", lw=2.2, label="进液扰动量 d(t)")
    ax.fill_between(t, d, 0, where=(d > 0), color="#c05020", alpha=0.25)
    ax.set_ylabel("扰动 (% 阀门当量)", fontweight="bold")
    ax.set_ylim(-2, 22)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"外部扰动时序：t={cfg['disturbance']['t_on']:.0f} s 起叠加 "
                 f"+{cfg['disturbance']['amplitude']:.0f}% 进液冲击，至 t={cfg['disturbance']['t_off']:.0f} s 结束",
                 fontsize=11)

    # 子图3：控制量
    ax = axs[2]
    for key in FIG_MAIN:
        if key not in data:
            continue
        ax.plot(t, data[key]["res"]["u"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=1.8,
                label=f"{data[key]['label']}（TV={data[key]['metrics']['tv']:.0f}）")
    ax.axhline(cfg["plant"]["u_max"], color="k", lw=1.0, ls=":")
    ax.set_ylabel("阀门开度 (%)", fontweight="bold")
    ax.set_ylim(-5, 108)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("执行机构控制量 u(t)（0~100% 阀门开度，含限幅）", fontsize=11)

    # 子图4：误差衰减（对数坐标）
    ax = axs[3]
    for key in FIG_MAIN:
        if key not in data:
            continue
        ax.semilogy(t, np.maximum(np.abs(data[key]["res"]["e"]), 1e-3),
                    color=SPECS[key]["color"], ls=SPECS[key]["linestyle"], lw=1.8,
                    label=f"{data[key]['label']}（ITAE={data[key]['metrics']['itae_abs']:.0f}）")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("时间 t (s)", fontweight="bold")
    ax.set_ylabel("|e(t)| (%, 对数轴)", fontweight="bold")
    ax.set_title("跟踪误差衰减（对数坐标，越低越快收敛）", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, cfg["simulation"]["t_end"])

    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "L1_level_response_main.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 L1 已保存: {path}")
    return path


def figure_L2_phase(data, cfg):
    """相平面轨迹对比（超调区阴影）"""
    keys = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, key in zip(axs.ravel(), keys):
        if key not in data:
            ax.set_visible(False)
            continue
        res = data[key]["res"]
        # 超调区（e < 0）阴影
        ax.axhspan(-12, 0, xmin=0.0, xmax=1.0, color="red", alpha=0.06, zorder=0,
                   label="超调区 (e < 0)")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.plot(res["e"], res["ec"], color=SPECS[key]["color"], lw=1.6,
                label=f"{data[key]['label']} 相轨迹")
        ax.scatter([res["e"][0]], [res["ec"][0]], color="darkblue", s=80, zorder=5,
                   label="起点 (e=40%, ec=0)")
        ax.scatter([0], [0], color="red", marker="*", s=170, zorder=5, label="稳态原点 (0,0)")
        m = data[key]["metrics"]
        ax.set_title(f"({key}) {data[key]['label']}：σ1={m['sigma1_pct']:.2f}%，"
                     f"反向最大偏差={res['e'].min():.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("误差 e = r - y (%)", fontweight="bold")
        ax.set_ylabel("误差变化率 ec (%/s)", fontweight="bold")
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_xlim(-15, 45)
        ax.set_ylim(-4, 3)
    fig.suptitle("图 L2  液位系统相平面轨迹对比（红色区域为超调区，轨迹进入表示液位过冲）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "L2_level_phase_plane.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 L2 已保存: {path}")
    return path


def figure_L3_gains(data, cfg):
    """模糊自适应增益整定轨迹"""
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    ph = cfg["phases"]
    names = ["比例增益 Kp(t)", "积分增益 Ki(t)", "微分增益 Kd(t)"]
    keys_map = [("kp", "kp0"), ("ki", "ki0"), ("kd", "kd0")]
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
    spans = [(ph["step1"], "#f4f6f9"), (ph["dist"], "#fff8f2"), (ph["step2"], "#f2f8fc")]
    for ax in axs:
        for (t0, t1), color in spans:
            ax.axvspan(t0, t1, color=color, zorder=0)
        ax.axvline(ph["dist"][0],  color="#c05020", ls="--", lw=1.1, alpha=0.8)
        ax.axvline(ph["step2"][0], color="darkblue", ls="--", lw=1.1, alpha=0.8)
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
    axs[0].set_title("图 L3  液位系统模糊自适应 PID 在线增益整定轨迹",
                     fontsize=13, fontweight="bold")
    axs[2].set_xlabel("时间 t (s)", fontweight="bold")
    axs[0].legend(loc="upper right", fontsize=8.5, ncol=2)
    for ax in axs:
        ax.set_xlim(0, cfg["simulation"]["t_end"])
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "L3_level_gain_adaptation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 L3 已保存: {path}")
    return path


def figure_L4_cascade(data, cfg):
    """串级特性验证：第一容腔（x1）vs 第二容腔（y）液位对比"""
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    r = data[DEFAULT_ORDER[0]]["res"]["r"]
    ph = cfg["phases"]
    fig, axs = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True)

    for ax in axs:
        ax.axvspan(*ph["step1"], color="#f4f6f9", zorder=0)
        ax.axvspan(*ph["dist"],  color="#fff8f2", zorder=0)
        ax.axvspan(*ph["step2"], color="#f2f8fc", zorder=0)
        ax.axvline(ph["dist"][0],  color="#c05020", ls="--", lw=1.2, alpha=0.85)
        ax.axvline(ph["step2"][0], color="darkblue", ls="--", lw=1.2, alpha=0.85)
        ax.grid(True, ls="--", alpha=0.45)

    ax = axs[0]
    ax.plot(t, r, "r--", lw=1.8, label="目标设定 r(t)")
    for key in ["pid_opt", "fuzzy"]:
        if key not in data:
            continue
        ax.plot(t, data[key]["res"]["x1"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=2.0,
                label=f"{data[key]['label']} — 第一容腔 x1(t)")
    ax.set_ylabel("第一容腔液位 x1 (%)", fontweight="bold")
    ax.set_title("图 L4  串级双容特性验证：上游槽（x1）与下游槽（y）液位动态对比",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(10, 80)
    ax.text(0.02, 0.95, "注：x1 先于 y 响应，体现二阶惯性的动态传播滞后", transform=ax.transAxes,
            fontsize=9, va="top", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.7))

    ax = axs[1]
    ax.plot(t, r, "r--", lw=1.8, label="目标设定 r(t)")
    for key in ["pid_opt", "fuzzy"]:
        if key not in data:
            continue
        ax.plot(t, data[key]["res"]["y"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=2.0,
                label=f"{data[key]['label']} — 输出 y(t)")
    ax.set_ylabel("输出液位 y (%)", fontweight="bold")
    ax.set_xlabel("时间 t (s)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(10, 80)
    ax.set_xlim(0, cfg["simulation"]["t_end"])

    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "L4_level_cascade_states.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 L4 已保存: {path}")
    return path


def figure_L5_radar(data, cfg):
    """多控制器综合性能雷达图"""
    keys = [k for k in DEFAULT_ORDER if k in data]
    labels_ctrl = [data[k]["label"] for k in keys]

    # 选取 6 个指标，值越小越好 → 归一化后翻转（取 1 - norm，使外圈代表"好"）
    metric_keys = ["sigma1_pct", "ts1_s", "dist_drop", "itae_abs", "tv", "sat_duty_pct"]
    metric_names = ["无超调\n(1-σ1)", "快调节\n(1-ts1)", "抗扰强\n(1-跌落)",
                    "低ITAE\n(1-ITAE)", "省执行\n(1-TV)", "少饱和\n(1-占空比)"]

    raw = np.array([[data[k]["metrics"][m] for m in metric_keys] for k in keys])
    col_max = raw.max(axis=0)
    col_max[col_max < 1e-9] = 1.0
    norm = 1.0 - raw / col_max    # 翻转：越好越接近 1

    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]            # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    colors = [SPECS[k]["color"] for k in keys]
    for i, (key, label_c, color) in enumerate(zip(keys, labels_ctrl, colors)):
        vals = norm[i].tolist() + norm[i][:1].tolist()
        ax.plot(angles, vals, color=color, lw=2.0,
                ls=SPECS[key]["linestyle"], label=label_c)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("图 L5  液位系统多控制器综合性能雷达图\n（外圈 = 越好；各维度已归一化翻转）",
                 fontsize=12, fontweight="bold", y=1.06)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "L5_level_radar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图 L5 已保存: {path}")
    return path


# ---------------------------------------------------------------------------
def main():
    cfg = scenario("level")

    print("=" * 118)
    print(" 场景三：双容串联液位过程（SOPDT: K=0.96, T1=30 s, T2=15 s, τ=3 s）模糊自适应 PID 对比实验")
    print(f" 工况: {cfg['simulation']['t_end']:.0f} s, dt={cfg['simulation']['dt']} s, "
          f"初始液位 {cfg['plant']['y0']}%, 目标 60%→40%, 进液冲击 +{cfg['disturbance']['amplitude']:.0f}% "
          f"@{cfg['disturbance']['t_on']:.0f} s")
    print("=" * 118)

    tuned = load_tuned()
    data = run_all_level(cfg, tuned)
    report_level(data, cfg)
    dump_level_timeseries(data, cfg)
    print(f"\n[OK] 指标 CSV: {os.path.join(RESULTS, 'level_metrics.csv')}")

    figure_L1_main(data, cfg)
    figure_L2_phase(data, cfg)
    figure_L3_gains(data, cfg)
    figure_L4_cascade(data, cfg)
    figure_L5_radar(data, cfg)

    print("\n[完成] 液位场景全部图表与数据已更新。\n")


if __name__ == "__main__":
    main()

