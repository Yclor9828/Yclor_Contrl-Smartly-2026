"""
场景一：工业电加热炉温度控制系统 —— 多维度仿真与实验分析
========================================================
对象：一阶惯性 + 纯滞后 (FOPDT)  ``G(s) = 1.5/(25s+1)·e^{-5s}``
工况：20 ℃ 起步 → 100 ℃ 阶跃 → t=130 s 突加 -18 ℃ 冷负荷 → t=200 s 二次升阶 125 ℃

本脚本相对旧版本的改进（全部可复现、可核查）：
1. 未再手写任何指标，全部由 ``metrics.py`` 从仿真数据计算；
2. 对照组从"1 个未加抗饱和的固定 PID"扩展为 5 个：无抗饱和 / 抗饱和 /
   ITAE 最优固定参数 / 模糊默认因子 / 模糊优化因子；
3. 增加了控制量总变差 TV(u) 与执行器饱和占空比两项执行机构代价指标；
4. 结果同时导出 CSV（``results/``），供 MATLAB 侧做跨语言一致性核查。

运行：  python python/temperature_simulation.py
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

from fuzzy_config import scenario, fuzzy_block
from fuzzy_pid import FuzzyInferenceSystem
from simcore import run_thermal, evaluate_thermal, time_vector
from metrics import print_table, write_csv, relative_gain
from controllers import build_set, SPECS, DEFAULT_ORDER, tuned_missing
from tuning import load_tuned

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"   # 保证对数轴指数中的负号有可用字形
plt.rcParams["figure.dpi"] = 300

FIG_MAIN = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy"]
FIG_FUZZY = ["fuzzy", "fuzzy_opt"]


# ---------------------------------------------------------------------------
def run_all(cfg, tuned, keys=DEFAULT_ORDER):
    data = {}
    for key, label, ctrl in build_set("thermal", keys, tuned=tuned):
        res = run_thermal(ctrl, cfg)
        data[key] = dict(res=res, metrics=evaluate_thermal(res, cfg, label=label),
                         label=label, ctrl=ctrl)
    return data


def report(data, cfg, tuned):
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
    print(" 表 1  工业加热炉多工况闭环性能指标总表（全部数值由仿真数据直接计算，无任何手工填写）")
    print("=" * 118)
    print_table("", ["控制器", "tr/s", "σ1/%", "ts1/s", "ess1/℃", "跌落/℃",
                     "恢复/s", "σ2/%", "ts2/s", "ITAE", "TV(u)", "饱和/%"], table)

    ref = data.get("pid_opt") or data.get("pid_aw") or data[DEFAULT_ORDER[0]]
    fz = data.get("fuzzy_opt") or data["fuzzy"]
    a, b = ref["metrics"], fz["metrics"]

    def fmt_rel(x, y):
        v = relative_gain(x, y)
        return "—（基准为 0）" if not np.isfinite(v) else f"{v:+.1f}%"

    print("\n 表 2  综合自适应收益（对照组 = %s，实验组 = %s）" % (ref["label"], fz["label"]))
    print_table("", ["指标", "对照组", "实验组", "改善"],
                [["初始超调 σ1 (%)", f"{a['sigma1_pct']:.2f}", f"{b['sigma1_pct']:.2f}",
                  fmt_rel(a['sigma1_pct'], b['sigma1_pct'])],
                 ["初始调节时间 ts1 (s)", f"{a['ts1_s']:.1f}", f"{b['ts1_s']:.1f}",
                  fmt_rel(a['ts1_s'], b['ts1_s'])],
                 ["冷负荷最大跌落 (℃)", f"{a['dist_drop']:.2f}", f"{b['dist_drop']:.2f}",
                  fmt_rel(a['dist_drop'], b['dist_drop'])],
                 ["二次升阶超调 σ2 (%)", f"{a['sigma2_pct']:.2f}", f"{b['sigma2_pct']:.2f}",
                  fmt_rel(a['sigma2_pct'], b['sigma2_pct'])],
                 ["全程 ITAE", f"{a['itae_abs']:.0f}", f"{b['itae_abs']:.0f}",
                  fmt_rel(a['itae_abs'], b['itae_abs'])],
                 ["控制量总变差 TV(u)", f"{a['tv']:.0f}", f"{b['tv']:.0f}",
                  fmt_rel(a['tv'], b['tv'])]])

    used = {}
    for key in DEFAULT_ORDER:
        if key in data:
            c = data[key]["ctrl"]
            used[key] = {"controller": data[key]["label"]}
            if hasattr(c, "kp0"):
                used[key].update(kp0=c.kp0, ki0=c.ki0, kd0=c.kd0,
                                 kup=c.kup, kui=c.kui, kud=c.kud)
            else:
                used[key].update(kp=c.kp_const, ki=c.ki_const, kd=c.kd_const,
                                 anti_windup=c.anti_windup)
    print("\n 表 3  本表所用控制器参数（对照组与实验组参数完全公开，便于复现）")
    print_table("", ["控制器", "Kp/Ki/Kd 或 基准+缩放"],
                [[used[k]["controller"],
                  (f"Kp={used[k]['kp']:.3f}, Ki={used[k]['ki']:.4f}, Kd={used[k]['kd']:.4f}"
                   if "kp" in used[k] else
                   f"Kp0={used[k]['kp0']}(+{used[k]['kup']}), Ki0={used[k]['ki0']}(+{used[k]['kui']}), "
                   f"Kd0={used[k]['kd0']}(+{used[k]['kud']})")] for k in used])

    write_csv(os.path.join(RESULTS, "thermal_metrics.csv"), rows)
    return rows

def dump_timeseries(data, cfg):
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    cols = {"t": t,
            "r": data[DEFAULT_ORDER[0]]["res"]["r"],
            "d": data[DEFAULT_ORDER[0]]["res"]["d"]}
    for key in DEFAULT_ORDER:
        if key in data:
            cols[f"y_{key}"] = data[key]["res"]["y"]
            cols[f"u_{key}"] = data[key]["res"]["u"]
    path = os.path.join(RESULTS, "thermal_timeseries_python.csv")
    header = ",".join(cols)
    arr = np.column_stack([cols[c] for c in cols])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.10g")
    return path


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def figure_main(data, cfg):
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    r = data[DEFAULT_ORDER[0]]["res"]["r"]
    d = data[DEFAULT_ORDER[0]]["res"]["d"]
    ph = cfg["phases"]
    fig, axs = plt.subplots(4, 1, figsize=(11.5, 12.5), sharex=True)

    for ax in axs:
        ax.axvspan(*ph["step1"], color="#f4f6f9", zorder=0)
        ax.axvspan(*ph["dist"], color="#fff2f2", zorder=0)
        ax.axvspan(*ph["step2"], color="#f2f8fc", zorder=0)
        ax.axvline(ph["dist"][0], color="crimson", ls="--", lw=1.2, alpha=0.85)
        ax.axvline(ph["step2"][0], color="darkblue", ls="--", lw=1.2, alpha=0.85)
        ax.grid(True, ls="--", alpha=0.45)

    ax = axs[0]
    ax.plot(t, r, "r--", lw=1.8, label="目标设定值 r(t)")
    ax.fill_between(t, r - 2.0, r + 2.0, color="gray", alpha=0.18, label="±2 ℃ 误差带")
    for key in FIG_MAIN:
        if key not in data:
            continue
        m = data[key]["metrics"]
        ax.plot(t, data[key]["res"]["y"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=2.0,
                label=f"{data[key]['label']}（σ1={m['sigma1_pct']:.2f}%, ts1={m['ts1_s']:.1f}s）")
    ax.set_ylabel("炉膛温度 (℃)", fontweight="bold")
    ax.set_title("图 1  工业加热炉多工况温度响应对比（含 5 s 纯滞后与突发冷负荷）",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(15, 145)
    ax.text(65, 137, "阶段1：20→100 ℃ 阶跃升温", ha="center", fontsize=9, fontweight="bold")
    ax.text(165, 137, "阶段2：冷负荷 -18 ℃", ha="center", fontsize=9, color="crimson", fontweight="bold")
    ax.text(240, 137, "阶段3：二次升温 125 ℃", ha="center", fontsize=9, color="darkblue", fontweight="bold")

    ax = axs[1]
    ax.plot(t, d, color="crimson", lw=2.2, label="等效冷负荷扰动 d(t)")
    ax.fill_between(t, d, 0, where=(d < 0), color="crimson", alpha=0.25)
    ax.set_ylabel("负荷扰动 (℃)", fontweight="bold")
    ax.set_ylim(-24, 4)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"外部扰动时序：t={cfg['disturbance']['t_on']:.0f} s 起叠加 "
                 f"{cfg['disturbance']['amplitude']:.0f} ℃ 冷负荷，至 t={cfg['disturbance']['t_off']:.0f} s 结束",
                 fontsize=11)

    ax = axs[2]
    for key in FIG_MAIN:
        if key not in data:
            continue
        ax.plot(t, data[key]["res"]["u"], color=SPECS[key]["color"],
                ls=SPECS[key]["linestyle"], lw=1.8,
                label=f"{data[key]['label']}（u 总变差 {data[key]['metrics']['tv']:.0f}）")
    ax.axhline(cfg["plant"]["u_max"], color="k", lw=1.0, ls=":")
    ax.set_ylabel("加热功率 (%)", fontweight="bold")
    ax.set_ylim(-5, 108)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("执行机构控制量 u(t)（0~100% 加热功率，含限幅）", fontsize=11)

    ax = axs[3]
    for key in FIG_MAIN:
        if key not in data:
            continue
        ax.semilogy(t, np.maximum(np.abs(data[key]["res"]["e"]), 1e-3),
                    color=SPECS[key]["color"], ls=SPECS[key]["linestyle"], lw=1.8,
                    label=f"{data[key]['label']}（ITAE={data[key]['metrics']['itae_abs']:.0f}）")
    # 误差幅值恒为正，用十进制刻度标签（避免对数轴指数里的 Unicode 负号缺字形问题）
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="lower"))
    ax.set_xlabel("时间 t (s)", fontweight="bold")
    ax.set_ylabel("|e(t)| (℃, 对数轴)", fontweight="bold")
    ax.set_title("跟踪误差衰减过程（对数坐标，可清晰比较收敛速率与是否出现残余振荡）", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, cfg["simulation"]["t_end"])

    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "1_temperature_power_disturbance_response.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 1 已保存: {path}")
    return path


def figure_phase(data, cfg):
    keys = ["pid_nowindup", "pid_aw", "pid_opt", "fuzzy"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, key in zip(axs.ravel(), keys):
        if key not in data:
            continue
        res = data[key]["res"]
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.plot(res["e"], res["ec"], color=SPECS[key]["color"], lw=1.6,
                label=f"{data[key]['label']} 相轨迹")
        ax.scatter([res["e"][0]], [res["ec"][0]], color="darkblue", s=80, zorder=5,
                   label="起点 (e=80 ℃, ec=0)")
        ax.scatter([0], [0], color="red", marker="*", s=170, zorder=5, label="稳态原点 (0,0)")
        m = data[key]["metrics"]
        ax.set_title(f"({key}) {data[key]['label']}：σ1={m['sigma1_pct']:.2f}%，"
                     f"反向最大偏差={res['e'].min():.2f} ℃", fontsize=11, fontweight="bold")
        ax.set_xlabel("误差 e = r - y (℃)", fontweight="bold")
        ax.set_ylabel("误差变化率 ec (℃/s)", fontweight="bold")
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_xlim(-24, 85)
        ax.set_ylim(-6, 4)
    fig.suptitle("图 2  相平面轨迹对比（e–ec 平面：轨迹是否越过 e<0 区域直接决定超调）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "2_control_phase_plane_trajectory.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 2 已保存: {path}")
    return path


def figure_gains(data, cfg):
    t = data[DEFAULT_ORDER[0]]["res"]["t"]
    ph = cfg["phases"]
    names = ["比例增益 Kp(t)", "积分增益 Ki(t)", "微分增益 Kd(t)"]
    keys_map = [("kp", "kp0"), ("ki", "ki0"), ("kd", "kd0")]
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
    spans = [(ph["step1"], "#f4f6f9"), (ph["dist"], "#fff2f2"), (ph["step2"], "#f2f8fc")]
    for ax in axs:
        for (t0, t1), color in spans:
            ax.axvspan(t0, t1, color=color, zorder=0)
        ax.axvline(ph["dist"][0], color="crimson", ls="--", lw=1.1, alpha=0.8)
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
                           label=f"其基准值 {base_key}={getattr(c, base_key):g}")
        axs[i].set_ylabel(names[i], fontweight="bold")
    axs[0].set_title("图 3  模糊自适应 PID 在线增益整定轨迹（真实仿真轨迹，非示意曲线）",
                     fontsize=13, fontweight="bold")
    axs[2].set_xlabel("时间 t (s)", fontweight="bold")
    axs[0].legend(loc="upper right", fontsize=8.5, ncol=2)
    for ax in axs:
        ax.set_xlim(0, cfg["simulation"]["t_end"])
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "3_fuzzy_parameter_adaptation_trajectories.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 3 已保存: {path}")
    return path


def figure_surfaces(cfg):
    fis = FuzzyInferenceSystem()
    e, ec, out = fis.surface(61)
    E, EC = np.meshgrid(e, ec)
    titles = ["(a) ΔKp* 决策曲面", "(b) ΔKi* 决策曲面", "(c) ΔKd* 决策曲面"]
    cmaps = ["viridis", "plasma", "coolwarm"]
    fig = plt.figure(figsize=(15, 4.8))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        surf = ax.plot_surface(E, EC, out[i], cmap=cmaps[i], edgecolor="none", alpha=0.94)
        ax.set_title(titles[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("误差论域 e*", fontsize=9)
        ax.set_ylabel("变化率论域 ec*", fontsize=9)
        ax.set_zlabel(["ΔKp*", "ΔKi*", "ΔKd*"][i], fontsize=9)
        ax.view_init(elev=28, azim=-125)
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12)
    fig.suptitle("图 4  Mamdani 49 规则三维非线性决策曲面（由 shared/fuzzy_config.json 唯一数据源生成）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "4_fuzzy_decision_surfaces_3d.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 4 已保存: {path}")
    return path


def figure_bars(data, cfg):
    keys = [k for k in DEFAULT_ORDER if k in data]
    labels = [data[k]["label"] for k in keys]
    items = [("sigma1_pct", "初始超调 σ1 (%)"), ("ts1_s", "初始调节时间 ts1 (s)"),
             ("dist_drop", "冷负荷跌落 (℃)"), ("itae_abs", "全程 ITAE"),
             ("tv", "控制量总变差 TV(u)"), ("sat_duty_pct", "饱和占空比 (%)")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title) in zip(axs.ravel(), items):
        vals = [data[k]["metrics"][key] for k in keys]
        bars = ax.bar(range(len(keys)), vals,
                      color=[SPECS[k]["color"] for k in keys], alpha=0.85)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([l.replace("模糊自适应PID", "模糊").replace("PID-", "")
                            for l in labels], fontsize=8.5, rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if v < 1000 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("图 5  多控制器综合性能对比（越小越好；TV(u) 与饱和占空比反映执行机构代价）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CURRENT_DIR, "5_multicontroller_metric_bars.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] 图 5 已保存: {path}")
    return path


# ---------------------------------------------------------------------------
def main():
    cfg = scenario("thermal")
    tuned = load_tuned()
    miss = tuned_missing(tuned, "thermal", DEFAULT_ORDER)
    if miss:
        print(f"[警告] 未找到寻优参数 {miss}，相关对照组将退化为配置默认参数。")
        print("       请先运行:  python python/tuning.py")

    print("=" * 118)
    print(" 场景一：工业电加热炉（FOPDT: K=1.5 ℃/%, T=25 s, τ=5 s）模糊自适应 PID 对比实验")
    print(f" 工况: {cfg['simulation']['t_end']:.0f} s, dt={cfg['simulation']['dt']} s, "
          f"初温 {cfg['plant']['y0']} ℃, 目标 100→125 ℃, 冷负荷 -18 ℃ @130 s")
    print("=" * 118)

    data = run_all(cfg, tuned)
    report(data, cfg, tuned)
    print("\n[OK] 指标 CSV:", os.path.join(RESULTS, "thermal_metrics.csv"))
    print("[OK] 时序 CSV:", dump_timeseries(data, cfg))

    figure_main(data, cfg)
    figure_phase(data, cfg)
    figure_gains(data, cfg)
    figure_surfaces(cfg)
    figure_bars(data, cfg)

    import shutil
    src = os.path.join(CURRENT_DIR, "1_temperature_power_disturbance_response.png")
    shutil.copyfile(src, os.path.join(CURRENT_DIR, "temperature_simulation_result.png"))
    print("\n[完成] 全部图表与数据已更新。\n")


if __name__ == "__main__":
    main()
