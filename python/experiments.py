"""
拓展实验套件 (Extended experiment suite)
======================================
主场景脚本只回答"在标称工况下谁更好"，这不足以支撑一个可靠的控制实验结论。
本套件用 5 组实验回答更有价值的问题：

E1 参数摄动蒙特卡洛鲁棒性  —— K、T、τ 同时 ±30% 随机摄动，配对比较 300 组工况
E2 控制器采样周期影响      —— 多速率零阶保持仿真，考察 PLC 扫描周期从 0.2 s 到 4 s
E3 量测噪声与微分滤波      —— 理想微分 vs 一阶滤波微分，量化执行机构抖振代价 TV(u)
E4 缩放因子敏感性          —— kup/kui/kud 单因素扫描 + (kup, kud) 二维代价面
E5 宽范围阶梯跟踪          —— 40→80→110→95→125 ℃ 多段变工况，检验大范围适应性

所有实验数据写入 ``results/``，图表写入 ``results/figures/``。
运行:  python python/experiments.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
ROOT = os.path.dirname(CURRENT_DIR)
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")

import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fuzzy_config import scenario
from simcore import run_thermal, evaluate_thermal
from metrics import (response_metrics, control_effort, write_csv, print_table,
                     relative_gain, _window)
from controllers import build, SPECS, DEFAULT_ORDER
from tuning import load_tuned

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"   # 保证对数轴指数中的负号有可用字形
plt.rcParams["figure.dpi"] = 300

KEYS = ["pid_aw", "pid_opt", "fuzzy", "fuzzy_opt"]
CMP_KEYS = ("pid_opt", "fuzzy_opt")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图已保存: {path}")
    return path


# ===========================================================================
# E1 参数摄动蒙特卡洛鲁棒性
# ===========================================================================
def exp_monte_carlo(cfg, tuned, n=300, seed=20240922, mismatch=0.30):
    rng = np.random.default_rng(seed)
    p = cfg["plant"]
    print("\n" + "=" * 100)
    print(f" E1 参数摄动蒙特卡洛鲁棒性实验：K、T、τ 各自独立均匀摄动 ±{mismatch*100:.0f}%，"
          f"共 {n} 组工况，控制器参数全程不重调")
    print("=" * 100)

    rows = []
    for i in range(n):
        f = 1.0 + rng.uniform(-mismatch, mismatch, size=3)
        K, T, tau = p["K"] * f[0], p["T"] * f[1], p["tau"] * f[2]
        for key in KEYS:
            ctrl = build("thermal", key, tuned=tuned)
            res = run_thermal(ctrl, cfg, K=K, T=T, tau=tau)
            m = evaluate_thermal(res, cfg, label=SPECS[key]["label"])
            rows.append(dict(run=i, K=K, T=T, tau=tau, key=key, **m))

    write_csv(os.path.join(RESULTS, "exp1_monte_carlo_raw.csv"), rows)

    # --- 汇总统计 ---
    stat_rows, table = [], []
    for key in KEYS:
        sub = [r for r in rows if r["key"] == key]
        itae = np.array([r["itae_abs"] for r in sub])
        sig = np.array([r["sigma1_pct"] for r in sub])
        drop = np.array([r["dist_drop"] for r in sub])
        ts = np.array([r["ts1_s"] for r in sub])
        ok = np.mean((sig < 10.0) & (np.abs([r["ess1"] for r in sub]) < 1.0)) * 100
        stat_rows.append(dict(key=key, label=SPECS[key]["label"],
                              itae_mean=itae.mean(), itae_std=itae.std(),
                              itae_p95=np.percentile(itae, 95), itae_max=itae.max(),
                              sigma_mean=sig.mean(), sigma_max=sig.max(),
                              drop_mean=drop.mean(), drop_max=drop.max(),
                              ts_mean=ts.mean(), pass_rate=ok))
        table.append([SPECS[key]["label"], f"{sig.mean():.2f}", f"{sig.max():.2f}",
                      f"{ts.mean():.1f}", f"{drop.mean():.2f}", f"{drop.max():.2f}",
                      f"{itae.mean():.0f}", f"{itae.std():.0f}",
                      f"{np.percentile(itae, 95):.0f}", f"{itae.max():.0f}", f"{ok:.1f}%"])
    print_table(" 表 E1-1  随机摄动下的统计性能（均值 / 最坏情况 / 95 分位）",
                ["控制器", "σ1均值/%", "σ1最坏/%", "ts1均值/s", "跌落均值/℃", "跌落最坏/℃",
                 "ITAE均值", "ITAE标准差", "ITAE 95%", "ITAE最坏", "合格率"],
                table)
    write_csv(os.path.join(RESULTS, "exp1_monte_carlo_summary.csv"), stat_rows)

    # --- 配对比较：同一工况下模糊 vs 最优固定 PID ---
    a = {r["run"]: r for r in rows if r["key"] == CMP_KEYS[0]}
    b = {r["run"]: r for r in rows if r["key"] == CMP_KEYS[1]}
    imp_itae = np.array([relative_gain(a[i]["itae_abs"], b[i]["itae_abs"]) for i in a])
    imp_sig = np.array([a[i]["sigma1_pct"] - b[i]["sigma1_pct"] for i in a])
    imp_drop = np.array([relative_gain(a[i]["dist_drop"], b[i]["dist_drop"]) for i in a])
    win = np.mean(imp_itae > 0) * 100
    print(f"\n 表 E1-2  配对比较（{SPECS[CMP_KEYS[1]]['label']} 相对 "
          f"{SPECS[CMP_KEYS[0]]['label']}，工况一一对应）")
    print_table("", ["指标", "改善均值", "改善中位数", "优于对照组工况占比"],
                [["全程 ITAE", f"{imp_itae.mean():+.2f}%", f"{np.median(imp_itae):+.2f}%", f"{win:.1f}%"],
                 ["抗扰跌落深度", f"{imp_drop.mean():+.2f}%", f"{np.median(imp_drop):+.2f}%",
                  f"{np.mean(imp_drop > 0)*100:.1f}%"],
                 ["超调量 σ1（对照组减实验组）", f"{imp_sig.mean():+.2f} 个百分点",
                  f"{np.median(imp_sig):+.2f} 个百分点", f"{np.mean(imp_sig > 0)*100:.1f}%"]])
    write_csv(os.path.join(RESULTS, "exp1_paired_comparison.csv"),
              [dict(run=i, itae_pid=a[i]["itae_abs"], itae_fuzzy=b[i]["itae_abs"],
                    improvement_pct=imp_itae[idx], sigma_pid=a[i]["sigma1_pct"],
                    sigma_fuzzy=b[i]["sigma1_pct"], K=a[i]["K"], T=a[i]["T"], tau=a[i]["tau"])
               for idx, i in enumerate(a)])

    # --- 绘图 ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    colors = [SPECS[k]["color"] for k in KEYS]
    names = [SPECS[k]["label"] for k in KEYS]

    ax = axs[0, 0]
    data = [[r["itae_abs"] for r in rows if r["key"] == k] for k in KEYS]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(KEYS) + 1))
    ax.set_xticklabels(names, fontsize=9, rotation=12)
    ax.set_yscale("log")
    ax.set_ylabel("全程 ITAE（对数轴）", fontweight="bold")
    ax.set_title(f"(a) {int(mismatch*100)}% 参数摄动下 ITAE 分布", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4, axis="y")

    ax = axs[0, 1]
    data = [[r["sigma1_pct"] for r in rows if r["key"] == k] for k in KEYS]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(KEYS) + 1))
    ax.set_xticklabels(names, fontsize=9, rotation=12)
    ax.set_ylabel("初始阶跃超调 σ1 (%)", fontweight="bold")
    ax.set_title("(b) 超调量分布", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4, axis="y")

    ax = axs[1, 0]
    x = np.array([a[i]["itae_abs"] for i in a])
    y = np.array([b[i]["itae_abs"] for i in a])
    lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
    ax.scatter(x, y, s=14, alpha=0.6, color="#2ca02c", edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1.2, label="性能相同线 y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{SPECS[CMP_KEYS[0]]['label']} ITAE", fontweight="bold")
    ax.set_ylabel(f"{SPECS[CMP_KEYS[1]]['label']} ITAE", fontweight="bold")
    ax.set_title(f"(c) 配对散点：{win:.1f}% 工况位于等值线下方（越低越好）", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

    ax = axs[1, 1]
    ax.hist(imp_itae, bins=30, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="crimson", ls="--", lw=1.6, label="无改善分界线")
    ax.axvline(imp_itae.mean(), color="darkgreen", lw=1.8,
               label=f"均值 {imp_itae.mean():.1f}%")
    ax.set_xlabel("ITAE 相对改善 (%)", fontweight="bold")
    ax.set_ylabel("工况数", fontweight="bold")
    ax.set_title("(d) 鲁棒性收益分布", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("图 E1  参数摄动蒙特卡洛鲁棒性实验（K、T、τ 独立 ±30% 随机摄动）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "E1_monte_carlo_robustness.png"), stat_rows


# ===========================================================================
# E2 控制器采样周期影响（多速率零阶保持）
# ===========================================================================
def exp_control_period(cfg, tuned, periods=(0.2, 0.5, 1.0, 2.0, 4.0)):
    print("\n" + "=" * 100)
    print(" E2 控制器采样周期实验：对象以 dt=0.2 s 积分，控制器按零阶保持每 Ts 秒动作一次")
    print("=" * 100)
    rows, traces = [], {}
    for key in CMP_KEYS:
        for ts in periods:
            ctrl = build("thermal", key, tuned=tuned)
            res = run_thermal(ctrl, cfg, ctrl_period=ts)
            m = evaluate_thermal(res, cfg, label=SPECS[key]["label"])
            rows.append(dict(key=key, label=SPECS[key]["label"], Ts=ts, **m))
            if ts in (0.2, 4.0):
                traces[(key, ts)] = res
    write_csv(os.path.join(RESULTS, "exp2_control_period.csv"), rows)

    table = []
    for ts in periods:
        r1 = next(r for r in rows if r["key"] == CMP_KEYS[0] and r["Ts"] == ts)
        r2 = next(r for r in rows if r["key"] == CMP_KEYS[1] and r["Ts"] == ts)
        table.append([f"{ts:g}", f"{r1['sigma1_pct']:.2f}", f"{r2['sigma1_pct']:.2f}",
                      f"{r1['ts1_s']:.0f}", f"{r2['ts1_s']:.0f}",
                      f"{r1['itae_abs']:.0f}", f"{r2['itae_abs']:.0f}",
                      f"{r2['tv']:.0f}"])
    print_table(" 表 E2 采样周期对控制品质的影响",
                ["Ts/s", "σ1(PID)/%", "σ1(模糊)/%", "ts1(PID)/s", "ts1(模糊)/s",
                 "ITAE(PID)", "ITAE(模糊)", "TV(模糊)"], table)

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for key in CMP_KEYS:
        sub = [r for r in rows if r["key"] == key]
        axs[0].semilogx([r["Ts"] for r in sub], [r["itae_abs"] for r in sub], "o-",
                        color=SPECS[key]["color"], lw=1.8, label=SPECS[key]["label"])
        axs[1].semilogx([r["Ts"] for r in sub], [r["sigma1_pct"] for r in sub], "s-",
                        color=SPECS[key]["color"], lw=1.8, label=SPECS[key]["label"])
        axs[2].semilogx([r["Ts"] for r in sub], [r["tv"] for r in sub], "^-",
                        color=SPECS[key]["color"], lw=1.8, label=SPECS[key]["label"])
    for ax, title, ylabel in zip(axs,
                                 ["(a) 跟踪精度随采样周期退化", "(b) 超调量随采样周期变化",
                                  "(c) 执行机构动作频度"],
                                 ["全程 ITAE", "σ1 (%)", "控制量总变差 TV(u)"]):
        ax.set_xlabel("控制器采样周期 Ts (s, 对数轴)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=9)
    fig.suptitle("图 E2  控制器采样周期（PLC 扫描周期）敏感性实验", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "E2_control_period.png")

    fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for (key, ts), res in traces.items():
        lab = f"{SPECS[key]['label']} (Ts={ts:g} s)"
        axs[0].plot(res["t"], res["y"], color=SPECS[key]["color"],
                    alpha=0.55 if ts > 1 else 1.0, lw=1.8, label=lab)
        axs[1].step(res["t"], res["u"], color=SPECS[key]["color"],
                    alpha=0.55 if ts > 1 else 1.0, lw=1.2, label=lab, where="post")
    axs[0].plot(traces[(CMP_KEYS[0], 0.2)]["t"], traces[(CMP_KEYS[0], 0.2)]["r"], "r--",
                lw=1.5, label="目标设定值")
    axs[0].set_ylabel("温度 (℃)", fontweight="bold")
    axs[0].set_title("Ts = 0.2 s 与 Ts = 4 s 的闭环响应对比（其余条件完全相同）", fontweight="bold")
    axs[1].set_ylabel("加热功率 (%)", fontweight="bold")
    axs[1].set_xlabel("时间 t (s)", fontweight="bold")
    axs[1].set_xlim(0, 130)
    for ax in axs:
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    _save(fig, "E2_timeseries_Ts.png")
    return rows


# ===========================================================================
# E3 量测噪声与微分滤波
# ===========================================================================
def exp_noise_filter(cfg, tuned, noises=(0.0, 0.25, 0.5, 1.0), tau_f=5.0, seed=7):
    print("\n" + "=" * 100)
    print(" E3 量测噪声与微分滤波实验：热电偶白噪声 σn，对比理想微分与一阶滤波微分 (Tf=%.1f s)"
          % tau_f)
    print("=" * 100)
    rows = []
    for sigma in noises:
        for key in CMP_KEYS:
            for tf in (0.0, tau_f):
                ctrl = build("thermal", key, tuned=tuned, kd_filter_tau=tf)
                res = run_thermal(ctrl, cfg, noise_std=sigma, seed=seed)
                m = evaluate_thermal(res, cfg, label=SPECS[key]["label"])
                m.update(noise_std=sigma, filter_tau=tf, key=key, label=SPECS[key]["label"])
                rows.append(m)
    write_csv(os.path.join(RESULTS, "exp3_noise_filter.csv"), rows)

    table = []
    for sigma in noises:
        for key in CMP_KEYS:
            for tf in (0.0, tau_f):
                r = next(x for x in rows if x["key"] == key and x["noise_std"] == sigma
                         and x["filter_tau"] == tf)
                table.append([SPECS[key]["label"], f"{sigma:g}",
                              "理想微分" if tf == 0 else f"滤波 Tf={tf:g}s",
                              f"{r['tv']:.0f}", f"{r['tail_std2']:.4f}",
                              f"{r['sigma1_pct']:.2f}", f"{r['itae_abs']:.0f}"])
    print_table(" 表 E3 噪声水平 × 微分实现方式 对控制品质与执行机构代价的影响",
                ["控制器", "噪声σn/℃", "微分实现", "TV(u)", "稳态波动/℃", "σ1/%", "ITAE"], table)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    res_pid = run_thermal(build("thermal", CMP_KEYS[0], tuned=tuned, kd_filter_tau=0.0),
                          cfg, noise_std=1.0, seed=seed)
    res_f0 = run_thermal(build("thermal", CMP_KEYS[1], tuned=tuned, kd_filter_tau=0.0),
                         cfg, noise_std=1.0, seed=seed)
    res_f5 = run_thermal(build("thermal", CMP_KEYS[1], tuned=tuned, kd_filter_tau=tau_f),
                         cfg, noise_std=1.0, seed=seed)

    ax = axs[0, 0]
    for res, lab, c in ((res_f0, "模糊(理想微分)", "#d62728"),
                        (res_f5, f"模糊(滤波微分 Tf={tau_f:g}s)", "#2ca02c"),
                        (res_pid, "最优固定PID(理想微分)", "#9467bd")):
        ax.plot(res["t"], res["y"], color=c, lw=1.3, label=lab)
    ax.plot(res_f0["t"], res_f0["r"], "r--", lw=1.4, label="设定值")
    ax.set_xlim(40, 120); ax.set_ylim(90, 106)
    ax.set_ylabel("温度 (℃)", fontweight="bold")
    ax.set_title("(a) σn=1.0 ℃ 时的温度响应（局部放大）", fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, ls="--", alpha=0.4)

    ax = axs[0, 1]
    for res, lab, c in ((res_f0, "模糊(理想微分)", "#d62728"),
                        (res_f5, f"模糊(滤波微分 Tf={tau_f:g}s)", "#2ca02c"),
                        (res_pid, "最优固定PID(理想微分)", "#9467bd")):
        ax.plot(res["t"], res["u"], color=c, lw=1.1, label=lab)
    ax.set_xlim(40, 120)
    ax.set_ylabel("加热功率 (%)", fontweight="bold")
    ax.set_title("(b) 噪声经微分环节放大后造成的执行机构抖振", fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, ls="--", alpha=0.4)

    ax = axs[1, 0]
    w = 0.35
    idx = np.arange(len(noises))
    for j, (key, tf) in enumerate([(CMP_KEYS[1], 0.0), (CMP_KEYS[1], tau_f)]):
        vals = [next(x for x in rows if x["key"] == key and x["noise_std"] == s
                     and x["filter_tau"] == tf)["tv"] for s in noises]
        ax.bar(idx + (j - 0.5) * w, vals, w * 0.95,
               color="#d62728" if tf == 0 else "#2ca02c",
               label="理想微分" if tf == 0 else f"滤波 Tf={tf:g}s")
    ax.set_xticks(idx); ax.set_xticklabels([f"{s:g}" for s in noises])
    ax.set_xlabel("量测噪声标准差 σn (℃)", fontweight="bold")
    ax.set_ylabel("控制量总变差 TV(u)", fontweight="bold")
    ax.set_title("(c) 执行机构动作频度随噪声的变化", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4, axis="y")

    ax = axs[1, 1]
    for key in CMP_KEYS:
        vals = [next(x for x in rows if x["key"] == key and x["noise_std"] == s
                     and x["filter_tau"] == 0.0)["itae_abs"] for s in noises]
        vals_f = [next(x for x in rows if x["key"] == key and x["noise_std"] == s
                       and x["filter_tau"] == tau_f)["itae_abs"] for s in noises]
        ax.plot(noises, vals, "o-", color=SPECS[key]["color"], lw=1.8,
                label=f"{SPECS[key]['label']} 理想微分")
        ax.plot(noises, vals_f, "s--", color=SPECS[key]["color"], lw=1.8, alpha=0.7,
                label=f"{SPECS[key]['label']} 滤波微分")
    ax.set_xlabel("量测噪声标准差 σn (℃)", fontweight="bold")
    ax.set_ylabel("全程 ITAE", fontweight="bold")
    ax.set_title("(d) 微分滤波对跟踪精度的影响", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("图 E3  量测噪声与微分滤波实验（工程实现细节对控制品质的影响）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "E3_noise_and_filter.png")
    return rows


# ===========================================================================
# E4 缩放因子敏感性
# ===========================================================================
def exp_scaling_sensitivity(cfg, tuned, n1d=7, n2d=11):
    print("\n" + "=" * 100)
    print(" E4 模糊控制器缩放因子敏感性实验：单因素扫描 + (kup, kud) 二维代价面")
    print("=" * 100)
    p = cfg["pid"]
    grid = {
        "kup": np.linspace(0.2, 4.0, n1d),
        "kui": np.linspace(0.005, 0.12, n1d),
        "kud": np.linspace(0.0, 8.0, n1d),
    }
    rows = []
    for name, values in grid.items():
        for v in values:
            kw = dict(kup=p["kup"], kui=p["kui"], kud=p["kud"])
            kw[name] = float(v)
            ctrl = build("thermal", "fuzzy", tuned=None, scaling=kw)
            res = run_thermal(ctrl, cfg)
            m = evaluate_thermal(res, cfg)
            rows.append(dict(factor=name, value=float(v), **{k: m[k] for k in
                        ("sigma1_pct", "ts1_s", "dist_drop", "itae_abs", "tv", "sat_duty_pct")}))
    write_csv(os.path.join(RESULTS, "exp4_scaling_sweep_1d.csv"), rows)

    kup_axis = np.linspace(0.2, 4.0, n2d)
    kud_axis = np.linspace(0.0, 8.0, n2d)
    cost = np.zeros((n2d, n2d))
    for i, kup in enumerate(kup_axis):
        for j, kud in enumerate(kud_axis):
            ctrl = build("thermal", "fuzzy", tuned=None,
                         scaling=dict(kup=float(kup), kui=p["kui"], kud=float(kud)))
            res = run_thermal(ctrl, cfg)
            cost[j, i] = evaluate_thermal(res, cfg)["itae_abs"]
    write_csv(os.path.join(RESULTS, "exp4_scaling_sweep_2d.csv"),
              [dict(kup=float(kup_axis[i]), kud=float(kud_axis[j]), itae=float(cost[j, i]))
               for j in range(n2d) for i in range(n2d)])

    table = []
    for name, values in grid.items():
        sub = [r for r in rows if r["factor"] == name]
        best = min(sub, key=lambda r: r["itae_abs"])
        worst = max(sub, key=lambda r: r["itae_abs"])
        table.append([name, f"{values[0]:g}~{values[-1]:g}", f"{best['value']:.3f}",
                      f"{best['itae_abs']:.0f}", f"{worst['itae_abs']:.0f}",
                      f"{(worst['itae_abs']-best['itae_abs'])/best['itae_abs']*100:.1f}%",
                      f"{best['sigma1_pct']:.2f}"])
    print_table(" 表 E4-1 单因素扫描结果（每个因子其余两个保持配置默认值）",
                ["因子", "扫描范围", "最优取值", "最优ITAE", "最差ITAE", "相对敏感度", "最优σ1/%"], table)
    print(f" 配置默认值: kup={p['kup']}, kui={p['kui']}, kud={p['kud']}")
    # cost 的行 = kud, 列 = kup（写入时 cost[j, i]）
    j_kud, i_kup = np.unravel_index(np.argmin(cost), cost.shape)
    print(f" (kup, kud) 二维最优: kup={kup_axis[i_kup]:.3f}, kud={kud_axis[j_kud]:.3f}, "
          f"ITAE={cost[j_kud, i_kup]:.0f}")

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, name in zip(axs[:3], grid):
        sub = [r for r in rows if r["factor"] == name]
        ax.plot([r["value"] for r in sub], [r["itae_abs"] for r in sub], "o-", color="#1f77b4")
        ax.axvline(p[name], color="crimson", ls="--", lw=1.4, label="配置默认值")
        ax2 = ax.twinx()
        ax2.plot([r["value"] for r in sub], [r["sigma1_pct"] for r in sub], "s--",
                 color="#ff7f0e", alpha=0.8)
        ax2.set_ylabel("σ1 (%)", color="#ff7f0e")
        ax.set_xlabel(name, fontweight="bold")
        ax.set_ylabel("全程 ITAE", color="#1f77b4", fontweight="bold")
        ax.set_title(f"({name}) 单因素敏感性", fontweight="bold")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8.5)
    im = axs[3].contourf(kup_axis, kud_axis, cost, levels=18, cmap="viridis")
    axs[3].plot(p["kup"], p["kud"], "r*", ms=16, label="配置默认值")
    axs[3].set_xlabel("kup", fontweight="bold")
    axs[3].set_ylabel("kud", fontweight="bold")
    axs[3].set_title("(kup × kud) ITAE 代价面", fontweight="bold")
    axs[3].legend(fontsize=9)
    fig.colorbar(im, ax=axs[3], shrink=0.8, label="ITAE")
    fig.suptitle("图 E4  模糊控制器缩放因子敏感性（参数不敏感区即工程可用的鲁棒整定区间）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "E4_scaling_sensitivity.png")
    return rows


# ===========================================================================
# E5 宽范围阶梯跟踪
# ===========================================================================
def exp_staircase_tracking(cfg, tuned,
                           segments=((0.0, 40.0), (60.0, 80.0), (120.0, 110.0),
                                     (180.0, 95.0), (240.0, 125.0))):
    print("\n" + "=" * 100)
    print(" E5 宽范围阶梯跟踪实验：40→80→110→95→125 ℃ 五段变工况（含一次降温反向阶跃）")
    print("=" * 100)
    cfg_s = copy.deepcopy(cfg)
    cfg_s["reference"]["segments"] = [list(s) for s in segments]
    cfg_s["disturbance"]["t_on"] = 1e9        # 本实验不含冷负荷扰动，专注跟踪品质
    cfg_s["phases"] = {f"seg{i}": [segments[i][0], segments[i + 1][0] if i + 1 < len(segments)
                                   else cfg["simulation"]["t_end"]]
                       for i in range(len(segments))}

    rows, results = [], {}
    for key in CMP_KEYS:
        ctrl = build("thermal", key, tuned=tuned)
        res = run_thermal(ctrl, cfg_s)
        results[key] = res
        for i, (t0, rv) in enumerate(segments):
            t1 = segments[i + 1][0] if i + 1 < len(segments) else cfg["simulation"]["t_end"]
            base = float(np.interp(t0 - 1e-6, res["t"], res["y"]))
            m = response_metrics(res["t"], res["y"], res["r"], t0, t1, y_base=base,
                                 scale=abs(rv - base), closed_end=(i == len(segments) - 1))
            rows.append(dict(key=key, label=SPECS[key]["label"], segment=i, t0=t0, t1=t1,
                             target=rv, from_y=base, tr_s=m["rise_time"],
                             sigma_pct=m["overshoot"], ts_s=m["settling_time"],
                             ess=m["ess"], iae=m["iae"], itae=m["itae"]))
    write_csv(os.path.join(RESULTS, "exp5_staircase.csv"), rows)

    table = []
    for i, (t0, rv) in enumerate(segments):
        line = [f"第{i+1}段", f"{rv:.0f}", f"{t0:.0f}"]
        for key in CMP_KEYS:
            r = next(x for x in rows if x["key"] == key and x["segment"] == i)
            line += [f"{r['tr_s']:.1f}", f"{r['sigma_pct']:.2f}", f"{r['ts_s']:.1f}",
                     f"{r['ess']:+.2f}"]
        table.append(line)
    print_table(" 表 E5 阶梯跟踪分段指标（PID 组 / 模糊组 并排）",
                ["段", "目标/℃", "起始/s", "tr(PID)", "σ(PID)", "ts(PID)", "ess(PID)",
                 "tr(模糊)", "σ(模糊)", "ts(模糊)", "ess(模糊)"], table)

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True)
    for key in CMP_KEYS:
        res = results[key]
        axs[0].plot(res["t"], res["y"], color=SPECS[key]["color"], lw=2.0,
                    label=f"{SPECS[key]['label']}")
        axs[1].plot(res["t"], res["u"], color=SPECS[key]["color"], lw=1.4,
                    label=f"{SPECS[key]['label']}")
    axs[0].plot(results[CMP_KEYS[0]]["t"], results[CMP_KEYS[0]]["r"], "r--", lw=1.6,
                label="设定值（阶梯）")
    for t0, rv in segments[1:]:
        for ax in axs:
            ax.axvline(t0, color="gray", ls=":", lw=1.0)
        axs[0].annotate(f"{rv:.0f}℃", (t0, rv), textcoords="offset points",
                        xytext=(4, 6), fontsize=9, color="darkred")
    axs[0].set_ylabel("温度 (℃)", fontweight="bold")
    axs[0].set_title("五段宽范围变工况跟踪（含 110→95 ℃ 反向降温）", fontweight="bold")
    axs[1].set_ylabel("加热功率 (%)", fontweight="bold")
    axs[1].set_xlabel("时间 t (s)", fontweight="bold")
    axs[1].set_xlim(0, cfg["simulation"]["t_end"])
    for ax in axs:
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "E5_staircase_tracking.png")
    return rows


# ===========================================================================
# E6 摄动幅度 / 摄动方向 与自适应收益的关系（回答"什么时候值得用自适应"）
# ===========================================================================
def exp_mismatch_crossover(cfg, tuned, levels=(0.0, 0.15, 0.30, 0.45, 0.60),
                           n_each=40, seed=777):
    print("\n" + "=" * 100)
    print(" E6 摄动幅度/方向 与自适应收益：考察模糊自适应相对'ITAE 最优固定 PID'的收益")
    print("    何时显现（各向同性随机摄动 + 单向结构化工况 两种视角）")
    print("=" * 100)
    P = cfg["plant"]

    # --- 6.1 各向同性随机摄动幅度扫描 ---
    rows, table = [], []
    for m in levels:
        rng = np.random.default_rng(seed + int(m * 1000))
        a_itae, b_itae, a_sig, b_sig = [], [], [], []
        for _ in range(n_each):
            f = 1.0 + rng.uniform(-m, m, 3)
            kw = dict(K=P["K"] * f[0], T=P["T"] * f[1], tau=P["tau"] * f[2])
            for key, acc_i, acc_s in ((CMP_KEYS[0], a_itae, a_sig), (CMP_KEYS[1], b_itae, b_sig)):
                ctrl = build("thermal", key, tuned=tuned)
                met = evaluate_thermal(run_thermal(ctrl, cfg, **kw), cfg)
                acc_i.append(met["itae_abs"])
                acc_s.append(met["sigma1_pct"])
        a_itae, b_itae = np.array(a_itae), np.array(b_itae)
        imp = (a_itae - b_itae) / a_itae * 100
        rows.append(dict(mode="isotropic", level=m, itae_pid=a_itae.mean(),
                         itae_fuzzy=b_itae.mean(), improvement_mean=imp.mean(),
                         improvement_median=float(np.median(imp)),
                         win_rate=float((b_itae < a_itae).mean() * 100),
                         sigma_pid=float(np.mean(a_sig)), sigma_fuzzy=float(np.mean(b_sig))))
        table.append([f"±{m*100:.0f}%", f"{a_itae.mean():.0f}", f"{b_itae.mean():.0f}",
                      f"{imp.mean():+.2f}%", f"{np.median(imp):+.2f}%",
                      f"{(b_itae < a_itae).mean()*100:.0f}%",
                      f"{np.mean(a_sig):.2f}", f"{np.mean(b_sig):.2f}"])
    print_table(" 表 E6-1 各向同性随机摄动（每档 %d 组配对工况）" % n_each,
                ["摄动幅度", "ITAE(PID)", "ITAE(模糊)", "改善均值", "改善中位数",
                 "模糊胜率", "σ1(PID)/%", "σ1(模糊)/%"], table)

    # --- 6.2 单向结构化工况（揭示"收益是有方向性的"） ---
    rows2, table2 = [], []
    for name, key_name in (("K（静态增益）", "K"), ("T（时间常数）", "T"), ("τ（纯滞后）", "tau")):
        for s in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
            kw = {key_name: P[key_name] * s}
            met = {}
            for k in CMP_KEYS:
                ctrl = build("thermal", k, tuned=tuned)
                met[k] = evaluate_thermal(run_thermal(ctrl, cfg, **kw), cfg)
            imp = relative_gain(met[CMP_KEYS[0]]["itae_abs"], met[CMP_KEYS[1]]["itae_abs"])
            rows2.append(dict(mode="structured", param=key_name, scale=s,
                              itae_pid=met[CMP_KEYS[0]]["itae_abs"],
                              itae_fuzzy=met[CMP_KEYS[1]]["itae_abs"],
                              improvement_pct=imp,
                              sigma_pid=met[CMP_KEYS[0]]["sigma1_pct"],
                              sigma_fuzzy=met[CMP_KEYS[1]]["sigma1_pct"]))
            table2.append([name, f"×{s:g}", f"{met[CMP_KEYS[0]]['itae_abs']:.0f}",
                           f"{met[CMP_KEYS[1]]['itae_abs']:.0f}", f"{imp:+.1f}%",
                           f"{met[CMP_KEYS[0]]['sigma1_pct']:.2f}",
                           f"{met[CMP_KEYS[1]]['sigma1_pct']:.2f}"])
    print_table(" 表 E6-2 单向结构化工况（单参数偏离标称，其余保持标称）",
                ["被摄动参数", "倍率", "ITAE(PID)", "ITAE(模糊)", "改善",
                 "σ1(PID)/%", "σ1(模糊)/%"], table2)

    write_csv(os.path.join(RESULTS, "exp6_mismatch_crossover.csv"), rows + rows2)

    # --- 绘图 ---
    iso = [r for r in rows if r["mode"] == "isotropic"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    xs = [r["level"] * 100 for r in iso]
    ys = [r["improvement_mean"] for r in iso]
    axs[0].plot(xs, ys, "o-", color="#2ca02c", lw=2.0, label="改善均值")
    axs[0].fill_between(xs, [0] * len(iso), ys, color="#2ca02c", alpha=0.15)
    axs[0].axhline(0, color="k", lw=1.0, ls="--")
    axs[0].set_xlabel("参数摄动幅度 ±Δ (%)", fontweight="bold")
    axs[0].set_ylabel("ITAE 相对改善 (%)", fontweight="bold")
    axs[0].set_title("(a) 随机摄动：改善随不确定性增大而转正", fontweight="bold")
    axs[0].grid(True, ls="--", alpha=0.4)
    axs[0].legend(fontsize=9)

    for name, key_name, c in (("K", "K", "#1f77b4"), ("T", "T", "#ff7f0e"),
                              ("τ(纯滞后)", "tau", "#d62728")):
        sub = [r for r in rows2 if r["param"] == key_name]
        axs[1].plot([r["scale"] for r in sub], [r["improvement_pct"] for r in sub],
                    "s-", color=c, lw=1.8, label=name)
        axs[2].plot([r["scale"] for r in sub], [r["sigma_pid"] for r in sub], "o--",
                    color=c, lw=1.4, alpha=0.65, label=f"{name} · PID")
        axs[2].plot([r["scale"] for r in sub], [r["sigma_fuzzy"] for r in sub], "o-",
                    color=c, lw=1.8, label=f"{name} · 模糊")
    axs[1].axhline(0, color="k", lw=1.0, ls="--")
    axs[1].set_xlabel("参数倍率（标称=1.0）", fontweight="bold")
    axs[1].set_ylabel("ITAE 相对改善 (%)", fontweight="bold")
    axs[1].set_title("(b) 单向摄动：收益具有明显方向性", fontweight="bold")
    axs[1].grid(True, ls="--", alpha=0.4)
    axs[1].legend(fontsize=8.5)
    axs[2].set_xlabel("参数倍率（标称=1.0）", fontweight="bold")
    axs[2].set_ylabel("初始超调 σ1 (%)", fontweight="bold")
    axs[2].set_title("(c) 超调抑制：模糊控制在全工况带更小", fontweight="bold")
    axs[2].grid(True, ls="--", alpha=0.4)
    axs[2].legend(fontsize=7.5, ncol=2)
    fig.suptitle("图 E6  自适应机制的收益边界（何时值得用模糊自适应替代精心整定的固定 PID）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "E6_mismatch_crossover.png")
    return rows


# ===========================================================================
# E7 参数在线跳变鲁棒性（最能体现自适应机制价值的实验）
# ===========================================================================
def exp_param_jump(cfg, tuned):
    """
    E7 参数在线跳变实验：在稳态运行时对象参数突变，不重调控制器。

    工况设计（3 段）：
      0 ~ 80 s   : 标称参数（K=1.5, T=25 s）→ 100 ℃，等待稳态
      80~160 s   : K 突变为 3.0（增益加倍，对象变"重"）
      160~280 s  : K 恢复回 1.5（等待再次稳态），观察恢复过程

    结论检验：当对象参数在运行中突然改变时，模糊自适应 PID 能否比固定参数 PID
    更快地消除新稳态误差、抑制超调？
    """
    from simcore import piecewise_reference, step_signal, time_vector
    from plants import thermal_plant_from_config
    import copy

    print("\n" + "=" * 100)
    print(" E7 参数在线跳变实验：K 从 1.5 突变为 3.0（t=80 s），再恢复（t=160 s）")
    print("=" * 100)

    p = cfg["plant"]
    t_jump1 = 80.0    # K 突变为 2×
    t_jump2 = 160.0   # K 恢复
    t_end   = 280.0
    K_base  = p["K"]
    K_high  = p["K"] * 2.0

    dt = float(cfg["simulation"]["dt"])
    n_total = int(round(t_end / dt)) + 1
    t = np.arange(n_total) * dt
    r = np.full(n_total, 100.0)   # 全程目标 100 ℃

    # 构造时变 K 序列（在仿真循环中分段切换）
    keys_e7 = ["pid_aw", "pid_opt", "fuzzy", "fuzzy_opt"]
    rows = []
    result_store = {}

    for key in keys_e7:
        ctrl = build("thermal", key, tuned=tuned)
        plant = thermal_plant_from_config(cfg)
        ctrl.reset()

        y_arr  = np.zeros(n_total)
        u_arr  = np.zeros(n_total)
        e_arr  = np.zeros(n_total)
        ec_arr = np.zeros(n_total)
        kp_arr = np.zeros(n_total)
        ki_arr = np.zeros(n_total)
        kd_arr = np.zeros(n_total)
        prev_e = 0.0

        for k in range(n_total):
            # 在线切换 K
            if t[k] >= t_jump2 - 1e-9:
                plant.K = K_base
            elif t[k] >= t_jump1 - 1e-9:
                plant.K = K_high
            else:
                plant.K = K_base

            ym = plant.measure()
            u  = ctrl.compute(r[k], ym)
            plant.step(u, 0.0)
            e = r[k] - plant.y
            y_arr[k]  = plant.y
            u_arr[k]  = u
            e_arr[k]  = e
            ec_arr[k] = (e - prev_e) / dt if k else 0.0
            prev_e = e
            kp_arr[k] = ctrl.kp
            ki_arr[k] = ctrl.ki
            kd_arr[k] = ctrl.kd

        # 指标：只评估跳变窗口 [80,160] 的恢复质量
        from metrics import response_metrics, whole_run_metrics
        m_jump = response_metrics(t, y_arr, r, t_jump1, t_jump2,
                                  y_base=None, scale=100.0 - float(cfg["plant"]["y0"]))
        m_all  = whole_run_metrics(t, y_arr, r, u_arr, p["u_min"], p["u_max"])
        rows.append(dict(
            controller=SPECS[key]["label"], key=key,
            jump_overshoot=m_jump["overshoot"],
            jump_max_dev=m_jump["max_dev"],
            jump_settling=m_jump["settling_time"],
            itae_abs=m_all["itae_abs"],
            tv=m_all["tv"],
        ))
        result_store[key] = dict(t=t, y=y_arr, u=u_arr, e=e_arr, ec=ec_arr,
                                 kp=kp_arr, ki=ki_arr, kd=kd_arr, r=r)
        print(f"  [{key}] 跳变超调={m_jump['overshoot']:.2f}%  最大偏差={m_jump['max_dev']:.2f}℃  "
              f"调节时间={m_jump['settling_time']:.1f}s  ITAE={m_all['itae_abs']:.0f}")

    # 写 CSV
    from metrics import write_csv
    write_csv(os.path.join(RESULTS, "exp7_param_jump.csv"), rows)

    # 画图
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 11), sharex=True)
    colors_e7 = {k: SPECS[k]["color"] for k in keys_e7}
    ls_e7     = {k: SPECS[k]["linestyle"] for k in keys_e7}

    # 子图1：温度响应
    ax = axs[0]
    ax.axvspan(0,        t_jump1, color="#f4f6f9", zorder=0)
    ax.axvspan(t_jump1,  t_jump2, color="#fff2ec", zorder=0)
    ax.axvspan(t_jump2,  t_end,   color="#f2f8fc", zorder=0)
    ax.axvline(t_jump1, color="crimson",  ls="--", lw=1.5, label=f"t={t_jump1:.0f}s K突变2倍")
    ax.axvline(t_jump2, color="darkblue", ls="--", lw=1.5, label=f"t={t_jump2:.0f}s K恢复")
    ax.plot(t, r, "r--", lw=1.8, label="目标 100℃")
    for key in keys_e7:
        m = rows[[r["key"] for r in rows].index(key)]
        ax.plot(result_store[key]["t"], result_store[key]["y"],
                color=colors_e7[key], ls=ls_e7[key], lw=2.0,
                label=f"{SPECS[key]['label']}（跳变超调={m['jump_overshoot']:.2f}%）")
    ax.set_ylabel("炉膛温度 (℃)", fontweight="bold")
    ax.set_title("图 E7  参数在线跳变实验：K 突然加倍后，模糊自适应能否更快恢复？",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax.set_ylim(15, 145)
    ax.grid(True, ls="--", alpha=0.45)
    ax.text(40,   138, "标称工况\n(K=1.5)", ha="center", fontsize=9, fontweight="bold")
    ax.text(120,  138, "增益加倍\n(K=3.0)", ha="center", fontsize=9, color="crimson", fontweight="bold")
    ax.text(220,  138, "恢复标称\n(K=1.5)", ha="center", fontsize=9, color="darkblue", fontweight="bold")

    # 子图2：跟踪误差
    ax = axs[1]
    ax.axvspan(t_jump1, t_jump2, color="#fff2ec", zorder=0)
    ax.axvline(t_jump1, color="crimson",  ls="--", lw=1.5)
    ax.axvline(t_jump2, color="darkblue", ls="--", lw=1.5)
    for key in keys_e7:
        ax.plot(result_store[key]["t"],
                np.abs(result_store[key]["e"]),
                color=colors_e7[key], ls=ls_e7[key], lw=1.8, label=SPECS[key]["label"])
    ax.set_ylabel("|e(t)| (℃)", fontweight="bold")
    ax.set_title("绝对跟踪误差（跳变后误差峰值越小、恢复越快 = 控制器越健壮）", fontsize=11)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, ls="--", alpha=0.45)

    # 子图3：模糊控制器增益轨迹（Kp 为代表）
    ax = axs[2]
    ax.axvspan(t_jump1, t_jump2, color="#fff2ec", zorder=0)
    ax.axvline(t_jump1, color="crimson",  ls="--", lw=1.5, label=f"K 跳变")
    ax.axvline(t_jump2, color="darkblue", ls="--", lw=1.5, label=f"K 恢复")
    for key in ["fuzzy", "fuzzy_opt"]:
        if key not in result_store:
            continue
        ax.plot(result_store[key]["t"], result_store[key]["kp"],
                color=colors_e7[key], ls=ls_e7[key], lw=2.0,
                label=f"{SPECS[key]['label']} Kp(t)")
    ax.set_xlabel("时间 t (s)", fontweight="bold")
    ax.set_ylabel("比例增益 Kp(t)", fontweight="bold")
    ax.set_title("模糊自适应 PID 的在线增益响应（Kp 在跳变后自动增大以补偿新对象增益）", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(True, ls="--", alpha=0.45)
    ax.set_xlim(0, t_end)

    fig.tight_layout()
    _save(fig, "E7_param_jump.png")
    print(f"[OK] E7 实验完成，结果已写入 results/exp7_param_jump.csv")
    return rows


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="用小样本快速跑通（约 1 分钟）")
    ap.add_argument("--skip-mc", action="store_true", help="跳过耗时的蒙特卡洛实验")
    args = ap.parse_args()

    cfg = scenario("thermal")
    tuned = load_tuned()
    if not tuned:
        print("[警告] 未找到 results/tuned_params.json，pid_opt/fuzzy_opt 将退化为配置默认参数。")
        print("       建议先运行: python python/tuning.py")
    os.makedirs(RESULTS, exist_ok=True)

    n = 40 if args.quick else 300
    if not args.skip_mc:
        exp_monte_carlo(cfg, tuned, n=n)
    else:
        print("[跳过] E1 蒙特卡洛（--skip-mc 已激活）")
    exp_control_period(cfg, tuned)
    exp_noise_filter(cfg, tuned)
    exp_scaling_sensitivity(cfg, tuned, n1d=5 if args.quick else 7, n2d=7 if args.quick else 11)
    exp_staircase_tracking(cfg, tuned)
    exp_mismatch_crossover(cfg, tuned, n_each=10 if args.quick else 40)
    exp_param_jump(cfg, tuned)
    print("\n" + "=" * 100)
    print(f" 全部拓展实验完成，数据 CSV 位于: {RESULTS}")
    print(f" 实验图表位于: {FIGDIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
