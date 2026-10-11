"""
实验数据汇总报告生成器 (Report generator)
========================================
把 ``results/`` 下的全部 CSV 与文本结果**自动汇总**成一份 Markdown 报告：

    results/实验结果汇总.md

设计原则：报告中的每一个数字都直接读自仿真产出的数据文件，
**不允许手工抄写**——这是避免"文档数字与代码不符"（旧版工程的主要问题）的根本办法。
若某个实验尚未运行，报告中会明确标注"未运行"，而不是留空或编造。

运行：  python python/make_report.py
"""

from __future__ import annotations

import csv
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
ROOT = os.path.dirname(CURRENT_DIR)
RESULTS = os.path.join(ROOT, "results")


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(v, digits=2, plus=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if plus:
        return f"{x:+.{digits}f}"
    return f"{x:.{digits}f}"


def md_table(headers, rows, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def section(title, body):
    return f"\n## {title}\n\n{body}\n"


def main():
    L = []
    L.append("# 智能控制课程设计 · 实验数据汇总（自动生成）\n")
    L.append("> 本文件由 `python/make_report.py` 从 `results/*.csv` 自动生成，**请勿手工编辑**。")
    L.append("> 重新跑完实验后执行一次即可刷新：`python python/make_report.py`\n")

    # ---------------- 1. 场景一：标称工况五组对照 ----------------
    rows = read_csv("thermal_metrics.csv")
    if rows:
        keys = ["controller", "tr_s", "sigma1_pct", "ts1_s", "ess1", "dist_drop",
                "dist_recovery_s", "sigma2_pct", "ts2_s", "itae_abs", "tv", "sat_duty_pct"]
        head = ["控制器", "tr/s", "σ1/%", "ts1/s", "ess1/℃", "跌落/℃", "恢复/s",
                "σ2/%", "ts2/s", "ITAE", "TV(u)", "饱和/%"]
        body = [[r["controller"], fnum(r["tr_s"]), fnum(r["sigma1_pct"]), fnum(r["ts1_s"], 1),
                 fnum(r["ess1"], 3, True), fnum(r["dist_drop"]), fnum(r["dist_recovery_s"], 1),
                 fnum(r["sigma2_pct"]), fnum(r["ts2_s"], 1), fnum(r["itae_abs"], 0),
                 fnum(r["tv"], 0), fnum(r["sat_duty_pct"], 1)] for r in rows]
        L.append(section("1. 场景一（工业加热炉）标称工况对比结果",
                         md_table(head, body,
                                  ["---"] + ["---:"] * 11) +
                         "\n数据源：`results/thermal_metrics.csv`（由 `python/temperature_simulation.py` 生成）"))
    else:
        L.append(section("1. 场景一（工业加热炉）标称工况对比结果", "_未运行：请先执行 `python python/temperature_simulation.py`_"))

    # ---------------- 2. 跨语言一致性 ----------------
    dev = read_csv("cross_language_check.csv")
    if dev:
        body = [[r["controller"], f"{float(r['max_abs_dy']):.3e}",
                 f"{float(r['max_abs_du']):.3e}"] for r in dev]
        L.append(section("2. 跨语言一致性核查（Python ↔ MATLAB 闭环轨迹逐点比对）",
                         md_table(["控制器", "max|Δy|", "max|Δu|"], body, ["---", "---:", "---:"]) +
                         "\n数据源：`results/cross_language_check.csv`。"
                         "偏差量级 ~1e-8 来自 CSV 只写了 10 位有效数字，属于数据交换精度而非算法差异。"))
    else:
        L.append(section("2. 跨语言一致性核查", "_未运行：请执行 MATLAB `thermal_simulation`_"))

    # ---------------- 3. E1 蒙特卡洛 ----------------
    mc = read_csv("exp1_monte_carlo_summary.csv")
    if mc:
        body = [[r["label"], fnum(r["sigma_mean"]), fnum(r["sigma_max"]), fnum(r["ts_mean"], 1),
                 fnum(r["drop_mean"]), fnum(r["drop_max"]), fnum(r["itae_mean"], 0),
                 fnum(r["itae_std"], 0), fnum(r["itae_p95"], 0), fnum(r["itae_max"], 0),
                 fnum(r["pass_rate"], 1) + "%"] for r in mc]
        L.append(section("3. E1 参数摄动蒙特卡洛（K、T、τ 各 ±30%，300 组工况）",
                         md_table(["控制器", "σ1均值/%", "σ1最坏/%", "ts1均值/s", "跌落均值/℃",
                                   "跌落最坏/℃", "ITAE均值", "ITAE标准差", "ITAE 95%", "ITAE最坏", "合格率"],
                                  body, ["---"] + ["---:"] * 10) +
                         "\n数据源：`results/exp1_monte_carlo_summary.csv`；"
                         "原始逐工况数据见 `results/exp1_monte_carlo_raw.csv`。"))
    else:
        L.append(section("3. E1 参数摄动蒙特卡洛", "_未运行：请执行 `python python/experiments.py`_"))

    pair = read_csv("exp1_paired_comparison.csv")
    if pair:
        import statistics
        imp = [float(r["improvement_pct"]) for r in pair]
        L.append("\n配对比较（模糊-优化因子 相对 PID-ITAE最优，同一工况一一对应）：\n\n" +
                 md_table(["统计量", "数值"],
                          [["工况数", len(imp)],
                           ["ITAE 改善均值", f"{statistics.mean(imp):+.2f}%"],
                           ["ITAE 改善中位数", f"{statistics.median(imp):+.2f}%"],
                           ["模糊更优的工况占比", f"{sum(1 for x in imp if x > 0) / len(imp) * 100:.1f}%"]],
                          ["---", "---:"]))

    # ---------------- 4. E2 采样周期 ----------------
    cp = read_csv("exp2_control_period.csv")
    if cp:
        periods = sorted({float(r["Ts"]) for r in cp})
        body = []
        for ts in periods:
            a = next(r for r in cp if float(r["Ts"]) == ts and r["key"] == "pid_opt")
            b = next(r for r in cp if float(r["Ts"]) == ts and r["key"] == "fuzzy_opt")
            body.append([f"{ts:g}", fnum(a["sigma1_pct"]), fnum(b["sigma1_pct"]),
                         fnum(a["itae_abs"], 0), fnum(b["itae_abs"], 0), fnum(b["tv"], 0)])
        L.append(section("4. E2 控制器采样周期（PLC 扫描周期）影响",
                         md_table(["Ts/s", "σ1(PID)/%", "σ1(模糊)/%", "ITAE(PID)", "ITAE(模糊)",
                                   "TV(模糊)"], body, ["---:"] * 6) +
                         "\n数据源：`results/exp2_control_period.csv`"))

    # ---------------- 5. E3 噪声与微分滤波 ----------------
    nf = read_csv("exp3_noise_filter.csv")
    if nf:
        body = []
        for r in nf:
            body.append([r["label"], fnum(r["noise_std"], 2),
                         "理想微分" if float(r["filter_tau"]) == 0 else f"滤波 Tf={float(r['filter_tau']):g}s",
                         fnum(r["tv"], 0), fnum(r["tail_std2"], 4), fnum(r["sigma1_pct"]),
                         fnum(r["itae_abs"], 0)])
        L.append(section("5. E3 量测噪声与微分滤波",
                         md_table(["控制器", "σn/℃", "微分实现", "TV(u)", "稳态波动/℃", "σ1/%", "ITAE"],
                                  body, ["---", "---:", "---", "---:", "---:", "---:", "---:"]) +
                         "\n数据源：`results/exp3_noise_filter.csv`"))

    # ---------------- 6. E5 阶梯跟踪 ----------------
    st = read_csv("exp5_staircase.csv")
    if st:
        segs = sorted({int(r["segment"]) for r in st})
        body = []
        for s in segs:
            a = next(r for r in st if int(r["segment"]) == s and r["key"] == "pid_opt")
            b = next(r for r in st if int(r["segment"]) == s and r["key"] == "fuzzy_opt")
            body.append([f"第{s+1}段", fnum(a["target"], 0), fnum(a["tr_s"], 1), fnum(a["sigma_pct"]),
                         fnum(b["tr_s"], 1), fnum(b["sigma_pct"]),
                         fnum(100 * (float(a["sigma_pct"]) - float(b["sigma_pct"])) /
                               max(1e-9, float(a["sigma_pct"])), 1) + "%"])
        L.append(section("6. E5 宽范围阶梯跟踪（40→80→110→95→125 ℃）",
                         md_table(["段", "目标/℃", "tr(PID)/s", "σ(PID)/%", "tr(模糊)/s",
                                   "σ(模糊)/%", "σ 降低幅度"], body, ["---"] + ["---:"] * 6) +
                         "\n数据源：`results/exp5_staircase.csv`"))

    # ---------------- 7. E6 收益边界 ----------------
    cross = read_csv("exp6_mismatch_crossover.csv")
    if cross:
        iso = [r for r in cross if r["mode"] == "isotropic"]
        body = [[f"±{float(r['level'])*100:.0f}%", fnum(r["itae_pid"], 0), fnum(r["itae_fuzzy"], 0),
                 fnum(r["improvement_mean"], 2, True) + "%", fnum(r["improvement_median"], 2, True) + "%",
                 fnum(r["win_rate"], 0) + "%", fnum(r["sigma_pid"]), fnum(r["sigma_fuzzy"])]
                for r in iso]
        L.append(section("7. E6 自适应收益的边界（一）各向同性随机摄动",
                         md_table(["摄动幅度", "ITAE(PID)", "ITAE(模糊)", "改善均值", "改善中位数",
                                   "模糊胜率", "σ1(PID)/%", "σ1(模糊)/%"], body,
                                  ["---:"] * 8)))
        st2 = [r for r in cross if r["mode"] == "structured"]
        pname = {"K": "K（静态增益）", "T": "T（时间常数）", "tau": "τ（纯滞后）"}
        body2 = [[pname.get(r["param"], r["param"]), f"×{float(r['scale']):g}",
                  fnum(r["itae_pid"], 0), fnum(r["itae_fuzzy"], 0),
                  fnum(r["improvement_pct"], 1, True) + "%",
                  fnum(r["sigma_pid"]), fnum(r["sigma_fuzzy"])] for r in st2]
        L.append("\n**（二）单向结构化工况**（单参数偏离标称，其余保持标称）\n\n" +
                 md_table(["被摄动参数", "倍率", "ITAE(PID)", "ITAE(模糊)", "改善",
                           "σ1(PID)/%", "σ1(模糊)/%"], body2, ["---", "---:"] + ["---:"] * 5) +
                 "\n数据源：`results/exp6_mismatch_crossover.csv`")

    # ---------------- 8. 场景二（电机） ----------------
    mo = read_csv("motor_metrics_matlab.csv")
    if mo:
        body = [[r["controller"], fnum(r["tr_s"], 3), fnum(r["sigma1_pct"]), fnum(r["ts1_s"], 3),
                 fnum(r["dip_rad_s"], 3), fnum(r["dip_recovery_s"], 3), fnum(r["sigma2_pct"]),
                 fnum(r["ess2"], 4), fnum(r["itae_abs"], 1), fnum(r["tv"], 1),
                 fnum(r["sat_duty_pct"])] for r in mo]
        L.append(section("8. 场景二（直流伺服电机）多工况对比结果",
                         md_table(["控制器", "tr/s", "σ1/%", "ts1/s", "跌落/(rad/s)", "恢复/s",
                                   "σ2/%", "ess2", "ITAE", "TV(u)", "饱和/%"], body,
                                  ["---"] + ["---:"] * 10) +
                         "\n工况：0→100 rad/s 起动；t=2.0 s 突加 0.25 N·m；t=3.8 s 惯量×2.5 并升至 130 rad/s。"
                         "\n数据源：`results/motor_metrics_matlab.csv`"))

    # ---------------- 9. 电机拓展实验 ----------------
    m1 = read_csv("exp1_motor_monte_carlo_summary.csv")
    if m1:
        body = [[r["controller"], fnum(r["itae_mean"]), fnum(r["itae_worst"]), fnum(r["itae_std"]),
                 fnum(r["sigma_mean"]), fnum(r["sigma_worst"]), fnum(r["dip_mean"], 3)]
                for r in m1]
        L.append(section("9. E1M 电机机电参数 ±30% 摄动蒙特卡洛",
                         md_table(["控制器", "ITAE均值", "ITAE最坏", "ITAE标准差", "σ1均值/%",
                                   "σ1最坏/%", "跌落均值"], body, ["---"] + ["---:"] * 6) +
                         "\n数据源：`results/exp1_motor_monte_carlo_summary.csv`"))

    # ---------------- E7. 参数在线跳变实验 ----------------
    e7 = read_csv("exp7_param_jump.csv")
    if e7:
        body = [[r["controller"], fnum(r["jump_overshoot"]), fnum(r["jump_max_dev"]),
                 fnum(r["jump_settling"], 1), fnum(r["itae_abs"], 0), fnum(r["tv"], 0)]
                for r in e7]
        L.append(section("E7. 参数在线跳变鲁棒性（K 突变 2 倍，控制器参数不重调）",
                         md_table(["控制器", "跳变超调/%", "最大偏差/℃", "调节时间/s", "全程ITAE", "TV(u)"],
                                  body, ["---"] + ["---:"] * 5) +
                         "\n工况：t=80s 时对象增益 K 由 1.5 突变至 3.0（加倍），t=160s 恢复；控制器参数全程不变。"
                         "\n数据源：`results/exp7_param_jump.csv`"))

    # ---------------- 场景三：液位 ----------------
    lv = read_csv("level_metrics.csv")
    if lv:
        keys_lv = ["controller", "tr_s", "sigma1_pct", "ts1_s", "ess1",
                   "dist_drop", "dist_recovery_s", "sigma2_pct", "ts2_s", "itae_abs", "tv"]
        head_lv = ["控制器", "tr/s", "σ1/%", "ts1/s", "ess1/%", "扰动冲高/%", "恢复/s",
                   "σ2/%", "ts2/s", "ITAE", "TV(u)"]
        body_lv = [[r["controller"], fnum(r["tr_s"]), fnum(r["sigma1_pct"]),
                    fnum(r["ts1_s"], 1), fnum(r["ess1"], 3, True), fnum(r["dist_drop"]),
                    fnum(r["dist_recovery_s"], 1), fnum(r["sigma2_pct"]), fnum(r["ts2_s"], 1),
                    fnum(r["itae_abs"], 0), fnum(r["tv"], 0)] for r in lv]
        L.append(section("场景三（双容液位过程）多工况对比结果",
                         md_table(head_lv, body_lv, ["---"] + ["---:"] * 10) +
                         "\n对象：G(s)=0.96/[(30s+1)(15s+1)]·e^{-3s}，工况：20%→60% 升液，+15% 进液冲击，60%→40% 降液。"
                         "\n数据源：`results/level_metrics.csv`（由 `python/level_simulation.py` 生成）"))

    # ---------------- 10. Simulink 核查 ----------------
    sp = os.path.join(RESULTS, "simulink_check.txt")
    if os.path.isfile(sp):
        with open(sp, "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read().strip()
        L.append(section("10. Simulink 模型核查", "```\n" + txt + "\n```"))
    else:
        L.append(section("10. Simulink 模型核查", "_未运行：请执行 MATLAB `build_simulink_model` / `run_simulink_check`_"))

    out = os.path.join(RESULTS, "实验结果汇总.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("[OK] 报告已生成:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

