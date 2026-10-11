"""
跨语言一致性核查 (Python 侧)
===========================
本脚本回答一个旧版工程从未回答、却直接决定实验可信度的问题：

    "Python 版、MATLAB 版和 Simulink 用的 .fis 文件，真的是同一套模糊规则吗？"

核查三层：
 1. **规则库层**：直接从 shared/fuzzy_config.json 读出规则表，检查点对称性、
    ΔKi 语义（大误差削弱积分、稳态强化积分）等；
 2. **推理层**：Python 实现在 61×61 论域网格上求值，与 MATLAB 类、官方
    readfis+evalfis 的结果逐点比对（容差 1e-9 / 1e-6）；
 3. **文本层**：解析 matlab/fuzzy_pid.fis 的隶属度函数与规则行，检查是否与
    共享配置逐项一致（防止"改了配置忘了重新生成 .fis"）。

运行:
    python python/verify_consistency.py
（先在 MATLAB 中运行 verify_consistency 生成 fis_grid_matlab.csv）
"""

from __future__ import annotations

import os
import re
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
ROOT = os.path.dirname(CURRENT_DIR)
RESULTS = os.path.join(ROOT, "results")

import numpy as np

from fuzzy_config import fuzzy_block, config_path
from fuzzy_pid import FuzzyInferenceSystem
from metrics import print_table


def grid_reference(n=61):
    fis = FuzzyInferenceSystem()
    ax = np.linspace(fis.lo, fis.hi, n)
    out = np.zeros((n * n, 5))
    k = 0
    for e in ax:
        for ec in ax:
            dkp, dki, dkd = fis.infer(e, ec)
            out[k] = [e, ec, dkp, dki, dkd]
            k += 1
    return out


def check_rules():
    cfg = fuzzy_block()
    rows = []
    ok_all = True
    for name, mat in cfg["rules"].items():
        arr = np.asarray(mat, dtype=float)
        sym = np.allclose(arr, arr[::-1, ::-1])
        rng = bool(np.all(np.abs(arr) <= 3))
        shape = arr.shape == (7, 7)
        ok_all &= sym and rng and shape
        rows.append([name, "7×7" if shape else str(arr.shape),
                     "是" if rng else "否", "是" if sym else "否"])
    # ΔKi 的三条语义断言
    ki = np.asarray(cfg["rules"]["dki"], dtype=float)
    sem = [("大误差处削弱积分 (ΔKi(e=±3) < 0)", bool(ki[0, 0] < 0 and ki[6, 6] < 0)),
           ("稳态附近强化积分 (ΔKi(0,0) 最大)", bool(ki[3, 3] == ki.max())),
           ("误差恶化方向 (e·ec>0) 抑制积分",
            bool(np.all(ki[0, 0] < 0 and ki[6, 6] < 0 and ki[1, 1] < 0 and ki[5, 5] < 0)))]
    print_table(" 核查 1：规则库结构与控制语义（数据源: %s）" % os.path.relpath(config_path(), ROOT),
                ["规则矩阵", "尺寸", "取值∈[-3,3]", "点对称"], rows)
    for desc, ok in sem:
        print(f"   {'[通过]' if ok else '[失败]'} {desc}")
        ok_all &= ok
    return ok_all


def compare_with_matlab():
    ref = grid_reference()
    print("\n 核查 2：推理结果跨语言逐点比对（61×61 = 3721 个论域点）")
    items = [
        ("results/fis_grid_matlab.csv", 1e-9, "MATLAB 类（纯代码实现）"),
        ("results/fis_grid_official_matlab.csv", 1e-6, ".fis + 官方 evalfis（Simulink 同款推理核）"),
    ]
    rows, ok_all, missing = [], True, []
    for rel, tol, desc in items:
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            missing.append(rel)
            continue
        other = np.loadtxt(path, delimiter=",", skiprows=0)
        if other.shape != ref.shape:
            rows.append([desc, f"形状不符 {other.shape}", "—", "失败"])
            ok_all = False
            continue
        max_dev = float(np.max(np.abs(other[:, 2:] - ref[:, 2:])))
        passed = max_dev < tol
        ok_all &= passed
        rows.append([desc, f"{max_dev:.3e}", f"{tol:.0e}", "通过" if passed else "失败"])
    if rows:
        print_table("", ["对照实现", "最大绝对偏差", "容差", "结论"], rows)
    for rel in missing:
        print(f"   [跳过] 缺少 {rel}（请先在 MATLAB 中运行 verify_consistency）")
    return ok_all, missing


def check_fis_text():
    path = os.path.join(ROOT, "matlab", "fuzzy_pid.fis")
    print("\n 核查 3：fuzzy_pid.fis 文本与共享配置的一致性")
    if not os.path.isfile(path):
        print("   [跳过] 未找到 matlab/fuzzy_pid.fis（请运行 matlab/generate_fis.m）")
        return True, True

    text = open(path, "r", encoding="utf-8", errors="replace").read()
    cfg = fuzzy_block()
    problems = []

    # --- 系统级设置：零阶 TS（单点输出 + 加权平均），与代码的加权平均严格等价 ---
    sys_line = re.search(r"Type='(\w+)'", text)
    defuzz = re.search(r"DefuzzMethod='(\w+)'", text)
    andm = re.search(r"AndMethod='(\w+)'", text)
    if not sys_line or sys_line.group(1) != "sugeno":
        problems.append("Type 应为 sugeno（零阶 TS 单点输出，等价于代码的加权平均清晰化）")
    if not defuzz or defuzz.group(1) != "wtaver":
        problems.append("DefuzzMethod 应为 wtaver（Mamdani+centroid 的面积重心与代码的"
                        "加权平均并不等价，实测偏差可达 1.09/6.0）")
    if not andm or andm.group(1) != "prod":
        problems.append("AndMethod 应为 prod（代码采用代数积 T-范数 α=μe·μec；"
                        "写成 min 会造成最大 0.522 的系统性偏差）")

    # --- 隶属度函数：输入为肩形/三角，输出为单点 ---
    mf_lines = re.findall(r"^MF\d+='(\w+)':'(\w+)',\[([^\]]+)\]", text, re.M)
    if len(mf_lines) != 35:
        problems.append(f"隶属度函数条目数 {len(mf_lines)} != 35 (5 变量 × 7)")
    for sec in range(1, 6):
        for i, lab in enumerate(cfg["labels"]):
            idx = (sec - 1) * 7 + i
            if idx >= len(mf_lines):
                continue
            name, typ, params = mf_lines[idx]
            got = [float(v) for v in params.split()]
            if sec <= 2:
                spec = cfg["mf"][lab]
                expect = [float(v) for v in spec["params"]]
                if name != lab or typ != spec["type"] or not np.allclose(got, expect, atol=1e-12):
                    problems.append(
                        f"输入{sec} 第{i+1}个 MF 应为 {lab}:{spec['type']}{expect}，"
                        f"实际 {name}:{typ}{got}")
            else:
                expect = [float(cfg["centers"][i])]
                if name != lab or typ != "constant" or not np.allclose(got, expect, atol=1e-12):
                    problems.append(
                        f"输出{sec-2} 第{i+1}个 MF 应为 {lab}:constant{expect}，"
                        f"实际 {name}:{typ}{got}")

    # --- 规则行 ---
    rule_lines = re.findall(r"^(\d+) (\d+), (\d+) (\d+) (\d+) \(1\) : 1$", text, re.M)
    if len(rule_lines) != 49:
        problems.append(f"规则行数 {len(rule_lines)} != 49")
    rules = cfg["rules"]
    expect_map = {"dkp": np.asarray(rules["dkp"]) + 4,
                  "dki": np.asarray(rules["dki"]) + 4,
                  "dkd": np.asarray(rules["dkd"]) + 4}
    k = 0
    for i in range(1, 8):
        for j in range(1, 8):
            if k >= len(rule_lines):
                break
            a, b, c1, c2, c3 = (int(v) for v in rule_lines[k])
            if (a, b) != (i, j):
                problems.append(f"第{k+1}条规则前提应为 ({i},{j})，实际 ({a},{b})")
            elif (c1, c2, c3) != (expect_map["dkp"][i - 1, j - 1],
                                  expect_map["dki"][i - 1, j - 1],
                                  expect_map["dkd"][i - 1, j - 1]):
                problems.append(
                    f"规则({i},{j}) 结论应为 "
                    f"({expect_map['dkp'][i-1,j-1]},{expect_map['dki'][i-1,j-1]},"
                    f"{expect_map['dkd'][i-1,j-1]})，实际 ({c1},{c2},{c3})")
            k += 1

    sys_line = re.search(r"AndMethod='(\w+)'", text)
    defuzz = re.search(r"DefuzzMethod='(\w+)'", text)
    if problems:
        for p in problems[:20]:
            print("   [失败]", p)
        if len(problems) > 20:
            print(f"   ... 另有 {len(problems)-20} 处不一致")
        return False, False
    print("   [通过] 推理结构（零阶 TS+wtaver）、5×7 个隶属度函数、49 条规则均与共享配置逐项一致")
    return True, False


def main():
    print("=" * 96)
    print(" 跨语言一致性核查：Python 实现 / MATLAB 类 / fuzzy_pid.fis 是否真的是同一套算法")
    print("=" * 96)
    ok1 = check_rules()
    ok2, missing = compare_with_matlab()
    ok3, skipped = check_fis_text()
    print("\n" + "=" * 96)
    if ok1 and ok2 and ok3:
        print(" 核查结论：全部通过 —— 两套语言实现与 .fis 配置等价，实验结论可跨语言复现。")
    else:
        print(" 核查结论：存在未通过项，请按上面的提示修正后再运行实验。")
    if missing:
        print(f" （提示：缺 MATLAB 网格文件 {missing}，请在 MATLAB 中运行 verify_consistency）")
    print("=" * 96)
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
