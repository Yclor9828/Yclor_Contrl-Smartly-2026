"""
一键运行脚本 —— 模糊自适应 PID 控制课程设计全流程
=====================================================
按顺序执行所有仿真、实验、可视化与报告生成步骤。
支持 --quick 模式（跳过耗时的蒙特卡洛，约 2 分钟内完成）。

使用方法：
    python python/run_all.py           # 完整模式（约 8~12 分钟）
    python python/run_all.py --quick   # 快速模式（约 2~3 分钟）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)


def run_step(name: str, script: str, extra_args: list[str] | None = None):
    """运行一个子步骤，打印耗时，失败时报错但继续执行。"""
    args = [sys.executable, os.path.join(CURRENT_DIR, script)] + (extra_args or [])
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  [步骤] {name}")
    print(f"  命令: {' '.join(args)}")
    print(f"{'='*80}")
    ret = subprocess.run(args, cwd=ROOT)
    elapsed = time.time() - t0
    status = "✓ 完成" if ret.returncode == 0 else "✗ 失败（继续下一步）"
    print(f"  {status}  耗时 {elapsed:.1f} s")
    return ret.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="模糊自适应PID课程设计一键运行脚本")
    ap.add_argument("--quick", action="store_true",
                    help="快速模式：跳过蒙特卡洛（300组→40组），约 2 分钟")
    ap.add_argument("--skip-matlab", action="store_true",
                    help="跳过 MATLAB 侧实验（如 MATLAB 不可用）")
    args = ap.parse_args()

    total_start = time.time()
    results = {}

    print("=" * 80)
    print("  模糊自适应 PID 控制课程设计 —— 全流程一键运行")
    print(f"  模式: {'快速（--quick）' if args.quick else '完整'}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 步骤1: 算法自检（单元测试 + 跨语言一致性）
    results["单元测试"] = run_step(
        "算法自检（21项单元测试）",
        "test_fuzzy_core.py"
    )

    # 步骤2: 参数整定
    results["参数整定"] = run_step(
        "参数整定（ITAE最优固定PID + 模糊缩放因子）",
        "tuning.py"
    )

    # 步骤3: 主场景 — 加热炉
    results["加热炉主场景"] = run_step(
        "场景一：工业电加热炉温度控制（5组对比 + 5张图）",
        "temperature_simulation.py"
    )

    # 步骤4: 拓展实验（加热炉）
    extra_exp = ["--skip-mc"] if args.quick else []
    results["拓展实验(加热炉)"] = run_step(
        "拓展实验 E1~E7（鲁棒性MC / 采样周期 / 噪声 / 参数跳变等）",
        "experiments.py",
        (["--quick"] if args.quick else []) + extra_exp
    )

    # 步骤5: 场景二 — 电机（Python侧）
    results["电机场景(Python)"] = run_step(
        "场景二：直流伺服电机转速控制（Python侧 + 4张图）",
        "motor_simulation.py"
    )

    # 步骤6: 场景三 — 液位
    results["液位场景"] = run_step(
        "场景三：双容串联液位过程（SOPDT + 5张图）",
        "level_simulation.py"
    )

    # 步骤7: 模糊推理机解剖图
    results["模糊解剖图"] = run_step(
        "模糊推理机解剖图（隶属度函数 + 推理过程热力图）",
        "figure_fuzzy_anatomy.py"
    )

    # 步骤8: 自动汇总报告
    results["汇总报告"] = run_step(
        "自动汇总报告（results/实验结果汇总.md）",
        "make_report.py"
    )

    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    ok = sum(v for v in results.values())
    total = len(results)

    print("\n" + "=" * 80)
    print(f"  全流程运行完成！总耗时 {total_elapsed:.1f} s（{total_elapsed/60:.1f} 分钟）")
    print(f"  成功: {ok}/{total} 步")
    print("-" * 80)
    for name, success in results.items():
        icon = "✓" if success else "✗"
        print(f"  {icon}  {name}")
    print("=" * 80)
    print("\n  生成的主要文件：")
    print(f"  📊 python/*.png            — 各场景仿真图表（共约15张）")
    print(f"  📊 results/figures/*.png   — 拓展实验图表")
    print(f"  📋 results/实验结果汇总.md  — 自动汇总数据报告")
    print(f"  📋 results/*.csv           — 各实验原始数据")
    print()


if __name__ == "__main__":
    main()

