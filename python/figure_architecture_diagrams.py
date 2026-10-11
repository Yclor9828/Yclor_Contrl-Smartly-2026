"""
模糊自适应 PID 架构图与三大场景闭环控制结构图生成器（精修零重叠版）
===================================================================
生成用于 PPT 汇报的高清架构图：
1. A1_fuzzy_pid_overall_architecture.png —— 模糊自适应 PID 控制系统总体内部架构图（双环结构）
2. A2_scenarios_control_loops.png       —— 三大应用场景闭环控制回路结构图（标明控制器所在环节）
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_box(ax, cx, cy, w, h, text, subtext="", fc="#ffffff", ec="#1e3a8a",
             tc="#0f172a", fontsize=10.5, subsize=8.5, lw=1.8, zorder=5):
    """在 (cx, cy) 中心绘制圆角矩形框并写字（pad=0 保证精确尺寸，绝不外扩碰线）"""
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    if subtext:
        ax.text(cx, cy + h * 0.20, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=tc, zorder=zorder + 1)
        ax.text(cx, cy - h * 0.18, subtext, ha="center", va="center",
                fontsize=subsize, color="#334155", zorder=zorder + 1)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=tc, zorder=zorder + 1)
    return box


def draw_arrow(ax, x1, y1, x2, y2, label="", label_pos="top", offset=0.14,
               color="#1e293b", lw=1.8, style="-|>", rad=0.0, zorder=4, bg_label=True):
    """绘制带箭头的连接线并标注信号名"""
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=conn,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        zorder=zorder
    )
    ax.add_patch(arr)
    if label:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        bbox_cfg = dict(boxstyle="round,pad=0.12", fc="#ffffff", ec="none", alpha=0.85) if bg_label else None
        if label_pos == "top":
            ax.text(mx, my + offset, label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color, bbox=bbox_cfg, zorder=zorder + 2)
        elif label_pos == "bottom":
            ax.text(mx, my - offset, label, ha="center", va="top",
                    fontsize=9, fontweight="bold", color=color, bbox=bbox_cfg, zorder=zorder + 2)
        elif label_pos == "left":
            ax.text(mx - offset, my, label, ha="right", va="center",
                    fontsize=9, fontweight="bold", color=color, bbox=bbox_cfg, zorder=zorder + 2)
        elif label_pos == "right":
            ax.text(mx + offset, my, label, ha="left", va="center",
                    fontsize=9, fontweight="bold", color=color, bbox=bbox_cfg, zorder=zorder + 2)
    return arr


def draw_sum_node(ax, cx, cy, r=0.26, signs=("+", "-"), zorder=5):
    """绘制求和比较节点"""
    c = Circle((cx, cy), r, facecolor="#ffffff", edgecolor="#0f172a", lw=1.8, zorder=zorder)
    ax.add_patch(c)
    ax.text(cx, cy, "∑", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#0f172a", zorder=zorder + 1)
    ax.text(cx - r - 0.10, cy + 0.16, signs[0], ha="center", va="center",
            fontsize=11, fontweight="bold", color="#16a34a", zorder=zorder + 1)
    ax.text(cx - 0.16, cy - r - 0.14, signs[1], ha="center", va="center",
            fontsize=12, fontweight="bold", color="#dc2626", zorder=zorder + 1)


def generate_A1_overall_architecture():
    """图 A1：模糊自适应 PID 控制器内部双环架构图"""
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    # 标题
    ax.text(8.25, 8.45, "图 A1  模糊自适应 PID 控制系统总体架构图（外环模糊推理自整定 + 内环抗饱和 PID 控制）",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")

    # 1. 模糊自适应控制器大外框 (虚线背景高亮)
    ctrl_bg = FancyBboxPatch(
        (2.25, 1.45), 8.85, 6.45,
        boxstyle="round,pad=0.0,rounding_size=0.2",
        facecolor="#f8fafc", edgecolor="#2563eb", linewidth=2.2, linestyle="--", zorder=1
    )
    ax.add_patch(ctrl_bg)
    ax.text(6.675, 7.62, "【模糊自适应 PID 控制器核心架构 (Fuzzy Adaptive PID Controller)】",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1d4ed8", zorder=2)

    # 上半部分：模糊推理自适应外环背景
    fuzzy_bg = FancyBboxPatch(
        (2.45, 4.80), 8.45, 2.55,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.4, zorder=2
    )
    ax.add_patch(fuzzy_bg)
    ax.text(6.675, 7.12, "▲ 上层：Mamdani / Sugeno 模糊自适应在线调参外环",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#1e40af", zorder=3)

    # 下半部分：PID 控制内环背景
    pid_bg = FancyBboxPatch(
        (2.45, 1.65), 8.45, 2.35,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.4, zorder=2
    )
    ax.add_patch(pid_bg)
    ax.text(4.85, 3.78, "▼ 下层：带条件抗积分饱和的 PID 控制内环",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#15803d", zorder=3)

    # 求和节点 (比较环节)
    sum_x, sum_y = 1.35, 2.65
    draw_sum_node(ax, sum_x, sum_y, r=0.26)

    # 输入设定值 r(t)
    draw_arrow(ax, 0.20, sum_y, sum_x - 0.26, sum_y, label="目标设定值\nr(t)", label_pos="top", offset=0.12, color="#0f172a")

    # 误差分叉点
    branch_x = 2.05
    ax.plot([sum_x + 0.26, branch_x], [sum_y, sum_y], color="#0f172a", lw=1.8, zorder=4)
    ax.plot(branch_x, sum_y, marker="o", markersize=6, color="#0f172a", zorder=6)
    ax.text(1.72, sum_y + 0.18, "e(t)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#dc2626")

    # --- 上层模块（模糊推理链）---
    y_top = 5.85
    # 模块 1：微分与量化映射
    draw_box(ax, 3.45, y_top, 1.60, 1.40, "误差微分 & 量化",
             "e* = clip(ke·e)\nec* = clip(kec·de/dt)\n映射至 [-3, +3]",
             fc="#ffffff", ec="#2563eb", fontsize=9.8, subsize=8)

    # 模块 2：模糊化模块
    draw_box(ax, 5.65, y_top, 1.60, 1.40, "模糊化接口",
             "7个模糊子集\nNB~PB 隶属度计算\n(trapmf / trimf)",
             fc="#ffffff", ec="#2563eb", fontsize=9.8, subsize=8)

    # 模块 3：49条模糊规则推理机
    draw_box(ax, 7.90, y_top, 1.75, 1.40, "49条规则推理引擎",
             "代数积激活 α=μe·μec\n零阶 Sugeno 加权重心\n输出 ΔKp*, ΔKi*, ΔKd*",
             fc="#fef2f2", ec="#dc2626", tc="#991b1b", fontsize=9.8, subsize=8)

    # 模块 4：缩放放大与增益合成
    draw_box(ax, 10.00, y_top, 1.50, 1.40, "物理增益合成",
             "Kp = Kp0 + kup·ΔKp*\nKi = Ki0 + kui·ΔKi*\nKd = Kd0 + kud·ΔKd*",
             fc="#fffbeb", ec="#d97706", tc="#92400e", fontsize=9.5, subsize=7.8)

    # 上层箭头连接
    ax.plot([branch_x, branch_x], [sum_y, y_top], color="#2563eb", lw=1.8, zorder=4)
    draw_arrow(ax, branch_x, y_top, 3.45 - 0.80, y_top, label="e(t)", label_pos="top", offset=0.10, color="#2563eb")
    draw_arrow(ax, 3.45 + 0.80, y_top, 5.65 - 0.80, y_top, label="e*, ec*", label_pos="top", offset=0.10, color="#2563eb")
    draw_arrow(ax, 5.65 + 0.80, y_top, 7.90 - 0.875, y_top, label="μe, μec", label_pos="top", offset=0.10, color="#2563eb")
    draw_arrow(ax, 7.90 + 0.875, y_top, 10.00 - 0.75, y_top, label="ΔK*", label_pos="top", offset=0.10, color="#dc2626")

    # --- 下层模块（PID 控制核心）---
    # 模块 5：基准参数底座
    draw_box(ax, 3.85, sum_y, 2.00, 1.35, "初始基准参数底座",
             "Kp0, Ki0, Kd0\n(提供系统稳态基础控制力)",
             fc="#ffffff", ec="#16a34a", tc="#166534", fontsize=9.8, subsize=8)

    # 模块 6：动态 PID 控制器核心
    draw_box(ax, 7.20, sum_y, 2.65, 1.35, "动态 PID 核心 + 条件抗饱和",
             "u = Kp(t)e + Ki(t)∫edt + Kd(t)de/dt\n(输出饱和且同向时冻结积分)",
             fc="#dcfce7", ec="#15803d", tc="#14532d", fontsize=9.8, subsize=8)

    # 模块 7：执行器物理限幅
    draw_box(ax, 10.00, sum_y, 1.40, 1.35, "执行器限幅",
             "[u_min, u_max]\n安全边界保护",
             fc="#ffffff", ec="#15803d", tc="#14532d", fontsize=9.5, subsize=8)

    # 下层连线
    draw_arrow(ax, branch_x, sum_y, 3.85 - 1.00, sum_y, color="#15803d")
    draw_arrow(ax, 3.85 + 1.00, sum_y, 7.20 - 1.325, sum_y, label="e(t)&基准", label_pos="top", offset=0.10, color="#15803d")
    draw_arrow(ax, 7.20 + 1.325, sum_y, 10.00 - 0.70, sum_y, label="u0(t)", label_pos="top", offset=0.10, color="#15803d")

    # 从“物理增益合成”向下注入到“动态 PID 核心”的实时参数流（处于两层之间的专属通道 y=4.40）
    y_mid_channel = 4.40
    ax.plot([10.00, 10.00], [y_top - 0.70, y_mid_channel], color="#d97706", lw=2.0, ls="--", zorder=4)
    ax.plot([10.00, 7.20], [y_mid_channel, y_mid_channel], color="#d97706", lw=2.0, ls="--", zorder=4)
    draw_arrow(ax, 7.20, y_mid_channel, 7.20, sum_y + 0.675, color="#d97706", lw=2.0)
    ax.text(8.60, y_mid_channel + 0.16, "实时在线注入动态增益 Kp(t), Ki(t), Kd(t)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#b45309",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fffbeb", ec="#f59e0b", lw=1.0), zorder=6)

    # --- 外部物理被控对象与扰动环节 ---
    dist_x = 11.95
    c_dist = Circle((dist_x, sum_y), 0.23, facecolor="#ffffff", edgecolor="#dc2626", lw=1.8, zorder=5)
    ax.add_patch(c_dist)
    ax.text(dist_x, sum_y, "+", ha="center", va="center", fontsize=12, fontweight="bold", color="#dc2626", zorder=6)

    draw_arrow(ax, 10.00 + 0.70, sum_y, dist_x - 0.23, sum_y, label="u(t)", label_pos="top", offset=0.10, color="#0f172a")

    # 外部扰动框在上部
    ax.text(dist_x, 4.35, "外部负载/环境扰动\nd(t)", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color="#dc2626",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff1f2", ec="#dc2626", lw=1.2), zorder=6)
    draw_arrow(ax, dist_x, 3.92, dist_x, sum_y + 0.23, color="#dc2626", lw=1.8)

    # 被控对象框
    draw_box(ax, 13.90, sum_y, 2.35, 1.50, "被控工业物理对象",
             "场景1: 电加热炉 (FOPDT)\n场景2: 直流电机 (机电二阶)\n场景3: 双容液位 (SOPDT)",
             fc="#fef3c7", ec="#b45309", tc="#78350f", fontsize=10.2, subsize=8.2)

    draw_arrow(ax, dist_x + 0.23, sum_y, 13.90 - 1.175, sum_y, color="#0f172a")
    draw_arrow(ax, 13.90 + 1.175, sum_y, 16.20, sum_y, label="系统输出\ny(t)", label_pos="top", offset=0.12, color="#0f172a")

    # 反馈回路 (zorder=3 在底层，反馈框 zorder=6 盖住线条)
    fb_x = 15.45
    fb_y = 0.65
    ax.plot(fb_x, sum_y, marker="o", markersize=6, color="#0f172a", zorder=6)
    ax.plot([fb_x, fb_x], [sum_y, fb_y], color="#475569", lw=1.8, zorder=3)
    ax.plot([fb_x, sum_x], [fb_y, fb_y], color="#475569", lw=1.8, zorder=3)
    draw_arrow(ax, sum_x, fb_y, sum_x, sum_y - 0.26, color="#475569", lw=1.8, zorder=4)
    draw_box(ax, 8.40, fb_y, 3.40, 0.55, "传感器实时采样反馈回路 y(t)",
             fc="#f1f5f9", ec="#64748b", tc="#334155", fontsize=9.5, lw=1.4, zorder=6)

    plt.tight_layout()
    out_path = os.path.join(CURRENT_DIR, "A1_fuzzy_pid_overall_architecture.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图 A1 已保存: {out_path}")
    return out_path


def generate_A2_scenarios_control_loops():
    """图 A2：三大预设应用场景整体控制结构与控制器所在环节图"""
    fig, axs = plt.subplots(3, 1, figsize=(15.5, 10.5))
    fig.suptitle("图 A2  三大预设应用场景整体闭环控制结构图（红色高亮框为模糊自适应 PID 控制器所在环节）",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    scenarios_info = [
        {
            "title": "场景一：工业电加热炉温度控制系统 (FOPDT 一阶大惯性 + 纯滞后过程)",
            "ref": "目标温度 r(t)\n20℃→100℃→70℃",
            "ctrl_sub": "ke=0.0375, kec=0.4\nKp0=3.5, Ki0=0.08, Kd0=1.2\n输出加热功率 u ∈ [0, 100]%",
            "actuator": "可控硅调功器\n功率限幅 0~100%",
            "dist": "冷水注入扰动 d(t)\nt=110~130s, -25%",
            "plant_title": "电加热炉热力学传递函数模型",
            "plant_sub": "G(s) = 1.5 / (25s + 1) · e^(-5s)\n(静态增益 K=1.5℃/%, 时间常数 T=25s, 纯滞后 τ=5s)\n[E7拓展实验: t=80s 时 K 突变为 3.0]",
            "out": "炉膛温度\ny(t) [℃]",
            "sensor": "热电偶温度传感器 (采样周期 Ts = 0.2s)"
        },
        {
            "title": "场景二：直流伺服电机转速控制系统 (机电耦合二阶快速动态 + 变惯量过程)",
            "ref": "目标转速 ω_ref(t)\n0→100→130 rad/s",
            "ctrl_sub": "ke=0.03, kec=0.005\nKp0=0.6, Ki0=2.5, Kd0=0.01\n输出电枢电压 u ∈ [-24, +24]V",
            "actuator": "H桥 PWM 驱动器\n电压限幅 ±24V",
            "dist": "突加负载力矩 TL\nt=2.0~3.2s, 0.25 N·m",
            "plant_title": "直流伺服电机机电耦合双状态模型",
            "plant_sub": "电枢回路: La·di/dt = u - Ra·i - Ke·ω (Ra=1.2Ω, La=0.03H)\n机械回路: J(t)·dω/dt = Kt·i - B·ω - TL (J0=0.008 kg·m^2)\n[变惯量事件: t=3.8s 时 J 突增为 2.5×J0]",
            "out": "电机转速\nω(t) [rad/s]",
            "sensor": "光电编码器测速反馈 (采样周期 Ts = 0.001s)"
        },
        {
            "title": "场景三：双容串联液位过程控制系统 (SOPDT 二阶串级大惯性 + 纯滞后过程)",
            "ref": "目标液位 H_ref(t)\n20%→60%→40%",
            "ctrl_sub": "ke=0.0625, kec=0.5\nKp0=1.8, Ki0=0.025, Kd0=3.0\n输出阀门开度 u ∈ [0, 100]%",
            "actuator": "电动调节阀\n开度限幅 0~100%\n纯滞后 τ=3.0s",
            "dist": "副管突加进液冲击\nt=120~150s, +15%",
            "plant_title": "上下游双容水箱串联模型 (SOPDT)",
            "plant_sub": "G(s) = 0.96 / [(30s + 1)(15s + 1)] · e^(-3s)\n上游水箱1 (T1=30s) → 串联流入 → 下游水箱2 (T2=15s)",
            "out": "下游液位\nH2(t) [%]",
            "sensor": "投入式液位变送器 (采样周期 Ts = 0.5s)"
        }
    ]

    for idx, (ax, info) in enumerate(zip(axs, scenarios_info)):
        ax.set_xlim(0, 16.5)
        ax.set_ylim(0, 3.3)
        ax.axis("off")

        # 场景子标题背景条
        header_rect = FancyBboxPatch(
            (0.2, 2.75), 16.1, 0.44,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor="#1e3a8a", edgecolor="none", zorder=2
        )
        ax.add_patch(header_rect)
        ax.text(0.45, 2.97, info["title"], ha="left", va="center",
                fontsize=11, fontweight="bold", color="#ffffff", zorder=3)

        cy = 1.65
        # 1. 设定值输入
        draw_box(ax, 1.05, cy, 1.60, 1.15, "设定值输入", info["ref"],
                 fc="#f8fafc", ec="#475569", fontsize=9.5, subsize=8)

        # 2. 求和比较节点
        sum_x = 2.45
        draw_sum_node(ax, sum_x, cy, r=0.22)
        draw_arrow(ax, 1.05 + 0.80, cy, sum_x - 0.22, cy, color="#0f172a")

        # 3. 模糊自适应 PID 控制器（核心所在环节高亮）
        ctrl_x = 4.95
        draw_box(ax, ctrl_x, cy, 3.20, 1.35, "★ 模糊自适应 PID 控制器", info["ctrl_sub"],
                 fc="#fef2f2", ec="#dc2626", tc="#991b1b", fontsize=10, subsize=8, lw=2.4)
        draw_arrow(ax, sum_x + 0.22, cy, ctrl_x - 1.60, cy, label="e(t)", label_pos="top", offset=0.10, color="#dc2626")

        # 4. 执行机构
        act_x = 8.05
        draw_box(ax, act_x, cy, 1.75, 1.25, "执行机构环节", info["actuator"],
                 fc="#eff6ff", ec="#2563eb", tc="#1e40af", fontsize=9.5, subsize=8)
        draw_arrow(ax, ctrl_x + 1.60, cy, act_x - 0.875, cy, label="u0", label_pos="top", offset=0.10, color="#0f172a")

        # 5. 扰动注入点
        dist_x = 9.65
        c_d = Circle((dist_x, cy), 0.19, facecolor="#ffffff", edgecolor="#dc2626", lw=1.6, zorder=5)
        ax.add_patch(c_d)
        ax.text(dist_x, cy, "+", ha="center", va="center", fontsize=11, fontweight="bold", color="#dc2626", zorder=6)
        draw_arrow(ax, act_x + 0.875, cy, dist_x - 0.19, cy, label="u(t)", label_pos="top", offset=0.10, color="#0f172a")

        # 扰动文本框在上部
        ax.text(dist_x, 2.50, info["dist"], ha="center", va="center",
                fontsize=8.2, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fff1f2", ec="#dc2626", lw=1.0), zorder=6)
        draw_arrow(ax, dist_x, 2.22, dist_x, cy + 0.19, color="#dc2626", lw=1.5)

        # 6. 被控物理对象模型
        plant_x = 12.50
        draw_box(ax, plant_x, cy, 4.30, 1.35, info["plant_title"], info["plant_sub"],
                 fc="#fefce8", ec="#ca8a04", tc="#854d0e", fontsize=9.5, subsize=7.8, lw=1.8)
        draw_arrow(ax, dist_x + 0.19, cy, plant_x - 2.15, cy, color="#0f172a")

        # 7. 输出
        draw_arrow(ax, plant_x + 2.15, cy, 16.15, cy, label=info["out"], label_pos="top", offset=0.08, color="#0f172a")

        # 8. 底部反馈通路
        fb_x = 15.15
        fb_y = 0.42
        ax.plot(fb_x, cy, marker="o", markersize=5, color="#0f172a", zorder=6)
        ax.plot([fb_x, fb_x], [cy, fb_y], color="#475569", lw=1.5, zorder=3)
        ax.plot([fb_x, sum_x], [fb_y, fb_y], color="#475569", lw=1.5, zorder=3)
        draw_arrow(ax, sum_x, fb_y, sum_x, cy - 0.22, color="#475569", lw=1.5, zorder=4)
        draw_box(ax, 8.80, fb_y, 4.20, 0.46, f"反馈测量环节：{info['sensor']}",
                 fc="#f1f5f9", ec="#64748b", tc="#334155", fontsize=8.5, lw=1.2, zorder=6)

    plt.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.98, hspace=0.12)
    out_path = os.path.join(CURRENT_DIR, "A2_scenarios_control_loops.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图 A2 已保存: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_A1_overall_architecture()
    generate_A2_scenarios_control_loops()

