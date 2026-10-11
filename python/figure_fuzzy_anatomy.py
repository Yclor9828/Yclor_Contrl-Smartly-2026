"""
模糊推理机内部机制可视化 —— 解剖图
====================================
本脚本产出 2 张图，把模糊推理过程从黑盒变为白盒：

F1  隶属度函数图：7 个模糊子集（NB~PB）在论域 [-3,3] 上的形状
    ├── 子图(a)：输入 e* 的 7 条隶属度曲线（trapmf / trimf）
    └── 子图(b)：输入 ec* 的 7 条隶属度曲线（与 e* 相同的函数，复制展示）

F2  以典型工况点 (e*=2.1, ec*=-1.3) 为例的推理过程三联图：
    ├── 子图(a)：输入 e* 在各子集的激活强度（μe 竖线 + 条形）
    ├── 子图(b)：49 条规则的前件激活强度热力图 α_ij = μe(i)·μec(j)
    └── 子图(c)：三个输出量（ΔKp*, ΔKi*, ΔKd*）的加权推理结果（散点 + 重心标注）

运行：  python python/figure_fuzzy_anatomy.py
"""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

from fuzzy_config import fuzzy_block
from fuzzy_pid import FuzzyInferenceSystem, trimf, trapmf, _MF_FUNC

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

LABEL_CN = {
    "NB": "NB（负大）",
    "NM": "NM（负中）",
    "NS": "NS（负小）",
    "ZO": "ZO（零）",
    "PS": "PS（正小）",
    "PM": "PM（正中）",
    "PB": "PB（正大）",
}
LABEL_COLORS = [
    "#d62728", "#ff7f0e", "#8c564b",
    "#2ca02c",
    "#1f77b4", "#9467bd", "#e377c2",
]


def _calc_mf(x_arr, cfg, label):
    mf = cfg["mf"][label]
    fn = _MF_FUNC[mf["type"]]
    return np.array([fn(x, mf["params"]) for x in x_arr])


def figure_F1_membership(fis, cfg):
    """
    隶属度函数双联图（e* 和 ec* 的 MF 形状）。
    使用 fis.calc_membership 真实推理核计算，确保论域边界处饱和平顶特性完全准确。
    """
    x = np.linspace(-3.5, 3.5, 700)
    labels = cfg["labels"]

    # 真实系统按论域 [-3, 3] 计算，边界外平直饱和
    mu_grid = np.array([fis.calc_membership(xi) for xi in x])  # shape (N, 7)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    titles = [
        "(a) 误差输入 e* 的 7 个语言变量隶属度函数 μ_e",
        "(b) 误差变化率输入 ec* 的 7 个语言变量隶属度函数 μ_ec"
    ]

    for ax, title in zip(axs, titles):
        for j, (lbl, color) in enumerate(zip(labels, LABEL_COLORS)):
            y = mu_grid[:, j]
            ax.plot(x, y, color=color, lw=2.4, label=LABEL_CN[lbl], zorder=3)
            ax.fill_between(x, 0, y, color=color, alpha=0.08, zorder=2)

            # 在平顶中心或三角形顶点标注名称
            pk_indices = np.where(y > 0.999)[0]
            if len(pk_indices) > 0:
                px = x[int(np.mean(pk_indices))]
            else:
                px = x[int(np.argmax(y))]

            ax.text(px, 1.05, lbl, ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.2, alpha=0.95),
                    zorder=5)

        # 标出 [-3, 3] 标准物理论域边界
        ax.axvspan(-3.5, -3.0, color="#f0f0f0", alpha=0.7, zorder=1)
        ax.axvspan(3.0, 3.5, color="#f0f0f0", alpha=0.7, zorder=1)
        ax.axvline(-3.0, color="#888888", lw=1.2, ls="--", zorder=2)
        ax.axvline(3.0, color="#888888", lw=1.2, ls="--", zorder=2)
        ax.text(-3.25, 0.5, "论域外\n饱和区", ha="center", va="center", fontsize=8.5, color="#666666")
        ax.text(3.25, 0.5, "论域外\n饱和区", ha="center", va="center", fontsize=8.5, color="#666666")

        for xv in [-3, -2, -1, 0, 1, 2, 3]:
            ax.axvline(xv, color="#e0e0e0", lw=0.8, ls=":", zorder=1)

        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_xticklabels(["-3\n(NB)", "-2\n(NM)", "-1\n(NS)", "0\n(ZO)",
                             "+1\n(PS)", "+2\n(PM)", "+3\n(PB)"], fontsize=9.5)
        ax.set_xlim(-3.55, 3.55)
        ax.set_ylim(-0.02, 1.25)
        ax.set_xlabel("标准归一化论域 [-3, +3]", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("隶属度 μ", fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
        ax.grid(True, ls=":", alpha=0.4)

    # 顶部居中单行图例，绝不遮挡任何曲线
    handles, labels_leg = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=7, fontsize=9.5, frameon=True, facecolor="#fafafa", edgecolor="#cccccc")

    fig.suptitle("图 F1  Mamdani 模糊推理系统——输入变量隶属度函数 (Membership Functions)\n"
                 "（两端 NB/PB 为梯形 trapmf 并向外饱和延拓，中间 NM~PM 为三角形 trimf）",
                 fontsize=13.5, fontweight="bold", y=1.04)

    plt.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.96, wspace=0.18)
    path = os.path.join(CURRENT_DIR, "F1_fuzzy_membership_functions.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图 F1 已保存: {path}")
    return path


def figure_F2_inference_anatomy(fis, cfg):
    """
    以典型工况点 (e*=2.1, ec*=-1.3) 为例展示模糊推理全链条：
    (a) 输入模糊化交点求取
    (b) 49条规则前件激活热力图 (origin='lower' 保证坐标严密对齐)
    (c) 三个输出分通道加权求和与重心清晰化
    """
    E_STAR = 2.1     # e* = 2.1
    EC_STAR = -1.3   # ec* = -1.3

    labels = cfg["labels"]
    centers = cfg["centers"]
    n_label = len(labels)

    # 计算隶属度向量
    mu_e  = fis.calc_membership(E_STAR)
    mu_ec = fis.calc_membership(EC_STAR)
    alpha = np.outer(mu_e, mu_ec)  # shape (7, 7), alpha[i, j] 对应 (e=i, ec=j)

    # 输出清晰化值
    d_kp, d_ki, d_kd = fis.infer(E_STAR, EC_STAR)

    fig = plt.figure(figsize=(17.5, 6.2))
    # 划分 3 列主布局
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.97, bottom=0.12, top=0.80, wspace=0.30)

    # -----------------------------------------------------------------------
    # 子图 (a)：输入模糊化求交点（动态显示所有非零激活子集）
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    x_plot = np.linspace(-3.5, 3.5, 500)
    mu_curve = np.array([fis.calc_membership(xi) for xi in x_plot])
    for j, (lbl, color) in enumerate(zip(labels, LABEL_COLORS)):
        ax_a.plot(x_plot, mu_curve[:, j], color=color, lw=1.5, alpha=0.45)

    # 画 e* = 2.1 竖线
    ax_a.axvline(E_STAR, color="#d62728", lw=2.2, ls="--", label=f"误差 e* = {E_STAR}")
    # 动态找出所有被 e* 激发的子集
    act_e = [(j, labels[j], LABEL_COLORS[j], mu_e[j]) for j in range(7) if mu_e[j] > 1e-3]
    for j, lbl, col, val in act_e:
        ax_a.plot(E_STAR, val, marker="o", markersize=8, color=col,
                  markeredgecolor="black", markeredgewidth=1.2, zorder=6)
        ax_a.text(E_STAR + 0.22, val, f"μ_{lbl}={val:.2f}",
                  ha="left", va="center", fontsize=9.5, fontweight="bold", color=col,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2, alpha=0.95))

    # 画 ec* = -1.3 竖线
    ax_a.axvline(EC_STAR, color="#1f77b4", lw=2.2, ls=":", label=f"变化率 ec* = {EC_STAR}")
    # 动态找出所有被 ec* 激发的子集（包含 NB, NM, NS）
    act_ec = [(j, labels[j], LABEL_COLORS[j], mu_ec[j]) for j in range(7) if mu_ec[j] > 1e-3]
    # 对重合位置错开左右与高度，确保文字完全不重叠
    for j, lbl, col, val in act_ec:
        ax_a.plot(EC_STAR, val, marker="s", markersize=7.5, color=col,
                  markeredgecolor="black", markeredgewidth=1.2, zorder=6)
        if lbl == "NB":
            # NB 放在竖线左侧
            ax_a.text(EC_STAR - 0.25, val, f"μ_{lbl}={val:.2f}",
                      ha="right", va="center", fontsize=9.5, fontweight="bold", color=col,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2, alpha=0.95))
        elif lbl == "NM":
            # NM 放在竖线右侧
            ax_a.text(EC_STAR + 0.25, val, f"μ_{lbl}={val:.2f}",
                      ha="left", va="center", fontsize=9.5, fontweight="bold", color=col,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2, alpha=0.95))
        else:
            # NS (0.70) 放在竖线左侧
            ax_a.text(EC_STAR - 0.25, val, f"μ_{lbl}={val:.2f}",
                      ha="right", va="center", fontsize=9.5, fontweight="bold", color=col,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2, alpha=0.95))

    ax_a.set_xlim(-3.5, 3.5)
    ax_a.set_ylim(-0.02, 1.22)
    ax_a.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax_a.set_xticklabels(labels, fontsize=9.5)
    ax_a.set_xlabel("输入变量论域", fontsize=10.5, fontweight="bold")
    ax_a.set_ylabel("隶属度 μ", fontsize=10.5, fontweight="bold")
    ax_a.set_title("(a) 输入模糊化求交点\n[红色虚线=e*, 蓝色虚线=ec*]",
                   fontsize=11.5, fontweight="bold", pad=10)
    ax_a.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_a.grid(True, ls=":", alpha=0.5)

    # -----------------------------------------------------------------------
    # 子图 (b)：49 条规则前件激活热力图（修正 origin='lower'，严密对应 6 条规则）
    # -----------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])
    im = ax_b.imshow(alpha, cmap="YlOrRd", vmin=0, vmax=0.75, aspect="auto",
                     origin="lower", extent=[-0.5, n_label - 0.5, -0.5, n_label - 0.5])
    ax_b.set_xticks(range(n_label))
    ax_b.set_xticklabels(labels, fontsize=9.5)
    ax_b.set_yticks(range(n_label))
    ax_b.set_yticklabels(labels, fontsize=9.5)
    ax_b.set_xlabel("ec* 子集 (列)", fontsize=10.5, fontweight="bold")
    ax_b.set_ylabel("e* 子集 (行)", fontsize=10.5, fontweight="bold")
    ax_b.set_title("(b) 49条规则激活强度热力图\n[代数积合成: α_ij = μ_e(i) · μ_ec(j)]",
                   fontsize=11.5, fontweight="bold", pad=10)

    # 标注所有激活格子的数值并加上深色框
    for i in range(n_label):
        for j in range(n_label):
            val = alpha[i, j]
            if val > 0.02:
                txt_color = "white" if val > 0.40 else "black"
                ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=9.5, fontweight="bold", color=txt_color)
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                     fill=False, edgecolor="#002244", lw=1.8, ls="--")
                ax_b.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("前件激活度 α", fontsize=9.5)

    # -----------------------------------------------------------------------
    # 子图 (c)：三个通道的清晰化输出（分为 3 个并排横条子面板，彻底杜绝拥挤）
    # -----------------------------------------------------------------------
    # 在第 3 列内部嵌套 3 行微型子图
    gs_c = gs[2].subgridspec(3, 1, hspace=0.38)
    channels = [
        ("ΔKp*", d_kp, fis.rule_delta_kp, "#2ca02c", "比例增益 (大幅提升响应)"),
        ("ΔKi*", d_ki, fis.rule_delta_ki, "#ff7f0e", "积分增益 (负值防超调饱和)"),
        ("ΔKd*", d_kd, fis.rule_delta_kd, "#1f77b4", "微分增益 (正值抑制震荡)")
    ]

    for idx_ch, (name, val, rule_mat, color, desc) in enumerate(channels):
        ax_ch = fig.add_subplot(gs_c[idx_ch])

        # 计算各个论域中心累积激活度
        w_by_center = np.zeros(7)
        for ci in range(7):
            mask = (rule_mat == fis.centers[ci])
            w_by_center[ci] = alpha[mask].sum()
        norm_w = w_by_center / max(w_by_center.sum(), 1e-9)

        # 绘制柱状分布
        bars = ax_ch.bar(fis.centers, norm_w, width=0.45, color=color, alpha=0.75,
                         edgecolor="black", lw=0.6, zorder=3)

        # 重心垂线
        ax_ch.axvline(val, color="#d62728", lw=2.2, ls="--", zorder=4)

        # 标注加权重心数值
        ax_ch.text(val, 0.78, f" 加权重心\n {name} = {val:+.2f}",
                   ha="center", va="center", fontsize=8.5, fontweight="bold",
                   color="#990000",
                   bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec="#d62728", lw=1.2),
                   zorder=5)

        ax_ch.set_xlim(-3.6, 3.6)
        ax_ch.set_ylim(0, 1.15)
        ax_ch.set_yticks([0, 0.5, 1.0])
        ax_ch.set_yticklabels(["0", "0.5", "1.0"], fontsize=7.5)

        if idx_ch == 2:
            ax_ch.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax_ch.set_xticklabels(labels, fontsize=8.5)
            ax_ch.set_xlabel("输出论域单点中心", fontsize=9.5, fontweight="bold")
        else:
            ax_ch.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax_ch.set_xticklabels([])

        ax_ch.set_ylabel("权重比", fontsize=8)
        ax_ch.set_title(f"{name} 激活分布与重心清晰化 —— {desc}",
                        fontsize=9.5, fontweight="bold", loc="left", pad=4)
        ax_ch.grid(True, ls=":", alpha=0.4, zorder=1)

    # 总体大标题，居中高置，与子图标题保留充裕距离
    fig.suptitle(f"图 F2  模糊推理全流程机理解剖 (典型工况：e* = {E_STAR}, ec* = {EC_STAR})\n"
                 f"清晰化推理输出：ΔKp* = {d_kp:+.3f},  ΔKi* = {d_ki:+.3f},  ΔKd* = {d_kd:+.3f}",
                 fontsize=13.5, fontweight="bold", y=0.96)

    path = os.path.join(CURRENT_DIR, "F2_fuzzy_inference_anatomy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图 F2 已保存: {path}")
    return path




def main():
    cfg = fuzzy_block()
    fis = FuzzyInferenceSystem()
    print("=" * 80)
    print(" 模糊推理机解剖图生成器")
    print("=" * 80)
    figure_F1_membership(fis, cfg)
    figure_F2_inference_anatomy(fis, cfg)
    print("\n[完成] 模糊推理机解剖图已保存到 python/\n")


if __name__ == "__main__":
    main()

