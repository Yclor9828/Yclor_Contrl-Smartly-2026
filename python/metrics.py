"""
控制系统性能评价指标库 (Control performance metrics)
==================================================
旧版本工程只在 README 中手写了超调、ITAE 等少数指标，且**代码里根本没有实现**
（例如"二次阶跃超调量 σ2"），导致文档数据无法复现。本模块提供统一、可验证的
指标计算，Python 与 MATLAB 两侧使用完全相同的定义。

指标定义（全部以"设定值窗户"为单位计算）
----------------------------------------
- 上升时间 tr      : 输出首次由 10% 上升到 90% 设定增量的时间 (s)
- 峰值时间 tp      : 到达峰值的时间 (相对窗口起点, s)
- 超调量 σ         : (峰值 - 设定值) / |设定增量| × 100%
- 调节时间 ts      : 最后一次离开 ±band 误差带的时间 (相对窗口起点, s)
- 稳态误差 ess     : 窗口末段 10% 时间内 (r - y) 的平均值
- 稳态波动 std     : 窗口末段 10% 时间内输出的标准差（衡量抖振）
- 最大偏差 max_dev : 窗口内 |y - r| 的最大值（扰动跌落深度即由此得到）
- 恢复时间 t_rec   : 最后一次离开 ±band 误差带后不再越界的时间
- IAE / ISE / ITAE : ∫|e|dt、∫e²dt、∫(t-t0)|e|dt（窗口内，时间从窗口起点计）
- TV(u)            : 控制量总变差 Σ|u(k)-u(k-1)|，衡量执行器动作频度与磨损
- 饱和占空比       : 控制量处于限幅边界的时间比例
"""

from __future__ import annotations

import csv
import os
import numpy as np

try:  # numpy>=2.0
    _trapz = np.trapezoid
except AttributeError:  # pragma: no cover - numpy<2.0
    _trapz = np.trapz


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _window(t, t0, t1, closed_end=False):
    """
    取窗口掩码。默认采用半开区间 [t0, t1)：因为设定值在 t1 处发生跳变，
    t1 时刻的样本属于**下一阶段**，若把 t1 计入本阶段，会把下一阶段的设定
    跳变误算成本阶段的"最大偏差"（这是旧版工程中一个隐性测量错误）。
    最后一个阶段可置 closed_end=True 以包含终点样本。
    """
    if closed_end:
        return (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    return (t >= t0 - 1e-9) & (t < t1 - 1e-9)


def _first_crossing(t, y, level, rising=True):
    """返回 y 首次跨过 level 的时刻，未跨过则返回 nan。"""
    if rising:
        idx = np.where(y >= level)[0]
    else:
        idx = np.where(y <= level)[0]
    return float(t[idx[0]]) if idx.size else float("nan")


def response_metrics(t, y, r, t0, t1, y_base=None, scale=None,
                     band=0.02, label="", closed_end=False) -> dict:
    """
    计算窗口 [t0, t1) 内的全部时域指标。

    参数
    ----
    y_base : 阶跃前的输出初值；给出时才计算上升时间与超调量
    scale  : 用于归一化误差带的"设定增量"；缺省取 |r_end - y_base|
    band   : 误差带宽度（相对 scale 的比例），默认 ±2%
    closed_end : 是否把 t1 时刻的样本计入本窗口（默认否，见 _window 说明）
    """
    m = _window(t, t0, t1, closed_end=closed_end)
    tt, yy, rr = t[m], y[m], r[m]
    if tt.size < 3:
        raise ValueError(f"窗口 [{t0}, {t1}] 内样本不足")

    r_ss = float(rr[-1])
    if y_base is None:
        y_base_eff = float(yy[0])
    else:
        y_base_eff = float(y_base)
    if scale is None:
        scale = abs(r_ss - y_base_eff)
    scale = float(scale) if abs(scale) > 1e-12 else 1.0

    err = rr - yy
    sign = 1.0 if r_ss >= y_base_eff else -1.0

    out = {
        "label": label,
        "window": (float(t0), float(t1)),
        "r_final": r_ss,
        "y_final": float(yy[-1]),
        "peak": float(yy[np.argmax(sign * yy)]),
        "ess": float(np.mean(err[tt >= tt[-1] - 0.1 * (tt[-1] - tt[0])])),
        "tail_std": float(np.std(yy[tt >= tt[-1] - 0.1 * (tt[-1] - tt[0])])),
        "max_dev": float(np.max(np.abs(err))),
        "t_max_dev": float(tt[np.argmax(np.abs(err))] - t0),
        "iae": float(_trapz(np.abs(err), tt)),
        "ise": float(_trapz(err ** 2, tt)),
        "itae": float(_trapz((tt - t0) * np.abs(err), tt)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }

    # --- 阶跃类指标 ---
    i_pk = int(np.argmax(sign * yy))
    out["t_peak"] = float(tt[i_pk] - t0)
    out["overshoot"] = max(0.0, sign * (out["peak"] - r_ss) / scale * 100.0)

    if y_base is None:
        out["rise_time"] = float("nan")
    else:
        lvl10 = y_base_eff + 0.1 * sign * scale
        lvl90 = y_base_eff + 0.9 * sign * scale
        c10 = _first_crossing(tt, sign * yy, sign * lvl10, rising=True)
        c90 = _first_crossing(tt, sign * yy, sign * lvl90, rising=True)
        out["rise_time"] = (c90 - c10) if (np.isfinite(c10) and np.isfinite(c90)) else float("nan")

    # --- 调节时间与恢复时间 ---
    tol = band * scale
    outside = np.abs(err) > tol
    if np.any(outside):
        i_last = int(np.where(outside)[0][-1])
        out["settling_time"] = float(tt[i_last] - t0)
        out["settled_after"] = bool(i_last < tt.size - 1)
    else:
        out["settling_time"] = 0.0
        out["settled_after"] = True
    out["recovery_time"] = out["settling_time"]
    return out


def control_effort(t, u, u_min=-np.inf, u_max=np.inf) -> dict:
    """执行器动作频度与饱和情况。"""
    du = np.abs(np.diff(u))
    span = max(abs(u_max - u_min), 1e-12)
    at_limit = (np.abs(u - u_min) < 1e-6 * span) | (np.abs(u - u_max) < 1e-6 * span)
    return {
        "u_mean": float(np.mean(u)),
        "u_std": float(np.std(u)),
        "tv": float(np.sum(du)),                      # 总变差
        "max_du": float(np.max(du)) if du.size else 0.0,
        "sat_duty": float(np.mean(at_limit)) * 100.0,  # 饱和占空比 (%)
    }


def whole_run_metrics(t, y, r, u, u_min, u_max) -> dict:
    """全程指标：采用绝对时间 t 的 ITAE（与工业惯例一致）。"""
    err = r - y
    out = {
        "itae_abs": float(_trapz(t * np.abs(err), t)),
        "iae": float(_trapz(np.abs(err), t)),
        "ise": float(_trapz(err ** 2, t)),
        "max_abs_err": float(np.max(np.abs(err))),
    }
    out.update(control_effort(t, u, u_min, u_max))
    return out


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def _disp_width(s: str) -> int:
    """终端显示宽度：CJK 字符按 2 列计。"""
    w = 0
    for ch in str(s):
        w += 2 if ord(ch) > 0x2E7F else 1
    return w


def _pad(s, width, align="left"):
    s = str(s)
    gap = max(0, width - _disp_width(s))
    if align == "right":
        return " " * gap + s
    return s + " " * gap


def format_table(headers, rows, aligns=None) -> str:
    """生成对齐良好的控制台表格（自动处理中英文混排宽度）。"""
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    widths = [_disp_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _disp_width(cell))
    sep = "-+-".join("-" * w for w in widths)
    lines = [" | ".join(_pad(h, widths[i], "left") for i, h in enumerate(headers)), sep]
    for row in rows:
        lines.append(" | ".join(_pad(c, widths[i], aligns[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def print_table(title, headers, rows, aligns=None):
    print("\n" + title)
    print(format_table(headers, rows, aligns))


def write_csv(path, rows, fieldnames=None):
    """把 list[dict] 写成 CSV（自动补全列名，保证列顺序稳定）。"""
    if not rows:
        raise ValueError("无用数据可写")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def relative_gain(a, b):
    """b 相对 a 的改善百分比（用于生成对比表最后一列）。"""
    if abs(a) < 1e-12:
        return float("nan")
    return (a - b) / abs(a) * 100.0
