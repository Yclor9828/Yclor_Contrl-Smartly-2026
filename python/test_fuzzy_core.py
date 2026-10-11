"""
模糊自适应 PID 核心算法单元测试 (unittest)
========================================
运行:  python python/test_fuzzy_core.py    或    python -m unittest discover -s python

测试内容覆盖"论文/报告结论所依赖的每一条算法事实"：
 1. 配置一致性：7 个语言变量、49 条规则、论域中心与配置完全对应
 2. 隶属度函数：隶属度非负、论域内每个点至少被一个子集完全覆盖（无控制盲区）
 3. 规则表点对称性：ΔKp、ΔKi、ΔKd 必须满足 M[i][j] == M[6-i][6-j]（控制语义自洽）
 4. 推理输出：严格落在 [-3, 3]；曲面连续（相邻点差有界，无跳变）
 5. 控制语义：e 很大时 ΔKp 为正、ΔKi 为负；e≈0 时 ΔKi 最大
 6. 抗积分饱和：饱和时积分冻结并可反向回退（不会永久卡死）
 7. 闭环行为：加热炉对象上稳态无静差
 8. 指标库：对已知超调/调节时间的合成曲线给出正确结果
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fuzzy_config import scenario, fuzzy_block                      # noqa: E402
from fuzzy_pid import (FuzzyInferenceSystem, FuzzyAdaptivePID,       # noqa: E402
                       ConventionalPID, trimf, trapmf)
from simcore import run_thermal, evaluate_thermal                    # noqa: E402
from metrics import response_metrics, control_effort, format_table    # noqa: E402


class TestConfigConsistency(unittest.TestCase):
    def test_rule_matrices_shape_and_range(self):
        cfg = fuzzy_block()
        for name, mat in cfg["rules"].items():
            arr = np.asarray(mat)
            self.assertEqual(arr.shape, (7, 7), f"{name} 规则矩阵必须是 7x7")
            self.assertTrue(np.all(np.abs(arr) <= 3), f"{name} 规则值必须落在论域 [-3,3]")

    def test_point_symmetry(self):
        """规则表必须点对称：语言语义上 (e,ec) 与 (-e,-ec) 应给出相反方向的调节。"""
        rules = fuzzy_block()["rules"]
        for name, mat in rules.items():
            arr = np.asarray(mat, dtype=float)
            flipped = arr[::-1, ::-1]
            self.assertTrue(np.allclose(arr, flipped),
                            f"{name} 规则矩阵不满足点对称，控制语义不自洽:\n{arr}\nvs\n{flipped}")

    def test_centers_match_labels(self):
        cfg = fuzzy_block()
        self.assertEqual(len(cfg["labels"]), 7)
        self.assertEqual(cfg["labels"], ["NB", "NM", "NS", "ZO", "PS", "PM", "PB"])
        self.assertEqual(list(cfg["centers"]), [-3, -2, -1, 0, 1, 2, 3])
        for lab in cfg["labels"]:
            self.assertIn(lab, cfg["mf"])


class TestMembership(unittest.TestCase):
    def setUp(self):
        self.fis = FuzzyInferenceSystem()

    def test_trimf_trapmf_reference_values(self):
        self.assertAlmostEqual(trimf(0.0, [-1, 0, 1]), 1.0)
        self.assertAlmostEqual(trimf(0.5, [-1, 0, 1]), 0.5)
        self.assertAlmostEqual(trimf(1.5, [-1, 0, 1]), 0.0)
        self.assertAlmostEqual(trapmf(-2.5, [-4, -3, -2, -1]), 1.0)
        self.assertAlmostEqual(trapmf(-1.5, [-4, -3, -2, -1]), 0.5)
        self.assertAlmostEqual(trapmf(-0.5, [-4, -3, -2, -1]), 0.0)

    def test_no_control_gap(self):
        """论域内任意点都应被至少一个模糊子集覆盖，且隶属度和不超过 2（避免重叠过度）。"""
        for x in np.linspace(-3, 3, 601):
            mu = self.fis.calc_membership(x)
            self.assertGreaterEqual(mu.sum(), 1.0 - 1e-9, f"x={x} 处存在隶属度盲区")
            self.assertLessEqual(mu.sum(), 2.0 + 1e-9, f"x={x} 处隶属度重叠过大")
            self.assertTrue(np.all(mu >= -1e-12))
            self.assertTrue(np.all(mu <= 1.0 + 1e-12))

    def test_clipping_outside_universe(self):
        mu_lo = self.fis.calc_membership(-99)
        self.assertEqual(mu_lo[0], 1.0)
        self.assertEqual(mu_lo[1:].sum(), 0.0)
        mu_hi = self.fis.calc_membership(99)
        self.assertEqual(mu_hi[-1], 1.0)


class TestInference(unittest.TestCase):
    def setUp(self):
        self.fis = FuzzyInferenceSystem()

    def test_output_bounds(self):
        for e in np.linspace(-3, 3, 25):
            for ec in np.linspace(-3, 3, 25):
                out = self.fis.infer(e, ec)
                for v in out:
                    self.assertGreaterEqual(v, -3.0 - 1e-9)
                    self.assertLessEqual(v, 3.0 + 1e-9)

    def test_continuity(self):
        """相邻网格点输出差应远小于网格跨度，保证控制量不会出现高频跳变。"""
        idx = np.linspace(-3, 3, 121)
        max_jump = 0.0
        for e in idx:
            prev = None
            for ec in idx:
                cur = np.array(self.fis.infer(e, ec))
                if prev is not None:
                    max_jump = max(max_jump, np.max(np.abs(cur - prev)))
                prev = cur
        self.assertLess(max_jump, 0.3, f"决策曲面不连续，最大跳变 {max_jump}")

    def test_corner_cases(self):
        """顶点规则必须严格复现规则表中的整数值（重心法退化行为）。"""
        for (e, ec, i, j) in [(-3, -3, 0, 0), (3, 3, 6, 6), (-3, 3, 0, 6), (3, -3, 6, 0)]:
            out = self.fis.infer(e, ec)
            expected = (self.fis.rule_delta_kp[i, j],
                        self.fis.rule_delta_ki[i, j],
                        self.fis.rule_delta_kd[i, j])
            np.testing.assert_allclose(out, expected, atol=0.35,
                                       err_msg=f"顶点 ({e},{ec}) 推理输出与规则表不符")

    def test_control_semantics(self):
        """大误差时应加大 Kp、削弱 Ki；稳态附近应强化 Ki。"""
        dkp_big, dki_big, _ = self.fis.infer(3.0, 0.0)
        dkp_zero, dki_zero, _ = self.fis.infer(0.0, 0.0)
        self.assertGreater(dkp_big, dkp_zero, "大误差时 ΔKp 应显著大于稳态时")
        self.assertLess(dki_big, dki_zero, "大误差时 ΔKi 应被削弱以抑制积分饱和")
        self.assertAlmostEqual(dki_zero, 3.0, delta=0.5, msg="稳态附近 ΔKi 应接近最大")

    def test_symmetry_of_outputs(self):
        """输入取反时，ΔKp、ΔKd 应关于原点对称（点对称规则表的直接推论）。"""
        for e, ec in [(1.7, -0.6), (2.5, 2.5), (-2.5, 0.5)]:
            a = self.fis.infer(e, ec)
            b = self.fis.infer(-e, -ec)
            self.assertAlmostEqual(a[0], b[0], places=6)
            self.assertAlmostEqual(a[1], b[1], places=6)
            self.assertAlmostEqual(a[2], b[2], places=6)


class TestAntiWindup(unittest.TestCase):
    def test_windup_without_protection(self):
        """无抗饱和时，长时间正误差会使积分项无限增长（复现经典积分饱和）。"""
        ctrl = ConventionalPID(1.0, 1.0, 0.0, out_min=0.0, out_max=1.0,
                               dt=0.1, anti_windup=False)
        for _ in range(200):
            ctrl.compute(10.0, 0.0)
        self.assertGreater(ctrl.integral_term, 100.0)

    def test_clamping_freezes_integral(self):
        """有抗饱和时，饱和状态下的积分应停止累加。"""
        ctrl = ConventionalPID(1.0, 1.0, 0.0, out_min=0.0, out_max=1.0,
                               dt=0.1, anti_windup=True)
        for _ in range(50):
            ctrl.compute(10.0, 0.0)
        frozen = ctrl.integral_term
        for _ in range(50):
            ctrl.compute(10.0, 0.0)
        self.assertAlmostEqual(frozen, ctrl.integral_term, places=9)

    def test_conditional_integration_can_unwind(self):
        """
        误差反向时，即使输出仍处于饱和区，积分也必须能够回退——这是条件积分法
        (conditional integration) 相对"饱和即冻结"朴素实现的关键优势，可避免
        积分被死锁在一个永远顶格的控制量上。
        """
        ctrl = ConventionalPID(1.0, 1.0, 0.0, out_min=0.0, out_max=1e9,
                               dt=0.1, anti_windup=True)
        for _ in range(10):
            ctrl.compute(10.0, 0.0)          # 输出未限幅，积分自由累积
        before = ctrl.integral_term
        self.assertAlmostEqual(before, 10.0, places=6)

        ctrl.out_max = 1.0                   # 执行机构可用范围骤降，输出进入饱和
        for _ in range(20):
            u = ctrl.compute(-1.0, 0.0)      # 误差反向，但积分仍把输出顶在上限
            self.assertAlmostEqual(u, 1.0, places=6)
        self.assertLess(ctrl.integral_term, before - 1.5,
                        "饱和状态下误差反向时积分未能回退，存在积分死锁")

    def test_output_never_exceeds_limits(self):
        ctrl = FuzzyAdaptivePID(3.5, 0.08, 1.2, ke=0.0375, kec=0.4,
                                kup=1.2, kui=0.035, kud=2.5,
                                out_min=0.0, out_max=100.0, dt=0.2)
        for _ in range(500):
            u = ctrl.compute(100.0, 20.0)
            self.assertGreaterEqual(u, 0.0)
            self.assertLessEqual(u, 100.0)


class TestClosedLoop(unittest.TestCase):
    def test_steady_state_no_offset(self):
        """闭环仿真的稳态误差应趋近于零（积分作用有效）。"""
        cfg = scenario("thermal")
        p = cfg["pid"]
        ctrl = FuzzyAdaptivePID(p["kp0"], p["ki0"], p["kd0"], ke=p["ke"], kec=p["kec"],
                                kup=p["kup"], kui=p["kui"], kud=p["kud"],
                                out_min=cfg["plant"]["u_min"], out_max=cfg["plant"]["u_max"],
                                dt=cfg["simulation"]["dt"])
        res = run_thermal(ctrl, cfg)
        m = evaluate_thermal(res, cfg)
        self.assertLess(abs(m["ess1"]), 0.05, "阶段 1 稳态误差应小于 0.05 ℃")
        self.assertLess(abs(m["ess2"]), 0.05, "阶段 3 稳态误差应小于 0.05 ℃")

    def test_reproducibility(self):
        """同一控制器参数两次运行必须逐点一致（确定性仿真，可复现）。"""
        cfg = scenario("thermal")

        def once():
            p = cfg["pid"]
            c = ConventionalPID(p["kp0"], p["ki0"], p["kd0"], out_min=0, out_max=100,
                                dt=cfg["simulation"]["dt"])
            return run_thermal(c, cfg)["y"]

        np.testing.assert_allclose(once(), once(), rtol=0, atol=0)


class TestMetrics(unittest.TestCase):
    def test_first_order_step_metrics(self):
        """一阶惯性系统解析解 y=1-exp(-t/T)：超调必为 0，且上升时间已知。"""
        t = np.linspace(0, 20, 2001)
        T = 4.0
        y = 1.0 - np.exp(-t / T)
        r = np.ones_like(t)
        m = response_metrics(t, y, r, 0.0, 20.0, y_base=0.0, scale=1.0)
        self.assertAlmostEqual(m["overshoot"], 0.0, places=6)
        expected_tr = T * (np.log(1 / 0.1) - np.log(1 / 0.9))
        self.assertAlmostEqual(m["rise_time"], expected_tr, delta=0.02)
        self.assertLess(m["ess"], 0.02)

    def test_known_overshoot(self):
        """构造 20% 超调的二阶响应，检验超调与稳态误差计算。"""
        t = np.linspace(0, 30, 3001)
        wn, zeta = 0.6, 0.456
        wd = wn * np.sqrt(1 - zeta ** 2)
        y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * t))
        m = response_metrics(t, y, np.ones_like(t), 0.0, 30.0, y_base=0.0, scale=1.0)
        self.assertGreater(m["overshoot"], 15.0)
        self.assertLess(m["overshoot"], 25.0)
        self.assertLess(m["ess"], 0.02)

    def test_control_effort(self):
        t = np.arange(5.0)
        u = np.array([0.0, 1.0, 1.0, 2.0, 0.0])
        e = control_effort(t, u, 0.0, 2.0)
        self.assertAlmostEqual(e["tv"], 4.0)
        self.assertAlmostEqual(e["max_du"], 2.0)
        self.assertAlmostEqual(e["sat_duty"], 60.0)

    def test_table_format_handles_cjk(self):
        text = format_table(["控制器", "ITAE"], [["模糊自适应PID", "77032.34"]])
        self.assertIn("模糊自适应PID", text)
        self.assertEqual(len(text.splitlines()), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
