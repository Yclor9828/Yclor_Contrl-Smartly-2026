# Python 子项目：模糊自适应 PID 控制器（工业电加热炉场景）

本目录是纯 Python（仅依赖 NumPy / Matplotlib / SciPy）实现的模糊自适应 PID
完整实验环境，**不依赖任何第三方模糊控制库**，推理系统、规则库、指标计算全部自研，
便于逐行核验与教学讲解。

> 与旧版本最大的差别：所有指标都由代码从仿真数据计算并写入 `results/*.csv`，
> 文档中的每个数字都能被复现；规则库与 MATLAB 侧、`fuzzy_pid.fis` 同源于
> `shared/fuzzy_config.json`，并配有自动一致性核查脚本。

---

## 1. 文件说明

| 文件 | 作用 |
| :--- | :--- |
| `fuzzy_config.py` | 读取唯一数据源 `shared/fuzzy_config.json`（向上搜索定位，带缓存） |
| `fuzzy_pid.py` | `FuzzyInferenceSystem`（零阶 TS 推理核）、`FuzzyAdaptivePID`、`ConventionalPID`、`PIDCore` 公共骨架 |
| `plants.py` | 被控对象：`FOPDTPlant`（加热炉）、`DCMotorPlant`（伺服电机），支持参数摄动与量测噪声 |
| `metrics.py` | 指标库：$t_r,t_p,\sigma,t_s,e_{ss}$、最大偏差与恢复时间、IAE/ISE/ITAE、TV(u)、饱和占空比、终端表格排版 |
| `simcore.py` | 仿真主循环（`run_thermal` / `run_motor`）与场景评价（`evaluate_thermal` / `evaluate_motor`），支持多速率零阶保持 |
| `controllers.py` | 对照组装配表（5 个对照组统一命名与配色） |
| `tuning.py` | ITAE 数值寻优：固定 PID（3 参数）与模糊控制器（6 参数）等预算整定 |
| `temperature_simulation.py` | 场景一主脚本：5 张图 + 指标/时序 CSV |
| `experiments.py` | 拓展实验 E1~E6（鲁棒性、采样周期、噪声滤波、敏感性、阶梯跟踪、收益边界） |
| `make_report.py` | 把 `results/*.csv` 自动汇总为 `results/实验结果汇总.md` |
| `test_fuzzy_core.py` | 21 项单元测试（配置一致性、隶属度覆盖、规则点对称、推理连续性、抗饱和、闭环无静差、指标正确性） |
| `verify_consistency.py` | 跨语言一致性核查（规则库语义、Python↔MATLAB↔.fis 推理逐点比对、`.fis` 文本核查） |

---

## 2. 运行方法（在仓库根目录执行）

```bash
python python/test_fuzzy_core.py            # 1. 单元测试（约 1 秒）
python python/tuning.py                     # 2. 参数寻优（约 5~10 分钟，生成 results/tuned_params.json）
python python/temperature_simulation.py     # 3. 场景一主实验（约 10 秒，出 5 张图）
python python/experiments.py                # 4. 拓展实验 E1~E6（约 5 分钟；--quick 可快速跑通）
python python/make_report.py                # 5. 自动汇总实验报告
python python/verify_consistency.py         # 6. 跨语言一致性核查（需先跑 MATLAB 的 verify_consistency）
```

---

## 3. 被控对象与工况

### 场景一：工业电加热炉（一阶惯性 + 纯滞后，FOPDT）

$$G(s)=\frac{K}{Ts+1}e^{-\tau s}=\frac{1.5}{25s+1}e^{-5s}$$

| 量 | 取值 | 说明 |
| :--- | :--- | :--- |
| $K$ | 1.5 ℃/% | 加热功率每增加 1%，稳态温升 1.5 ℃ |
| $T$ | 25.0 s | 炉体热惯性时间常数 |
| $\tau$ | 5.0 s | 纯滞后（电热丝通电到测温探头响应） |
| 执行器 | 0~100 % | 加热功率限幅（**限幅是积分饱和的根源，必须建模**） |
| 初始/目标 | 20 ℃ → 100 ℃ → 125 ℃ | $t=200$ s 二次升阶 |
| 扰动 | $-18$ ℃，$t=130\sim200$ s | 开炉门/冷料入炉造成的等效热负荷损失 |
| 仿真 | $dt=0.2$ s，280 s | 显式欧拉积分 |

---

## 4. 控制器设计要点

### 4.1 推理链（零阶 Takagi-Sugeno，与 `.fis`/Simulink 严格一致）

$$\alpha_{ij}=\mu_e(i)\cdot\mu_{ec}(j),\qquad
y^*=\frac{\sum_{i,j}\alpha_{ij}C(\mathrm{Rule}(i,j))}{\sum_{i,j}\alpha_{ij}}$$

- 隶属度函数：NB/PB 为肩形 `trapmf`，NM~PM 为三角形 `trimf`（参数定义在配置文件中）；
- 前件合成：**代数积** T-范数（对应 MATLAB `AndMethod='prod'`）；
- 清晰化：单点输出**加权平均**（对应 `Type='sugeno'` + `DefuzzMethod='wtaver'`）。

> 为什么不用 Mamdani + 面积重心？两者**不等价**：实测在同一网格上最大偏差 1.09
> （论域宽度仅 6.0）。本项目选择与嵌入式实现一致的加权平均，并让 `.fis` 与之对齐，
> 从而实现"脚本—`.fis`—Simulink"三者一致（实测偏差 $<10^{-14}$）。

### 4.2 参数量化

| 参数 | 场景一取值 | 依据 |
| :--- | :--- | :--- |
| $k_e$ | 0.0375 | 初始误差量程 $E_{max}=80$ ℃ → $3/80$ |
| $k_{ec}$ | 0.4 | 误差变化率量程按 $EC_{max}=7.5$ ℃/s 取 |
| $k_{up},k_{ui},k_{ud}$ | 1.2 / 0.035 / 2.5 | 基准增益的允许波动范围 ÷ 3（配置默认值，亦可用 `tuning.py` 寻优） |

### 4.3 三个必须处理的工程细节

1. **增量式积分**：$\Delta I(k)=K_i(k)e(k)\Delta t$ 逐步累加，避免 $K_i(t)$ 变化时历史积分被整体缩放造成控制量跳变；
2. **条件积分抗饱和**：输出饱和且误差继续推向饱和方向时冻结积分，误差反向时允许积分回退（`test_fuzzy_core.py::test_conditional_integration_can_unwind` 专门检验该性质）；
3. **微分滤波**：`kd_filter_tau > 0` 时对 $ec$ 施加一阶低通，抑制量测噪声被微分放大（实验 E3 给出定量代价）。

---

## 5. 实验设计与对照组

| 对照组 | 含义 | 作用 |
| :--- | :--- | :--- |
| `pid_nowindup` | 固定 PID，朴素积分 | 复现经典积分饱和现象（超调 16.7%） |
| `pid_aw` | 固定 PID + 条件积分抗饱和 | 与模糊控制器"同等工程条件"的公平基准 |
| `pid_opt` | 固定 PID，标称工况 ITAE 数值寻优 | **最强固定对照组**，避免"拿随手写的 PID 作对比" |
| `fuzzy` | 模糊自适应 PID（配置默认缩放因子） | 展示默认整定下的行为 |
| `fuzzy_opt` | 模糊自适应 PID（6 参数等预算寻优） | 与 `pid_opt` 同等预算，构成核心对比 |

指标定义见 `metrics.py` 头部说明；报告与图表中出现的全部数值均可由
`results/*.csv` 溯源。

---

## 6. 实验结论速览（详细数据见 `results/实验结果汇总.md`）

1. **标称工况**：等预算寻优后，`fuzzy_opt` 与 `pid_opt` 基本持平（ITAE 相差约 3%，
   最优固定 PID 略好）——这与理论一致：标称工况下自适应机制没有发挥余地。
   报告若在此处宣称"模糊控制大幅优于 PID"，需要先检查对照组是否被刻意削弱；
2. **参数摄动（±30%，300 组）**：两者统计上接近，但模糊控制器的**超调更小**
   （均值与最坏情况均更优），抗扰跌落基本相同；
3. **收益具有方向性**：当对象变得"更难控"（纯滞后增大、时间常数减小）时，
   模糊自适应明显占优；对象变"更容易"时固定 PID 略优——这解释了为什么
   不同文献会得出相反结论；
4. **宽范围阶梯跟踪**：5 个工作点上模糊控制器的超调均更小（最大降幅 65%），
   这是增益调度最经典的价值；
5. **实现代价**：模糊控制器控制量总变差 TV(u) 不高于固定 PID；但若使用理想微分，
   噪声会把 TV(u) 放大近 10 倍，必须配合一阶滤波微分。

---

## 7. 常见问题

| 现象 | 处理 |
| :--- | :--- |
| `FileNotFoundError: shared/fuzzy_config.json` | 在仓库根目录运行，或使用绝对路径调用脚本 |
| 图中中文显示为方块 | `plt.rcParams['font.sans-serif']` 中加入本机中文字体 |
| 寻优耗时较长 | `tuning.py` 中每个场景两组参数各做 3 次单纯形重启；可减小 `maxiter` 或 `restarts` |
| 结果与旧报告数字不同 | 旧报告的部分指标**代码里并未实现**（例如"二次阶跃超调 σ2"），且部分参数与代码不符；本版本以 `results/*.csv` 为准 |
