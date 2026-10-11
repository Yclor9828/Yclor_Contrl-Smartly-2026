# MATLAB 子项目：模糊自适应 PID 控制器（直流伺服电机场景 + Simulink 落地）

本目录是 MATLAB 侧实现，包含三件事：

1. **场景二**：直流伺服电机多工况调速（突加负载 + 转动惯量摄动）的完整实验；
2. **跨语言核查**：用 MATLAB 独立重跑场景一，与 Python 结果**逐点比对**，
   证明两套实现（以及 Simulink 所用的 `.fis`）确实是同一套算法；
3. **Simulink 落地**：由代码自动搭建 `fuzzy_pid_simulink.slx`，并与脚本结果互证。

> 所有隶属度函数与 49 条规则均来自 `shared/fuzzy_config.json`，MATLAB 侧不再自带一份规则表——
> 这是修复旧版本"Python 规则库与 MATLAB 规则库不一致（尤其 ΔKi）"问题的根本措施。
> **不需要** Fuzzy Logic Toolbox 即可完成全部脚本级实验；只有 Simulink 部分用到官方模块。

---

## 1. 文件说明

| 文件 | 作用 |
| :--- | :--- |
| `fuzzy_config.m` | 读取 `shared/fuzzy_config.json`（向上搜索定位，persistent 缓存） |
| `fuzzy_pid_controller.m` | 模糊自适应 PID 类（配置驱动推理核、条件积分抗饱和、滤波微分、增益安全边界） |
| `fixed_pid_controller.m` | 固定 PID 类（与模糊类共用同一积分/抗饱和骨架，保证对比实验唯一变量是"自适应机制"） |
| `perf_metrics.m` | 窗口时域指标（与 Python `metrics.response_metrics` 完全同定义） |
| `integration_metrics.m` | 全程 IAE/ISE/ITAE、TV(u)、饱和占空比 |
| `run_thermal.m` / `run_motor.m` | 两个场景的仿真内核（时间推进顺序与 Python 一致） |
| `evaluate_thermal.m` / `evaluate_motor.m` | 场景指标汇总 |
| `build_controller.m` | 对照组装配（与 Python `controllers.py` 一一对应） |
| `load_tuned.m` | 读取 Python 侧寻优得到的"最优对照组"参数（两种语言共用） |
| `generate_fis.m` | 生成 `fuzzy_pid.fis` 并自动回读校验 |
| `verify_consistency.m` | 导出 61×61 推理网格，供 Python 侧逐点比对 |
| `thermal_simulation.m` | 场景一 MATLAB 实现 + 与 Python 轨迹逐点比对 |
| `motor_simulation.m` | 场景二主脚本（出图 + 指标 CSV） |
| `experiments_motor.m` | 电机侧拓展实验 E1M~E3M |
| `build_simulink_model.m` | 代码自动搭建 Simulink 模型 |
| `run_simulink_check.m` | 运行 Simulink 并与 Python 脚本结果比对 |
| `save_table.m` / `results_dir.m` | CSV 输出与结果目录工具 |

---

## 2. 运行方法

```matlab
cd matlab
generate_fis            % 1. 生成并校验 fuzzy_pid.fis（含与代码的逐点比对）
verify_consistency      % 2. 导出推理网格，供 Python 侧核查
thermal_simulation      % 3. 场景一 + Python/MATLAB 轨迹一致性核查（写入 results/cross_language_check.csv）
motor_simulation        % 4. 场景二主实验
experiments_motor       % 5. 电机侧拓展实验 E1M~E3M
build_simulink_model    % 6. 自动搭建 Simulink 模型
run_simulink_check      % 7. 运行 Simulink 并输出与 Python 的偏差
```

命令行（无图形界面）运行示例：

```powershell
$env:MATLAB_PREFDIR="$env:TEMP\mlprefs_dsh"   # 关键：给 MATLAB 一个可写的配置目录
cd matlab
& 'D:\Matlab\bin\matlab.exe' -batch "thermal_simulation; motor_simulation"
```

> ⚠️ 若不加 `MATLAB_PREFDIR`，在受限环境下 `matlab -batch` 会以
> `failed to load settings errors_warnings plugin` 启动失败。

---

## 3. 场景二模型与工况

$$\frac{di_a}{dt}=\frac{u_a-R_ai_a-K_e\omega}{L_a},\qquad
\frac{d\omega}{dt}=\frac{K_ti_a-B\omega-T_L}{J}$$

| 参数 | 取值 | 参数 | 取值 |
| :--- | :--- | :--- | :--- |
| $R_a$ | 1.2 Ω | $K_t$ | 0.08 N·m/A |
| $L_a$ | 0.03 H | $K_e$ | 0.08 V·s/rad |
| $B$ | 0.005 N·m·s/rad | $J_0$ | 0.008 kg·m² |
| 电压限幅 | ±36 V | 仿真步长 | 1 ms（电气时间常数 $L_a/R_a=25$ ms 的 1/25） |

**工况时序**（与配置文件一致，不再出现"README 与代码参数不符"）：

| 时段 | 工况 |
| :--- | :--- |
| 0 ~ 2.0 s | 空载起动 0 → 100 rad/s |
| 2.0 ~ 3.8 s | 突加负载力矩 $T_L=0.25$ N·m |
| 3.8 ~ 6.0 s | 转动惯量增至 2.5 倍，同时二次升速到 130 rad/s |

控制器参数：$K_{p0}=0.8,\ K_{i0}=18,\ K_{d0}=0.005$；$k_e=3/100=0.03$，
$k_{ec}=3/300=0.01$；$k_{up}=0.4,\ k_{ui}=10,\ k_{ud}=0.012$。

---

## 4. 关于 `.fis` 与 Simulink 的三个关键决定（都经过实测）

1. **隶属度函数用肩形**：NB/PB 必须用 `trapmf`（`[-4 -3 -2 -1]` / `[1 2 3 4]`），
   与代码的半梯形实现等价。旧版 `.fis` 用 `trimf [-5 -3 -2]`，其在 $x=-2$ 处隶属度已降为 0，
   与代码相差极大，必然导致 Simulink 与脚本结果对不上；
2. **前件合成为代数积**（`AndMethod='prod'`）：代码用 $\alpha_{ij}=\mu_e(i)\mu_{ec}(j)$，
   若写成 `min`，实测最大偏差 0.522（论域宽度仅 6.0）；
3. **清晰化为单点输出加权平均**（`Type='sugeno'` + `DefuzzMethod='wtaver'`）：
   Mamdani + `centroid` 计算的是聚合输出隶属度函数的**面积重心**，与代码的加权平均不等价，
   实测偏差可达 1.09。改为零阶 TS 后，官方 `evalfis` 与代码的偏差降至 $<10^{-14}$。

> 即：**"脚本—`.fis`—Simulink"三者一致性是被量化验证过的**，而不是"看起来一样"。
> 如果指导老师要求必须使用 Mamdani 面积重心法，请同时修改代码与 `.fis`，
> 并在报告中说明两套结果的差异来源与量级。

`generate_fis.m` 生成后会自动回读 `.fis`，在 21×21 网格上与 MATLAB 类逐点比对并打印最大偏差；
`verify_consistency.m` 导出 61×61 网格供 Python 侧核查（`python python/verify_consistency.py`）。

---

## 5. 电机侧拓展实验（`experiments_motor.m`）

| 编号 | 内容 | 观察点 |
| :--- | :--- | :--- |
| E1M | $J_0,K_t,K_e,R_a$ 各 ±30% 蒙特卡洛（150 组） | ITAE 分布与配对胜率 |
| E2M | 控制器采样周期 1/2/5/10 ms | 采样周期对超调与执行机构动作频度的影响 |
| E3M | 转速量测噪声 0~2 rad/s × 微分实现 | 理想微分 vs 滤波微分的 TV(u) 与跟踪精度权衡 |

结果写入 `results/exp*_motor_*.csv`，图写入 `results/figures/E1M~E3M*.png`。

---

## 6. 已知问题与注意事项

| 现象 | 原因与处理 |
| :--- | :--- |
| `matlab -batch` 启动失败 | 设置 `MATLAB_PREFDIR` 到可写目录 |
| 文件名以下划线开头报"文本字符无效" | MATLAB 要求 `.m` 文件名以字母开头 |
| `evalfis(x, fis)` 提示将来会被移除 | 新语法为 `evalfis(fis, x)`；脚本已做兼容 |
| Simulink 报代数环 | 反馈回路中使用理想微分所致；用 $Ns/(s+N)$ 滤波微分替代 |
| 与旧版报告数字不同 | 旧版报告的电机参数（150 rad/s、0.5 N·m、±24 V）与实际代码（100 rad/s、0.25 N·m、±36 V）不一致，且部分指标代码中并未实现；本版以 `results/*.csv` 为准 |
