> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# Ratnam phase/timing validation

先读 [中文技术报告](REPORT.md)。本工程验证轮内相对 CPE/定时参考，**不输出已验证的物理 ToF 或定位精度**。原数据和论文不在本目录内修改。

## 入口与结论

- 主方法：式（27）的多链条件 ML 适配，首包相对参考；式（28）WLS 仅为对照。
- A：既有基线；B：仅公共相位；C：通过公共误差门槛后进行相位+定时修正。
- 四点全部 39 轮、234 状态，233 状态通过；真正空场 12 轮以及 A–D 首批 12 轮单独报告。
- 共计算 375 状态、524,099 包，A 的三个 G4 状态不足 128 包，保留失败记录。
- 主要数据支持轮内相对稳定化；不证明跨轮/跨状态标定，不保证真实微动完全不受影响。

## 文件结构

| 文件 | 用途 |
|---|---|
| `METHOD.md` | 原文依据、公式、适配及冻结条件 |
| `REPORT.md` | 中文实测结论与图表解释 |
| `VALIDATION.md` | 对目标逐项核验、失败边界、视觉检查 |
| `config.json`, `config.sha256` | 冻结配置及校验值 |
| `input_manifest.csv` | 90 个唯一来源、标签、初始处理安排 |
| `bounded_manifest.csv` | 最早两批的初始 30 轮 |
| `extension_manifest.csv` | 通过阶段检查后增加的 33 轮 |
| `provenance/` | 解析器、旧方法与数据清单快照及来源哈希 |
| `core.py` | 公共误差拟合、门槛、前向校正、协方差及 MUSIC |
| `test_synthetic.py`, `test_extended.py`, `test_fades.py` | 合成正例和负例 |
| `run_real.py` | 分角色运行，按完整轮次保存检查点 |
| `augment.py` | 从原 CSI 和保存参数重建 A/B/C 的内部稳定性诊断 |
| `summarize.py`, `make_figures.py` | 结果汇总及七组 PNG/SVG 图 |
| `audit.py` | 重新核对输入、配置、掩码、失败回退和数值不变量 |
| `package.py` | 文件哈希清单及可校验 ZIP；不包含原始 CSI |

`input_manifest.csv` 的 role/scheduled 是初始安排；最终处理安排以两个执行清单的并集及 `results/run_summary.csv` 为准。27 个未处理文件是补充数据，本阶段没有扩大至全部 A–D 批次。

## 环境与运行

验证环境：Windows，Python 3.12.7，NumPy 1.26.4，SciPy 1.13.1，Matplotlib 3.9.2，另需 pandas、threadpoolctl。使用 `python`，主分析限制每进程一个 BLAS 线程；部分不同角色曾并行运行，记录耗时合计不是墙钟总耗时。

在工作目录 `<private-workspace>` 的 PowerShell 中运行以下命令。路径依赖原始数据及旧协方差缓存，交付 ZIP 不包含这些大文件。完整输入路径已保存在清单中。

```powershell
$ratnamPy = 'python'
$ratnamDir = 'output/ratnam_phase_timing_validation'
& $ratnamPy "$ratnamDir/test_synthetic.py"
& $ratnamPy "$ratnamDir/test_extended.py"
& $ratnamPy "$ratnamDir/test_fades.py"
& $ratnamPy "$ratnamDir/audit.py"
```

上述合成测试很小，审计只核对原文件及已有结果，不重新运行实测估计。已有主分析按运行检查点复用。需要完整重算时，先复制工程到另一个 output 子目录，保留全部输入清单、provenance、配置以及 development_review/expansion_decision 决策文件；在副本的空 results/packets 目录重算，不覆盖本次归档结果。以下为原处理顺序，开发检查与扩展决定必须先于对应下一阶段：

```powershell
& $ratnamPy "$ratnamDir/run_real.py" development
# 检查 development 结果并记录 development_review.json；不得在留出数据上调参。
& $ratnamPy "$ratnamDir/run_real.py" held_out
& $ratnamPy "$ratnamDir/summarize.py"
& $ratnamPy "$ratnamDir/decide_expansion.py"
# 仅在 expansion_decision.json 的证据支持扩展时执行：
& $ratnamPy "$ratnamDir/run_real.py" expansion
& $ratnamPy "$ratnamDir/run_real.py" supplementary
& $ratnamPy "$ratnamDir/augment.py"
& $ratnamPy "$ratnamDir/summarize.py"
& $ratnamPy "$ratnamDir/make_figures.py"
& $ratnamPy "$ratnamDir/audit.py"
```

默认 `run_real.py` 遇到相同配置的已有完整轮次会跳过，不会强制重算。修改算法、依赖或输入后必须新建结果副本并重新验证，不能仅依赖旧检查点。`prepare.py` 是首次冻结入口；不要用它悄悄替换本次清单或配置。

## 每包参数和指标语义

`packets/{index:03d}_{state}.npz` 包含：

- `packet_id`：原解码包索引；`host_ns`、`nic_timestamp`、`window_id`：原观测时间与切换窗口。
- `B_tau_ns`（恒为零）、`B_psi_rad`、`B_correlation`：B 的逐包参数。
- `C_tau_ns`、`C_psi_rad`、`C_correlation`、`C_edge`：C 候选参数及质量标志。
- `C_applied`：实际应用掩码。False 时保留基线包；不能无视掩码全部应用候选参数。
- `prefix_warmup`：前 64 个状态包的启动期。
- `first_packet_reference`、`reference_prefix`：首包参考及留出特征诊断使用的早期参考。
- `cov_A/B/C`、`spectrum_A/B/C`：所有有效状态包的实际 A/B/C 结果，包含被拒绝但保留的原包。
- `prediction`：后续每八包一次的特征留出检查。列依次为包偏移、训练特征延迟、测试特征延迟、两者环形差、A 的实相关/模相关/相位、B 的三项、C 候选的三项、边界标志。这里不是实际 C 全链更新参数，也不是物理误差真值。
- `wls_comparison`：包偏移、WLS−ML 延迟差(ns)、相位差(°)、WLS 圆残差(°)、ML 耗时(s)、WLS 耗时(s)。WLS 耗时未含 ML 初值。

重建：用快照解析器解码 raw，按 `packet_id` 取包，按 `core.unit()` 归一化。B 全部应用 `exp(j*(2*pi*f*tau + psi))`；C 只对 `C_applied=True` 的包应用。tau 存储单位 ns，乘 1e-9 后进入指数。C 在内部重复归一化会产生约 1e-16 浮点差异，不属于科学改善。

`state_summary.csv` 中所有 `pred_*` 均是候选的条件模型诊断；`actual_output_stability.csv` 是重建后针对同轮参考的内部检查。二者均不等于已知物理误差。所有谱轴仍未完成硬件校准，最大峰未被命名为人体路径。

`packet_acceptance.csv` 的拒绝原因计数可重叠（低相关和边界），不能直接相加当作唯一拒绝包数。`wls_diagnostics.csv` 是风险提示，不是真值标注的展开错误。三份 `synthetic*.json` 的 all_pass 意味着测试预期被满足，其中包含展示失真的反例，不意味着方法对一切信号无失真。

## 引用与可移植性

论文依据与公式对照在 METHOD/REPORT；本目录不重新分发受版权保护的论文 PDF。数据路径为当前 Windows 主机绝对路径。移至其他主机时需显式更新副本的路径映射并重新核验哈希，不能忽略来源变化。

交付 ZIP 包含代码、图表、参数、清单和审计；不含原 CSI。`artifact_manifest.csv` 可逐项检查归档内容；ZIP 的最终 SHA256 在旁边的 `.zip.sha256` 文件中。
