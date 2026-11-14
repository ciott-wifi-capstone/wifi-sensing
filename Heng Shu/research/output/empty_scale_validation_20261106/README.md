> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# 空场优先的 PCD 噪声尺度验证

先阅读 [中文报告](空场跨轮模型验证报告.md)。本轮只计算 A–E 的 R1/R3 空场，不对 R2 人体数据应用新模型。NSI、NSD 均未通过冻结的空场检查，故人体分支停止；新角度为 NA。旧候选角度只作为历史等包数对照。

## 主要文件

- `protocol.json`：运行前确定的模型、时间/频率划分、门槛与停止条件。
- `validate_empty_models.py`：主模型计算，复用旧 PCD floor；两种模型没有逐点择优。
- `input_hashes.json`：五点缓存及来源清单、旧内核的 SHA256。
- `packet_budget.csv`：每方案每轮 864 个不同包，六态各 144 包，G2 控制 864 包。
- `model_parameters.csv`：包括负值在内的全部拟合系数及可用性。
- `held_empty_checks.csv`、`state_model_gate.csv`：所有留出检验及失败，未删除困难状态。
- `human_branch_eligibility.csv`：每点、每模型、每方案是否允许进入人体分析。
- `five_point_conditional_comparison.csv`：新角度/误差为空，历史候选有明确标签。
- `empty_stage_freeze.json`：决策时冻结的原始输出哈希。
- `report_empty_validation.py`：仅汇总结果、生成中文报告及 PNG/PDF 图，不估计新人体角度。
- `review/execution_audit.md`、`review/tolerance_equivalence.json`：独立公式、切分和数值容差复核。
- `diagnostics/空场尺度独立审计.md`：五种解释性背景模型及固定 R1 数字尺度对照。其划分不同于主模型，报告已单独标注。
- `verification.json`、`delivery_manifest.json`：交付检查与文件哈希。

## 复核与复现

环境：现有 `python`，使用 NumPy、Pandas、Matplotlib、threadpoolctl；无新增依赖。

在工作目录 `<private-workspace>` 运行只读数值审计：

```powershell
& 'python' -B -X utf8 'output/empty_scale_validation_20261106/review/audit_execution.py'
```

重新生成报告及图：

```powershell
& 'python' -B -X utf8 'output/empty_scale_validation_20261106/report_empty_validation.py'
```

主计算入口为 `validate_empty_models.py`。它会重写本目录派生结果，完整重算应先复制保存现有目录；源 CSI 和旧分析目录不修改。它依赖相邻的 `noncoherent_bc_followup_20261106` 内核及 `coherent_array_validation_20261106` 包缓存。交付清单在报告生成和复核后创建，重生成派生文件后应重新生成清单。

## 解释边界

本次检验的是跨轮空场变化能否由一个共同噪声尺度或三个链路噪声尺度预测。失败不等于原始数据完全不可用，也不是人体未检出率。每模型 35 个状态记录、630 个重叠频段/时间检验单元不是独立重复试验。全部数据已经用于开发，内部留出不能代替新采集的独立测试。

“正变化频点数不增加”只是一项模型资格检查，不等于空场方向误检率已验证。未放宽 PCD 门槛，未证明十二阵元收益，未修改论文或 Overleaf。
