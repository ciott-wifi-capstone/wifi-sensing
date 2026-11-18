> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# 连续空场与四点静态恢复可行性

先读 [中文报告](REPORT.md)，再看 [阶段决策](results/decision.json) 和 [逐点状态](results/point_status.csv)。四点均为 A：未能建立可信背景参考。没有输出未经验证的位置。

## 本次产物

- 8组诊断图：`figures/01_...` 至 `08_...`，PNG和PDF各一份。
- 原四点39轮逐状态A/C图册：`figures/human_atlas/`，4份PDF共39页。谱是未校准参数，不是已认定人体路径。
- `results/`：来源、逐状态、完整轮次配对、参考失败、空场谱差、四点稳定性和验收记录。
- `parameters/`：参考模板、逐包候选与实际应用标记、跨记录预测及谱。
- `cache/`：六轮空场解析后的状态数据；原始CSI未复制或修改。
- `provenance/`：原处理代码、原报告快照、原始哈希及用户确认事实。

当前事实：E1与E3连续、同会话；文件名日期不是实际日期。有人数据与空场不是同一会话。E2/E4只作来源审计。冻结配置中的历史关系字段由 `provenance/confirmed_facts.json` 覆盖；数值门槛保持不变。

## 方法与结果的边界

原Ratnam条件ML的多链适配 + 分状态二维MUSIC诊断；没有使用Widar。B只修正公共相位，协方差不变。空场C全部被门槛拒绝，实际输出等于A。背景白化、人体新恢复和物理定位均因前置验证失败而未进入。空场谱差检验是空场零假设诊断，不是已部署检测器的虚警率。

## 顺序复现

运行环境：Windows，Python `python`，NumPy、SciPy、Pandas、Matplotlib、threadpoolctl、tabulate。原有数据与 `output/ratnam_phase_timing_validation/` 必须保留在其来源路径。

在本目录逐条执行，**不要并行执行写缓存与读缓存的脚本**：

```powershell
& 'python' setup.py
& 'python' audit_sources.py
& 'python' test_preservation.py
& 'python' diagnose_empty.py development
& 'python' select_reference.py
& 'python' diagnose_empty.py heldout
& 'python' compare_empty.py
& 'python' evaluate.py
& 'python' human_atlas.py
& 'python' make_figures.py
& 'python' build_report.py
& 'python' audit_delivery.py
```

`diagnose_empty.py` 已存在逐轮结果时跳过，因此当前目录中的上述命令是校验/恢复缺失产物流程。若要从头重复，复制源码、冻结配置及provenance到新的兄弟目录（不复制results/cache/parameters），再按顺序执行；不要删除本次记录或覆盖旧实验。

`human_atlas.py`只读取旧谱和协方差、核对原始文件哈希，不对39轮原始CSI重新估计。`qa_render.py`只渲染PDF并检查页数/内容，使用Codex bundled Python及Poppler；不修改PDF。完整运行顺序、依赖、输入来源与数值配置均可追溯。

## 运行更正

首次 `compare_empty.py` 运行时曾与缓存审计写入重叠，出现临时ZIP头不完整错误；原始数据未变。缓存审计完成后顺序重跑成功，最终CSV完整性及缓存可读性已纳入验收。该失败不是CSI数据质量证据。

论文、Overleaf和旧分析目录没有更新。建议补采同会话空场—有人—空场及独立固定响应标定，详见报告第7节。
