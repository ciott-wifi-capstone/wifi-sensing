> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# 独立初始相位标定准备包

先读 `先做这一步_独立初始相位标定清单.md`。新数据尚未采集。

`check_calibration.py` 是离线工具，不能控制或替你移动TX，也不启动采集设备。它只支持本项目已核实的 PicoScenes/Intel5300、HT40−、中心5230 MHz 的原始格式。采集命令继续以现场已核实的程序为准。

## 文件

- `TX独立验证点位.png/.svg`：计划点位图。
- `TX_positions_to_measure.csv`：计划坐标和实测待填字段。
- `capture_log.csv`：15个最小采集窗口。
- `session_template.json`：原目录映射、实际坐标、高度和同会话信息。
- `采前G1专项诊断.md`、`existing_*.csv`：来自现有冻结缓存的分析，不能误认成新测结果。
- `check_calibration.py`：原始输入检查、传播相位补偿、常数/仿射相位模型、冻结模板、G1–G4和12阵元主动源验证。
- `self_test_result.json`：合成软件验证，不是实测精度。

## 运行

复制 `session_template.json` 为 `session_filled.json` 后，填写真实坐标、实际高度、每轮目录和会话信息。不要把没有测量的字段标为已测量。每个run目录需有唯一`.csi`、`events.log`、`run.json`。程序不会修改这些原件。

在 PowerShell 中检查输入。

```powershell
& "python" -B -X utf8 "<private-workspace>/output/independent_phase_calibration_20261106/check_calibration.py" preflight --manifest "<private-workspace>/output/independent_phase_calibration_20261106/session_filled.json"
```

离线验证必须写入新的空输出目录，以保留以前的结果。

```powershell
& "python" -B -X utf8 "<private-workspace>/output/independent_phase_calibration_20261106/check_calibration.py" run --manifest "<private-workspace>/output/independent_phase_calibration_20261106/session_filled.json" --out "<private-workspace>/output/independent_phase_calibration_20261106/capture_check_01"
```

当前空模板执行preflight会返回 `pending_actual_capture_or_metadata`。这是预期结果，不是校准失败；还没有实际数据可检查。

运行完整检查时，先冻结 `model_freeze.json`，再产生侧向预测；`prediction_freeze.json`保存预测哈希，之后才计算主动源角度误差。`STATUS.json`可返回记录质量失败、开发模型失败或源验证结果。只有主动源通过仍不能宣称已识别人体，也不能说明绝对ToF零点已校准。

三根主对照G2与12阵元每测试窗口各用864个原始包；12阵元每24周期、每状态每周期6包，包括C1/C2开销。G1/G3/G4另作支路诊断；它们不是主要等预算对照。模型拟合最多每状态1024包，校准采集成本另列。

当前工具对主动源使用单源空间MUSIC与全局最强峰，适用性由未拟合角度检验。它不是人体选峰器。人体实验需回到原来的背景抑制、候选识别与负对照流程。

依赖现有 `output/coherent_array_validation_20261106/`、`output/music_subarray_gain_20261105/` 等解析和计算代码。没有安装新依赖或更改设备脚本。最小范围只验证0°和±20°；扩大方向范围时需先扩充manifest，再采集，不把近角度验证泛化成整个视场通过。
