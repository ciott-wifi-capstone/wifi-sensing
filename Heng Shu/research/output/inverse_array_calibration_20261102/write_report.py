from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;O=R/'scaled'
s=pd.read_csv(O/'descriptive_summary.csv');fit=pd.read_csv(O/'fit_summary.csv');conf=pd.read_csv(O/'dominant_conflict_summary.csv',index_col=0)
def table(split):
 g=s.query('split==@split')
 lines=['| 方法 | 均值 / ° | 中位数 / ° | P90 / ° | 最大偏离 / ° | ≤8°比例 |','|---|---:|---:|---:|---:|---:|']
 for r in g.itertuples():lines.append(f'| {r.model} | {r.mean_unassigned_peak_deviation_deg:.2f} | {r.median_deviation_deg:.2f} | {r.p90_deviation_deg:.2f} | {r.max_deviation_deg:.2f} | {r.fraction_within8:.1%} |')
 return '\n'.join(lines)
text=f"""# 共享天线初始相位逆向标定：有界验证报告

完成日期：2026-11-03。目录保留开始日期20261102。未修改论文或Overleaf。

## 结论

现有数据能够产生一套或多套共享的相位候选参数，但本轮没有得到可验证的硬件初始相位，也没有改善到可用的人体方向/二维定位。不能把拟合完成当作标定成功。

仅TX约束让强谱峰靠近0°，与对齐稳定传播分量相容，不能独立证实该分量就是直达波。使用已知人体位置的共享拟合仍不能使未拟合位置的峰可靠跟随人体。人体假设和镜像假设都能获得相近量级的训练损失，没有形成充分的位置辨识证据。

本轮不支持扩大相干12阵元拼接或计算位置误差。实际人体AoA MAE、物理超额ToF误差与位置误差仍为NA。输出的相位CSV属于模型依赖的候选，不能当作已测定设备标定表。

## 数据划分与覆盖

- 标定：记录0、12，即全局(1.64,+1)、(1.64,-1)。明确允许这两条记录的已知位置用于拟合。
- 同站位未拟合重复：记录3、15。
- 未拟合站位：记录21、24、30、33，来自±2 m两个位置，各两条记录。
- 全部记录此前均在旧分析中出现，本轮是回顾性验证，不是全新独立测试。
- 每状态原缓存768个选中包；前64包、偶数频率槽用于每相对RF链的0/π分支判定，其后704包用于本次协方差/拟合/谱计算。跨记录分支选择共用记录0参考，不用人体真值。
- 8条记录、48个状态；没有扩展到全部39条主数据。
- Office人体坐标和TX距离未知，不能提供绝对初始相位或AoA精度约束。本轮复用先前Office相对参考验证作为辅助事实，未对Office强加四点几何、未重跑Office。
- 空场与四点不属同一已确认硬件会话，未直接复数相减。
- 原始数据、旧处理结果均保留。source_manifest.csv对本次读取的缓存计算SHA256，原始源路径/原始哈希继承既有来源清单，不声称本轮再次重读全部原始文件。

## 比较了什么

1. 未校正基线；2.仅π分支修正。
3. 主分量=TX的圆统计图拟合；4.主分量=人体的圆统计图拟合。
5. 仅TX噪声子空间拟合；6.仅人体噪声子空间拟合。
7. TX+人体联合噪声子空间拟合；8. TX+镜像人体位置对照。

全部拟合共享12根物理天线的相位，天线1固定为0，仅11个自由相位；不逐点定制相位。它代表天线、线缆、开关及RF链的合并有效相位候选，不能分解为每一硬件部件的独立初相。

每条标定记录/状态允许一个公共时延参考偏移，共12个。联合模型中TX和人体共享该偏移，几何超额路径不能分别自由拟合。这些偏移为干扰参数，不是物理ToF标定值。没有把6个非同时状态直接拼成同步12阵元观测。

## 相位约束是否相容

两个标定位置在扣除各自假定的真实传播相位后，如果剩余主要是固定硬件响应，应该得到相容的相位约束。12条参考天线差分约束的结果：

| 假定主分量 | 两标定位置相位冲突中位数 | 最大冲突 |
|---|---:|---:|
| TX路径 | {conf.loc['tx','median']:.2f}° | {conf.loc['tx','max']:.2f}° |
| 人体路径 | {conf.loc['human','median']:.2f}° | {conf.loc['human','max']:.2f}° |

这不证明所有人体反射都不存在；它反对的是“观测主分量可直接当作人体反射，从而反推固定初相”这一简化假设。TX假设相对更一致，但仍有频率/状态残差和多径混合，不能将它提升为已识别直达径。

## 数值求解与非唯一性

原始ns变量的首轮32个起点中31个触及300次迭代上限。保留全部原结果，并在读取评价误差之前声明变量尺度修正：delta=20*u ns，目标函数、边界、8个初值与300次上限不变。scaled下32个拟合全部收敛。收敛只表示优化器停止条件满足，不代表全局最优或物理标定成立。

同一初值经不同数值尺度可落入不同局部极小值；未追加依据评价误差的初值搜索。原未收敛human-only候选损失比scaled最佳更低，也保留在根目录。它说明当前局部优化及模型辨识仍有局限，不能声称找到最优相位。

| 拟合模型 | 归一化训练损失 | 近优解数量 | 近优解最大相位RMS差 |
|---|---:|---:|---:|
"""
for r in fit.itertuples():
 text+=f'| {r.model} | {r.cost:.6f} | {r.near_optimal_starts} | {r.max_near_optimal_phase_rms_deg:.2f}° |\n'
text+="""
近优定义为损失不超过最佳+max(最佳×1%,1e-4)，只作描述；相位RMS差不是置信区间。TX-only与联合模型的目标包含不同假设数，不能简单用损失大小进行统计模型优选。相同双路径设定下，真实人体约束损失略小于镜像，但没有相应的可靠留出方向改善。

## 未拟合站位：±2 m四条记录

以下均为“未归属最强融合峰与人体几何角度的偏离”，不是人体AoA MAE。谱输出率为100%，已验证人体输出率为0%。不得据此强制输出位置。

"""+table('withheld_position')+"""

## 全部六条非标定记录

"""+table('all_noncalibration')+"""

## 与旧结果如何对比

先前基线16.98°仅对应记录3、15、24、33四条的未归属峰平均偏离。本次六条平均35.27°包含另外两条记录21、30，且统一使用排除前缀后的704包。不是同一评价集，不应将两数字写成算法退步。保持同样四条记录的比较见descriptive_summary.csv的previous_four_record_comparison行；当前四条基线依然约16.98°。

本轮六条结果不优于基线；单看±2 m集合，联合模型42.68°→41.11°的变化不足以支撑成功，而且未验证人体归属。不能只挑这一小幅变化，也不能用一次接近真值的峰替代总体有效率。

## 为什么扣除真实直达传播相位仍不够

多径观测H_m(f)=g_m(f)[a_0 exp(-j2πf d_m/c)+Σ_l a_l exp(-j2πf L_lm/c)]。
乘以exp(+j2πf d_m/c)后，括号中仍有“直达+所有相对多径”的复数和。因此其相角一般不是arg(g_m)。已知TX几何允许计算传播相位，却不提供已隔离的直达分量。

当前静止人体与房间背景都可能稳定存在；一个稳定模板可同时吸收两者。模板相位更稳定、或校正后强峰接近TX，都不是人体路径归属证据。已知人体位置可用于标定约束，但人体不是已知训练序列的独立发射源，其反射分量仍需从混合信道识别。

## 当前推荐

保留此前“π分支识别+逐频点场景参考”用于相位重复性诊断；它并非硬件预校准。此次共享TX相位拟合作为几何一致性对照保留，不替换为正式人体定位前端。避免继续使用逐点真值调相位或在真实位置附近挑峰。

仍可在现有数据上研究显式多径/频率相关响应模型，但当前结果没有证明该模型可辨识；增加参数可能吸收场景。任何后续研究须另设开发与评价协议，并把本轮已查看记录视作开发/回顾数据。不能在频率平滑后的协方差上随意添加逐频相位校正；需回到原始频率窗一致重算。

若要把有效相位候选变成可复用的物理标定，最小缺口是独立已知源方向的标定与未用于拟合的方向检查，或经已知线缆/分路器响应校正的硬件参考。人体归属还需要同一会话空场—有人—空场及重复站位对照。用户已确认目前没有这些额外数据，本轮没有等待补采或编造结果。

## 方法来源和状态

- Zhang等，Calibrating Phase Offsets for Commodity WiFi，DOI10.1109/JSYST.2019.2904714，IEEE Systems Journal14(1),661–664,2020（online2019）。作者PDF：https://zhangdh24.github.io/pub/CalWiFi-SJ.pdf
  文献以受控线缆/分路器实验研究相对链相位的π双分支；不把其硬件相位数值移植到本装置，也不把被动人体已知位置当成已验证已知源标定。
- 本次“共享相位噪声子空间拟合”是当前项目的约束拟合改造，目标是检验可行性，不声称复现原论文的完整算法或精度。
- Ratnam启发的公共误差处理及二维MUSIC基线复用既有缓存/代码，保留原未通过公共模型门槛时的基线，不新增CFO-Hz估计声明。
- 方法未使用Widar。

## 文件及复现

根目录保留首轮未收敛结果；scaled目录为数值尺度修正后的完整结果。
运行：fit_inverse.py → fit_inverse_scaled.py → predict_inverse.py → evaluate_inverse.py → write_report.py。
这些脚本依赖相邻static_human_aoa_validation_20261101中的缓存、common.py、music_baseline.py和Ratnam core.py。

- PROTOCOL.md、scaled/PROTOCOL.md：拟合/验证约定和预测前的数值修改说明。
- scaled/shared_phase_candidates.csv：48行有效相位候选，全部physical_calibration_verified=False。
- scaled/source_manifest.csv、branch_choices.csv：来源与跨记录离散分支。
- scaled/dominant_between_positions.csv：两标定位置相位冲突。
- scaled/multistart_fits.csv、calibration_path_losses.csv：所有起点与逐状态损失。
- scaled/frozen_predictions.json：引入留出真值前保存的预测哈希。
- scaled/descriptive_summary.csv、post_prediction_comparison.csv：不删失败的逐记录与分组结果。
- scaled/inverse_comparison、inverse_spectra、inverse_identifiability：PNG和矢量PDF。
- METHOD.md：观测与拟合公式。
- verification.json：数值、来源、预测冻结与图像核查记录。

分析并未证明全局不可定位；它说明当前所试共享初相方案尚不足以支撑人体定位。阶段结论是可复查的失败边界及下一步需要的约束，而不是人为制造≤8°。
"""
(R/'REPORT_共享初相逆向标定.md').write_text(text,encoding='utf8')
print('Report written')
