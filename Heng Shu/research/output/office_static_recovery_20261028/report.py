import analyze as a
import pandas as pd
import numpy as np
import json,hashlib
from threadpoolctl import threadpool_limits
def main():
 p=a.O/'results';s=pd.read_csv(p/'state_results.csv');g=pd.read_csv(p/'shared_reference_gate.csv');h=pd.read_csv(p/'heldout_reference_summary.csv');hh=h[h.run>1];b=pd.read_csv(p/'between_runs.csv');d=pd.read_csv(p/'shared_reference_between_runs.csv');cl=pd.read_csv(p/'shared_pair_closure.csv');v=json.loads((p/'verification.json').read_text());audit=json.loads((p/'audit.json').read_text())
 order=[]
 for run in [1,2,3]:
  for state in a.STATES:
   r=a.cov(np.load(a.O/'packets'/f'shared_R{run}_{state}.npz')['corrected'])
   for q in [2,3,4]:
    angle,tau=a.peak(a.music(r,a.steer(state),q));order.append(dict(run=run,state=state,q=q,angle_deg=angle,tau_ns=tau))
 a.csv('shared_reference_model_order.csv',order)
 tab=g.pivot(index='state',columns='run',values=['peak_angle_deg','peak_tau_ns']).reindex(a.STATES)
 peak_table='| 状态 | R1 角度/时延 | R2 角度/时延 | R3 角度/时延 |\n|---|---|---|---|\n'
 for st in a.STATES:peak_table+='| '+st+' | '+' | '.join(f"{tab.loc[st,('peak_angle_deg',r)]:.0f}° / {tab.loc[st,('peak_tau_ns',r)]:.0f} ns" for r in [1,2,3])+' |\n'
 metrics={'test_timing_raw_iqr_ns':float(hh.raw_test_delay_iqr_ns.median()),'test_timing_corrected_iqr_ns':float(hh.corrected_test_delay_iqr_ns.median()),'test_phase_raw_sd_deg':float(hh.raw_test_phase_sd_deg.median()),'test_phase_corrected_sd_deg':float(hh.corrected_test_phase_sd_deg.median()),'test_real_correlation_after':float(hh.test_real_after.median()),'raw_cov_distance':float(b.raw_cov_distance.median()),'separate_ref_cov_distance':float(b.corrected_cov_distance.median()),'shared_ref_cov_distance':float(d.shared_cov_distance.median())}
 (p/'headline_metrics.json').write_text(json.dumps(metrics,indent=2))
 text=f'''# Office 静态人体 CSI 恢复分析

日期：2026-10-28。结论：**该记录支持恢复稳定的逐包相对相位、定时参考及跨轮可重复的分状态 MUSIC 谱；尚不能恢复经过物理验证的人体位置。**

## 1. 场景、海报与来源

- 原始目录：`office_broadside_raw6state_20260907/`。三轮各30秒，同一有人位置；不是三个人体点。
- 用户早前确认全程静止，本次进一步确认：28 mm 相邻间距、天线编号方向与后来一致，人体在TX–RX中点旁侧；人体横向距离、TX–RX间距、配对空场均未确认。
- 原配置0.54波长不再用作间距；原README/scene的空房标签不用于分析。原文件不改写，冲突和用户更正保存在此。
- `Dynamic Human Localization with Virtual Wi-Fi Antenna Arrays.pptx` 的海报文字与全部嵌入图已查阅，提到约−18°的次响应，同时明确它只是候选人体相关特征，使用场景相位/时延参考。其图形还注明探索性阵列假设。此次没有把旧图当作原始处理代码、硬件标定或人体方向真值，也没有以−18°选择峰。
- 海报含其他采集例子，其空房图不能仅因出现在同一海报就作为本Office记录的配对空场。
- 原始文件哈希与交付清单全部一致。连续记录{audit['records']:,}包，日志跨度{audit['logger_span_s']:.3f}秒；依事件而非等分文件识别3轮。
- HT20，5.2 GHz，Intel 5300原始`.dat`，3接收链，30频率组；不能套用HT40/PicoScenes解析器。全部包已解析并核查结构/链重排，实际分析保留 {int(s.valid_packets.sum()):,} 个边界保护包的索引范围，各状态等间隔抽取768包，合计13,824包做有界预处理。不是对全部包执行所有估计。

## 2. 本次方法及可恢复内容

方法名称：**Ratnam启发的多链公共相位—定时条件最大似然对齐 + 冻结首轮参考 + 分状态二维MUSIC**。这是原有Ratnam实现的HT20适配和本研究参考策略；不是原文完整算法复现，也不是Widar。

观测采用 x[p,m,k] ≈ H[m,k] exp(−j2π f[k] τ[p] −jψ[p])。H包含静止人体、背景和固定硬件响应。求解

`τ_hat = argmaxτ |Σ_mk conj(x[p,m,k]) ref[m,k] exp(−j2π f[k] τ)|`，

`ψ_hat = arg Σ_mk conj(x[p,m,k]) ref[m,k] exp(−j2π f[k] τ_hat)`，

再对同一包全部链施加相同的频率相关旋转。保留各链参考差异，不独立把天线相位拉平。不估计以Hz计的CFO；ψ是残余公共相位参数，不能将其全部归因于CFO。

对照分支：

- **A**：原始CSI，仅逐包Frobenius归一化。
- **B**：仅公共相位旋转；未中心化协方差理论不变，实测最大相对差{audit['cpe_covariance_max_relative_error']:.3g}，是程序不变量检查。
- **C**：每轮/每状态各用自己的首包和因果累积参考，公共相位与斜率校正。
- **D**：每状态只用R1前64个抽样包经C对齐后的均值建立冻结参考，R2/R3不更新此参考，公共修正通过门槛才应用。**不同状态仍有各自参考，D不是六状态公共零点或12阵元相干拼接。**

C在轮内很稳定，但不同首包的线性相位自由度导致跨轮偏移；D解决这组数据内可验证的参考不一致。H本身的未知固定相位/时延仍保留。

## 3. 参数、状态时间轴与验证约束

1. 实际频率为配置中的非均匀整数索引×312.5 kHz。将相关权重散布到真实整数频点，800点FFT求4 ns粗网格，再用原复数目标细化；完整相对定时周期3200 ns。合成无噪声检验误差<0.01 ns仅验证程序，不是实测精度。
2. 状态映射沿用 G1=[1,2,3], G2=[4,5,6], G3=[7,8,9], G4=[10,11,12], C1=[1,8,6], C2=[10,8,6]；先按接收置换字段重排。阵元 y=(6.5−m)×0.028 m，局部正前方朝TX；真实硬件相位与角度符号仍未用已知方向验证。
3. 状态周期25.112 ms，6状态周期150.672 ms。复用事件因果界、100 ppm漂移假设，边界保护约3.66–3.93 ms。设备/主机时间关系尚无外部同步证明；不是硬件级同时观测。缩到中央10 ms后，全频空间协方差变化最大{v['central_full_spatial_distance_max']:.6f}，支持当前稳定区选择。
4. 公共模型门槛沿用：前三链独立拟合差的p90≤5 ns和15°，前缀留出预测相关中位数≥0.9；实际包相关≥0.9且未触搜索边界。独立链拟合只用于诊断，不施加链独立修正。
5. 前64包用于初始化，其后每8包验证一次：用链1/2的部分频点拟合，在其余频点与链3组成的未参与拟合条目上检验。对R2/R3保留整轮身份，没有随机拆包训练。
6. R1建参考、R2/R3评估，没有利用人体位置或谱峰调参。**这是回顾性验证**：先看到C的跨轮偏移，才检查D；因此不称为此前完全未见的盲测或独立新实验。
7. MUSIC使用q=3，另列q=2/4敏感性；角度步长3°、时延步长8 ns。为避免非均匀频点上错误平滑，仅用10个可平移的长度10、间隔625 kHz窗口（负频5个、正频5个），不跨中间不规则间隔。全部30频点仍参与相位/定时拟合。窗口频率跨度5.625 MHz，不能把该谱的分辨能力等同于理想完整20 MHz处理。
8. MUSIC使用5.2GHz的窄带空间导向近似与真实阵元坐标；稀疏C状态有角度多解风险。三路径合成模型检验通过，但不验证真实硬件模型。频率平滑谱的周期1600 ns与原始30频点定时拟合的3200 ns周期不同，均不是物理传播时延零点。

## 4. 主要结果

### 4.1 公共参考恢复

18/18个轮次×状态通过C与D的模型门槛；本次抽样包全部通过接受规则。代表参考（前缀空间外积medoid）也通过18/18，但只是敏感性检查，没有替代首包分支或挑选最优结果。

下表取**R2/R3共12个状态级指标的中位数**，不是独立样本的置信区间。残差在未参与拟合的条目上重新估计，避免只报告优化目标自身变好。

| 检验量 | 修正前 | 冻结R1参考后 |
|---|---:|---:|
| 留出条目定时参数IQR | {metrics['test_timing_raw_iqr_ns']:.3f} ns | {metrics['test_timing_corrected_iqr_ns']:.3f} ns |
| 留出条目公共相位圆标准差 | {metrics['test_phase_raw_sd_deg']:.3f}° | {metrics['test_phase_corrected_sd_deg']:.3f}° |
| 留出条目实部归一化相关中位数 | {hh.test_real_before.median():.6f} | {metrics['test_real_correlation_after']:.6f} |

定时IQR减少约{100*(1-metrics['test_timing_corrected_iqr_ns']/metrics['test_timing_raw_iqr_ns']):.1f}%。这些是**相对参考下的信号一致性指标，不是传播时延精度、距离误差或定位精度**。全量拟合后再用同一条目评估会得到更小的约0.003 ns残差，见state_results；该值不作主要成功证据。

### 4.2 跨轮参考是否一致

对3对轮次×6状态，共18项比较，未中心化平滑协方差的对称相对距离中位数：

| 分支 | 中位数 |
|---|---:|
| A 原始 | {metrics['raw_cov_distance']:.6f} |
| C 各轮各自参考 | {metrics['separate_ref_cov_distance']:.6f} |
| D 冻结R1参考 | {metrics['shared_ref_cov_distance']:.6f} |

解释：原始数据的跨轮重复性本来较好；C引入的任意参考差反而放大跨轮变化；D恢复了统一的**逐状态相对**参考并略优于A。不得仅凭轮内残差下降就宣称算法整体改善。

原始三种链对的跨轮相位差中位数分别约{b.chain12_phase_deg.median():.3f}°、{b.chain13_phase_deg.median():.3f}°、{b.chain23_phase_deg.median():.3f}°，没有支持把E1–E4中近π模式移植到Office的证据。本次不执行任何链符号翻转。

全有效包对比768包抽样的平滑协方差差异最大{v['sample_full_cov_distance_max']:.4f}。该项量化抽样影响，未证明全部包的预处理效果等同于抽样结果。

### 4.3 谱峰稳定，但不是人体位置

D、q=3下全局最大谱峰如下。角度和时延均未物理校准；它们只是谱轴坐标，不是识别到的人体路径。

{peak_table}

观察：同一状态的三轮峰较重复，但不同状态的主峰方向不一致。支持的解释是静态CSI中有可重复结构；不能据此认定不同子阵选中了同一人体反射。固定链/线缆/开关响应、多径及稀疏阵列多解仍未分离。

模型阶数敏感性保留在model_order.csv和shared_reference_model_order.csv。例如G1在q=2与q=3可选择不同角度峰；因此最大峰不是稳健的人体判据。不按海报的−18°响应改模型阶数、符号或角度偏置。

### 4.4 共享天线为后续校准提供了什么

C1/C2中物理天线8与6均由相同接收链2/3观测；三轮该天线对相位闭合差中位数分别为{', '.join(f'{x:.3f}°' for x in cl.median_abs_phase_deg)}。它支持这条冗余约束的重复性，比直接假设所有切换状态随机失配更具体。

但这只是一条共享天线对检查，**没有完成整个状态图同步、跨周期相干拼接或固定支路响应标定**。不能据此宣布完整12阵元联合MUSIC已经恢复。

## 5. 能恢复到哪一步

| 层级 | 当前结论 | 证据/缺口 |
|---|---|---|
| 原始CSI解析、链重排、状态识别 | 可完成 | 全记录结构核查、哈希、事件边界与中央窗口敏感性 |
| 同状态逐包公共相位和相对定时稳定 | 有支持 | 全18状态模型检查；R2/R3留出条目验证 |
| 同状态跨轮共同参考 | 有支持 | 冻结R1参考后12状态验证、跨轮协方差改善 |
| 分状态二维MUSIC诊断谱 | 已生成 | 所有18状态、原始/C/D对照、模型阶数敏感性 |
| 完整12阵元相干阵列 | 尚未验证 | 整个共享图、切换时刻及支路固定响应仍缺验证 |
| 静止人体反射识别 | 不能确认 | 没有已确认配对空场或人体出现/离开证据 |
| 绝对AoA/超额ToF与二维定位 | 不能确认 | 固定响应/直达径参考、TX–RX及人体实测几何不足 |

相位稳定化会保留模板中的静止人体反射，也保留静态背景和硬件响应；本次没有减去有人模板。它不会自动把人体从背景中分离。已有数据也不能独立保证所有公共变化都来自硬件误差，而非共同信道变化。

## 6. 下一步最小补充

先找回本Office会话的TX–RX距离、人体横向偏移及天线/人体高度；不能用后来的3.65 m填入。确认是否存在这个会话的配对空场，不能用其他会话空场直接复数相减。

算法上的下一小步可用现有数据完成：以R1冻结整个共享天线约束图的公共相位/斜率关系，按周期和R2/R3检查冗余闭合与预测残差；仅当通过时生成明确标为未硬件校准的联合阵列诊断，不报告人体定位。

若找不到匹配空场/几何，建议新采一次保持接线和设备不变的“空场→已知旁侧站位静止人体→空场”，至少三轮，记录状态日志、实际距离和人体朝向；另采已知源几何的固定响应标定，并验证直达径是否能与环境多径区分。新会话标定可验证新数据，不自动校准过去这次Office。

## 7. 交付与复现

- `analyze.py`：独立Intel5300解析器快照、准确HT20频率、A/B/C、质量与谱图。
- `shared_reference.py`：D冻结R1参考与R2/R3预测验证。
- `verify.py`：源文件未变、中央窗口/抽样影响、合成MUSIC与数值检查。
- `report.py`：本报告、共享参考模型阶数检查和指标汇总。
- `results/source_hashes.csv`：来源与哈希；`packet_acceptance.csv`：包索引、接受标志与修正；`packets/`：有界输入/输出，可回溯session.dat记录索引。
- `results/state_results.csv`、`shared_reference_gate.csv`、`heldout_reference_summary.csv`、`between_runs.csv`、`shared_reference_between_runs.csv`、`shared_pair_closure.csv`：支撑以上各段数字。
- `figures/01_music_A`、`01_music_C`、`04_music_shared_reference`：三轮六状态原始、独立参考和固定参考谱。
- `figures/05_shared_reference_validation`：留出条目及跨轮对照。全部图有PNG和SVG。
- `poster_text.txt`、`poster_media_contact.png`：海报只作来源与方法边界核查。

从工作目录依次运行上述四个Python脚本。运行环境python，NumPy/SciPy/Pandas/Matplotlib，单线程BLAS。复现只写本分析目录，不改原始数据、旧结果、论文或Overleaf。

方法来源：既有 `output/ratnam_phase_timing_validation/METHOD.md` / `core.py`；其记录的论文为 Ratnam et al., Optimal Preprocessing of WiFi CSI for Sensing Applications, IEEE TWC 23(9),10820–10833,2024, DOI 10.1109/TWC.2024.3376332。本文区分公共相位/定时估计与绝对硬件校准；HT20、多链、参考策略和平滑窗口为本次适配。
'''
 (a.O/'REPORT.md').write_text(text,encoding='utf-8')
 sources=[]
 for f in sorted(a.O.rglob('*')):
  if f.is_file() and f.suffix not in ['.pyc'] and f.name!='artifact_manifest.csv':sources.append(dict(file=str(f.relative_to(a.O)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 pd.DataFrame(sources).to_csv(a.O/'artifact_manifest.csv',index=False)
 print(json.dumps(metrics));print('Report written')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
