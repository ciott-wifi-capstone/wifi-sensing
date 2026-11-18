from audit import *
def table(df):
 def fm(v):
  if isinstance(v,(float,np.floating)):return f'{v:.3f}'
  return str(v).replace('|','/')
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fm(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))
def main():
 p=pd.read_csv(R/'phase_model_validation.csv').query('run==3 and frequency_scope=="all"')
 g=pd.read_csv(R/'gain_phase_validation.csv').query('run==3');g['model']='phase_slope_gain'
 combo=pd.concat([p,g],ignore_index=True)
 agg=combo.groupby(['pair','model']).agg(phase=('phase_median_deg','median'),amp=('amplitude_distance','median'),spatial=('spatial_distance','median'),passed=('gate','sum')).reset_index()
 counts=pd.read_csv(R/'empty_candidate_recheck.csv').query('run==3').groupby(['pair','model']).stable_candidates.sum()
 e=agg[agg.pair=='E1_E3'].copy()
 e['stable_false']=e['model'].map(lambda m:counts.get(('E1_E3',m),14))
 e=e.set_index('model').loc[['raw','pi','constant_phase','phase_slope','phase_slope_gain']].reset_index()
 e=e[['model','phase','amp','spatial','passed','stable_false']]
 e.columns=['修正模型','链间相位残差中位数/度','幅度距离','协方差距离','通过状态/6','稳定空场假峰数']
 c=agg[(agg.pair!='E1_E3')&(agg.model=='pi')][['pair','phase','amp','spatial','passed']].copy()
 c.columns=['对照','相位残差/度','幅度距离','协方差距离','通过状态/6']
 inv=pd.read_csv(R/'occupied_within_invariants.csv')
 w=inv.groupby(['batch_first','index']).agg(phase=('phase_median_deg','median'),amp=('amplitude_distance','median'),spatial=('spatial_distance','median'),passed=('gate','sum')).reset_index()
 w.insert(0,'point',w.batch_first.map({0:'+1 m',12:'-1 m',21:'+2 m',30:'-2 m'}))
 w['complete']=w['index']!=2
 w=w[['point','index','phase','amp','spatial','passed','complete']]
 w.columns=['站位','记录','相位残差/度','幅度距离','协方差距离','通过状态/6','完整10秒']
 ses=pd.read_csv(R/'revised_session_manifest.csv')
 sess=ses[ses.role=='empty'].groupby('label').first().reset_index()[['label','phy','ifindex','successful_batch_driver_reload']]
 sess.columns=['空场','PHY','接口编号','批次前驱动重载成功']
 body=r"""# E1/E3 差异排查与四点同批次重复验证

2026-11-04。基于已有原始文件与日志的有界、回顾性分析；未采集新数据，未修改论文或 Overleaf。

## 1. 最重要的新发现

**采集流程在每个批次开始前自动卸载并重新加载了接收网卡驱动。此前把 run.json 内的“无初始化”理解为覆盖批次间，是证据范围上的错误，现予更正。**

E1、E3 的 operations.log 都记录了以下四项成功操作（returncode=0）：

    rmmod iwldvm
    rmmod iwlwifi
    modprobe iwlwifi debug=0
    modprobe iwldvm

随后执行 array_prepare_for_picoscenes，并重新建立 mon1。rx_kernel.log 同时记录了固件加载、对应 PHY 的建立及后续三个轮次的 CSI 流。operations.log、内核日志、radio_prepared.json 和 run.json 均与各自原 manifest.json 的 SHA256 和大小一致。

这支持以下修正：

- 用户描述的“同一场景、没有手动重启/重新接线/移动设备”继续保留；无需发生人工操作，脚本准备阶段仍可能重建无线参考。
- E1/E3 是同一 OS 启动记录、同一设备标识，但属于不同的接收驱动初始化段，不能当成未经中断的复数相位参考。
- run.json 的 “none; logger restarts only” 对应一批内部三轮；三个轮次的 before/after 身份确实保持不变。它没有覆盖批次准备操作。
- 两批开始的 ESP32 计时差约 4127.698 s，内核首个 CSI 流记录的计时差约 4127.708 s，两者约为 **68.8 分钟**；不采用文件标题日期解释为相隔15天。上述相对计时不证明这段时间内环境完全不变。

{sessions}

**驱动重载是参考可能改变的直接线索，但不足以单独解释 E1/E3 的全部差异。** E1/E2、E3/E4 也有批次重载，却能被有限 π 符号修正较好对齐。E1/E3 的额外幅度、相位及状态依赖变化仍不能唯一归因于驱动、接线、环境或多径；本次未把其中任何一种猜测写成已证实原因。

证据：driver_reload_evidence.csv（原始行号）、batch_provenance_hashes.csv（原 manifest 哈希核验）、revised_session_manifest.csv、E1_E3_session_evidence.json。协议原文不覆盖，证据更新另存 SESSION_EVIDENCE_UPDATE.md。

## 2. 状态与接收链：没有找到可应用的重排修复

- 28 个相关轮次日志共 12380 条 READY 事件，G1–G4/C1/C2 的顺序、seq_idx 与 chains 均符合已确认的六状态映射。
- 从 E1/E3 六个原始 CSI 文件重建 36 个状态缓存，包索引全部精确一致，归一化 CSI 最大差异小于 \(10^{-12}\)。
- 独立逐位解码 384 个分布在六文件不同时间位置的包，按物理接收链编号赋值，结果与现有向量化解码器完全一致。检查覆盖合法接收链重排。
- 原始帧确认 HT40、中心 5230 MHz、控制频率 5240 MHz、三接收链及30个频率组；未把 HT20 Office 参数代入。

另做了仅用于诊断的状态/链重排比较：用 run1 的归一化幅度拟合匹配，在 run3 检查。E1/E2、E3/E4 的原始状态/链顺序均优于其他候选。E1/E3 即使允许自由状态分配及链置换，最低平均幅度距离仍约 0.334；所得重排与日志映射不相符。因此没有依据采用这种“为相似度而重排”的结果。

代码意义上的状态映射一致，不等于独立验证了当时每一条实体线缆接法。日志只能证明软件命令和数据解析的一致性。

解析依据已核对 [Intel 5300 原始解码器](https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bfee.c) 与 [接收链重排逻辑](https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bf_file.m)。当前脚本使用每链2位掩码及逆排列，未因哪个排列“结果好”而改变。原始时间戳字段含义对照 [PicoScenes RxSBasic 定义](https://github.com/wifisensing/RXS-Parsing-Core/blob/master/RxSBasicSegment.cxx)。

## 3. 时间对齐：不支持把大差异归为几毫秒切分错误

六轮 E1/E3 内部的主机时间和 NIC 时间均严格递增。线性拟合后，主机/NIC 时间残差的1%–99%范围约为 −0.15 至 +0.20 ms，明显小于现有约1.47–2.04 ms的切换保护区。

READY 串口到达时间有约0–10 ms抖动，因此继续使用原有设备时间差和因果边界，而不是直接把串口到达时刻当切换时刻。相位跳变在1 ms分箱下的最大值落在现有切换边界附近，前后半段均如此；这只是粗时间一致性，不能称为亚毫秒外部时钟标定。

固定 E1→E3 的源开发相位/斜率模型后，将保护区改为4、6、9 ms，另在6 ms保护区下检查±3 ms偏移：

- run3 的归一化幅度距离中位数仍为约 **0.412–0.414**；
- 空间协方差距离仍为约 **0.840–0.846**；
- 各变体都为 **0/6 状态通过**。

因此这些合理范围内的切换边缘污染与常数时间偏移无法解释主要差异。没有选取最有利偏移去改有人数据。

![状态相位与时间敏感性](<{fig2}>)

左图是跨记录相位差的描述性圆统计，不是天线硬件标定值；低圆集中度的状态也完整保留。右图各点均使用同一冻结相位/斜率模型，只改变时间保护区或测试偏移。

## 4. 能否通过预校准初始相位修复 E1/E3

按模型复杂度逐级检验：

- raw：原始对照；
- pi：此前冻结的链间 π 符号；
- constant_phase：每条 RF 链一个固定相对相位；
- phase_slope：相对相位加频率线性斜率；
- phase_slope_gain：另加每条 RF 链一个固定正增益（观察到幅度失配后的有界扩展，另存 GAIN_EXTENSION.md）。

训练仅使用两空场的 run1、G1/G2 与三分之二频点；检查未参与拟合的频点、G3/G4/C1/C2 及完整 run2/run3。链1作为规范自由度参考，禁止每状态、每包自由旋转。

相位/斜率修正：
\[
x'_{m,k}=x_{m,k}\exp\{j[\beta_m+2\pi(f_k-f_c)\delta_m]\}.
\]
以源/目标链间共轭乘积的圆相关拟合 \(\beta_m,\delta_m\)，实际频率采样下的斜率主区间为 \([-400,400)\) ns。它是相对参考模型，不是物理传播时延。增益只用训练状态/频点的幅度比稳健统计，不能解释为已识别出的硬件响应。

工程门槛固定：可靠链对/频点支持至少0.5；相位残差中位数≤15°、P90≤30°，幅度距离≤0.20，空间协方差距离≤0.25。它们不是人体角度精度门槛；全6状态通过才支持全状态转移。同时用原有空场候选门槛检查假候选，未放宽门槛使模型过关。

E1→E3 第3轮结果（表中连续量先按状态计算，再取六状态中位数）：

{phase}

相位/斜率拟合使部分残差下降，但没有通过留出验证。增益扩展也只是把幅度距离从约0.412降到0.367，仍显著失配，不能把“拟合变好”写成校准成功。不同状态需要的相位变化差别明显，单一 RF 链相位/斜率模型不足。

正面对照：同一套门槛、相同方法在其他空场配对上可以成功。第3轮 π 符号模型：

{controls}

更复杂的连续模型并未稳定优于 π 模型，例如 E1/E2 的稳定空场方向假候选由 π 模型的0增加到相位/斜率模型的2。因此不能以模型复杂度代替证据。

![跨空场模型验证](<{fig1}>)

虚线是相位中位数及协方差距离的单项工程门槛；完整通过还取决于其他门槛，不能从曲线中位数直接判断每个状态都通过。右图为状态内稳定方向候选总数，不代表多个真实目标。

## 5. 四点提取的实际调整：先按驱动初始化段检验重复性

此次将同批次重复验证明确落实到四点各自第一批的 run1/run2/run3：
(0,1,2)、(12,13,14)、(21,22,23)、(30,31,32)。复用4条 run1，新增解析8个既有文件，共12轮；没有扩大到全部39轮。

- 每组内部 OS/PHY/ifindex 和 before/after 均相同；每个批次之前存在驱动重载。
- 保留原状态切分与实际阵元映射。每状态最多768个确定性时间抽样包，前64包排除后评分；不随机拆包。
- E1/E2 背景和门槛保持原冻结值。每组采用该组 run1 已冻结的有限符号向量，run2/run3 不重新拟合相位。
- 相位/斜率/增益模型没有应用到有人数据，因为它在 E1/E3 空场上已失败，且不具备转移到有人初始化段的独立证据。

同批次 run2/run3 相对 run1 的结果：

{within}

**这里的2.7°–8.9°是链间相位重复性残差，绝对不是人体 AoA 误差，不能用于声称达到≤8°定位目标。** 通过状态数采用同一套诊断门槛；部分状态未通过仍保留。

+1 m 的 run3（记录2）尾部缺约6.854秒，仅有约3.15秒实际记录。它的已有片段可以做诊断，但不能作为完整10秒成功复验。12轮中只有11轮完整。

![四点同批次重复性](<{fig3}>)

新结果支持“有人数据内部存在可重复空间结构”。但用两种原背景分别评分后，12轮均未形成三 G 子阵相容的方向簇，因此同批次重复人体候选仍为空。截短轮次没有从表中隐藏，也没有被当作完整有效输出。

相对于先前把不同批次的两条记录作为重复对照，本次改善在于：
1. 对每条记录注明驱动初始化段，区分同批次重复与跨初始化迁移；
2. 用同一 run1 有限符号参考评价该批后续完整轮次，避免每轮重新拟合吸收变化；
3. 将“信道结构可重复”与“人体路径可识别”分开验收。

没有经过独立标定的固定相位会使同批次三轮稳定地输出同一种偏差，重复本身不能赋予物理角度或人体归因。人体 AoA、物理超额ToF与位置误差继续为 NA。

## 6. 现在解决了什么，仍缺什么

已解决的是来源与处理解释问题：
- 找到并核验了批次准备阶段的自动驱动重载；
- 区分一批内部的连续采样与跨批次初始化；
- 排查了实际六状态命令、链重排、CSI解码和毫秒级切分敏感性；
- 验证简单固定相位/斜率/增益模型对 E1/E3 不充分；
- 用已有三轮数据证明四点同批次重复性，完整保留失败和截短记录。

没有解决的是从未知跨初始化、跨环境混合变化中唯一分离硬件响应和人体反射。允许每状态自由校相或依据站位挑峰可以降低表面误差，但会把真实传播相位吸收进校准，不是可验证修复。

现有数据下保留分状态、同频空间统计基线，E1/E3 分别建模，不合并成一个复数背景。未通过验证的修正不进入四点路径估计。若后续需要物理人体方向，应在一次接收机准备之后保持驱动、接口、频道连续，记录空场—有人—空场，并加入可隔离直达径的已知源校准；这是补充条件，本次未要求新采集，也未修改采集机器。

## 7. 文件与验证

- REPORT_E1_E3排查与四点同批次验证.md：本报告。
- audit.py：原始元数据/缓存重建、状态匹配、源训练相位模型。
- timing_checks.py：保护区与偏移敏感性、切换边界诊断。
- extra_checks.py：独立逐位解析、固定增益扩展、空场假候选检查。
- session_audit.py：批次 operations/kernel/manifest 交叉核验。
- within_batch.py：四点同批次三轮重复性与原冻结方向规则。
- verify.py：来源/冻结/独立解码及合成已知相位对照。
- metadata_audit.csv、revised_session_manifest.csv、driver_reload_evidence.csv、batch_provenance_hashes.csv：采集依据。
- phase_model_validation.csv、phase_by_state.csv、gain_phase_validation.csv、timing_sensitivity.csv：各模型及全部失败。
- occupied_within_records.csv、occupied_within_invariants.csv、occupied_within_validation.csv、occupied_within_candidates.csv：全部同批次输出。
- occupied_within_clusters.csv、occupied_within_repeats.csv：无通过候选，保留空表头。
- 三张图均提供 PNG 和矢量 PDF；图表已查看。
- verification.json、artifact_hashes.json：最终审计。

核验通过：128个唯一来源文件、28个轮次事件日志、12380个READY映射、36个精确重建状态缓存、384个独立逐位解析包。合成已知相位/斜率在训练状态拟合后，未训练状态协方差恢复误差小于 \(2\times10^{-16}\)。这只验证实现，不是实测硬件标定。

复现顺序：

    python audit.py
    python timing_checks.py
    python extra_checks.py
    python session_audit.py
    python within_batch.py
    python make_figures.py
    python verify.py
    python build_report.py

本目录保留与旧工程的显式依赖；原始数据、旧结果与论文未覆盖。当前规则是在多次查看过的资料上进行回顾性验证，不声称独立测试性能。
"""
 for name,value in {'sessions':table(sess),'phase':table(e),'controls':table(c),'within':table(w),'fig1':(R/'phase_transfer_validation.png').as_posix(),'fig2':(R/'phase_state_and_timing.png').as_posix(),'fig3':(R/'within_batch_repeatability.png').as_posix()}.items():body=body.replace('{'+name+'}',value)
 (R/'REPORT_E1_E3排查与四点同批次验证.md').write_text(body,encoding='utf8')
 notes="""# External format cross-checks (read 2026-11-04)
Intel original read_bfee.c uses a 0x3 mask for each two-bit antenna field; read_bf_file.m assigns CSI into the antenna positions. The current decoder uses the equivalent inverse-permutation gather.
https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bfee.c
https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/read_bf_file.m
PicoScenes RxSBasic v4 defines separate hardware and system timestamps. Source matches the observed raw segment layout.
https://github.com/wifisensing/RXS-Parsing-Core/blob/master/RxSBasicSegment.cxx
An upstream discrepancy was noticed in the currently accessible RXS-Parsing-Core CSISegment.hxx: the first antenna mask is 0x1 rather than the original Intel reference's 0x3. We did not switch algorithms to this divergent mask. All observed true 3-chain fields are valid permutations under the original 2-bit definition; independent scalar decoding checked the existing implementation.
https://github.com/wifisensing/RXS-Parsing-Core/blob/master/CSISegment.hxx
No package was installed and no parser was silently replaced from a web source.
"""
 (R/'REFERENCE_NOTES.md').write_text(notes,encoding='utf8')
 paths=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md','.csv','.png','.pdf','.npz'] and p.name!='artifact_hashes.json']
 (R/'artifact_hashes.json').write_text(json.dumps({p.name:hashfile(p) for p in paths},ensure_ascii=False,indent=2),encoding='utf8')
 print('report',len(body),'characters','artifacts',len(paths))
if __name__=='__main__':main()
