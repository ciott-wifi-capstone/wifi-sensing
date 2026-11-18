from spatial_model import *
def table(d):
 def fmt(x):
  if isinstance(x,(float,np.floating)):return f'{x:.3f}'
  return str(x).replace('|','/').replace('\n',' ')
 cols=list(d.columns)
 return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
def main():
 es=pd.read_csv(R/'empty_summary.csv')
 w=es[es['mode']=='within'][['source','states','matrix_change_states','raw_false_peaks','stable_false_peaks','median_distance']].copy()
 w.columns=['空场','状态数','协方差超阈状态','初步方向峰','稳定方向峰','协方差距离中位数']
 tr=es[(es.source!=es.target)&(es.run==3)][['source','target','mode','matrix_change_states','raw_false_peaks','stable_false_peaks','median_distance']].copy()
 tr.columns=['参考','待检空场','处理','协方差超阈状态','初步方向峰','稳定方向峰','协方差距离中位数']
 peaks=pd.read_csv(R/'stable_G_peaks_E1.csv',dtype=str)
 peaks.insert(1,'站位',peaks.record.map({'0':'+1 m','3':'+1 m','12':'-1 m','15':'-1 m','21':'+2 m','24':'+2 m','30':'-2 m','33':'-2 m'}))
 peaks.columns=['记录','站位','G1','G2','G3','G4']
 os=pd.read_csv(R/'occupied_summary.csv').query('mode=="pi_hypothesis"')
 ot=os.pivot(index='record',columns='source',values='stable_state_peaks').reset_index()
 ot['跨 G 方向簇（E1/E2）']='0 / 0';ot.columns=['记录','E1：状态内稳定峰数','E2：状态内稳定峰数','跨 G 方向簇（E1/E2）']
 originals=pd.read_csv(R/'original_sources.csv');oc=originals[originals.role=='occupied'].copy()
 oc['batch']=oc.path.map(lambda s:Path(s).parents[2].name)
 provenance=oc[['index','label','batch','run','duration','sha256']].copy()
 provenance.sha256=provenance.sha256.str[:12]
 provenance.columns=['记录','标签','批次（仅来源标识）','轮次','设定秒数','原始 SHA256 前12位']
 text=r"""# 同频空间协方差背景模型：空场稳定性与四点候选路径验证

日期：2026-11-04  
性质：现有数据上的有界、回顾性验证。未修改论文、未上传 Overleaf、未采集新数据。  
方法：正则化同频空间协方差背景对照；不使用 Widar，不将本次方向评分称为二维 MUSIC。

## 1. 结论

**空场自身的方向输出可以得到一定稳定性，但空场参考的可迁移性仍有明显边界；四点有人记录存在可重复的空间变化，目前没有得到可验证的人体候选路径。**

- 四组空场分别用第 1 轮建模、第 2 轮定门槛、第 3 轮验证。24 个“空场 × 状态”检查中，初步方向候选和稳定方向候选均为 0。E4 的 G1/G3 协方差距离仍超阈，不能说所有统计量都完全稳定。
- 同时长配对中，冻结的链间 π 符号修正后，E1→E2（10 s→10 s）、E3→E4（15 s→15 s）第 3 轮均无稳定方向假候选，仍有部分协方差超阈。
- E1→E3 仍有 16 个状态内稳定假候选，并形成 1 个跨 G 子阵方向簇。用户确认两者连续、同场景、无重启/移动/重新接线；本结果不否定此事实，也不能用文件日期解释差异。“公共误差不变量 + π 符号”仍不能使这两个空场共享背景模型。
- 四点共检查 8 条既有代表记录，每条 6 状态，分别使用 E1/E2 作参考。状态内反复出现超阈方向变化，但两种参考下都是 **0/8 条记录通过跨 G 子阵相容规则**，同站位重复方向簇和双参考一致方向簇均为 0。
- 人体 AoA、AoA MAE、物理超额 ToF 和位置误差继续为 **NA**。这是受限验证结果，不是“原始数据完全不可用”的结论，也不是全数据人体检出率为零的证明。

## 2. 数据范围与划分

| 标识 | 目录标识 | 设定时长 | 用途 |
|---|---|---:|---|
| E1 | 20261003_130149_student01_6613 | 10 s | run1 建模、run2 门槛、run3 检查 |
| E2 | 20261003_130256_student01_11355 | 10 s | 同上，另作 E1 同时长迁移检查 |
| E3 | 20261018_151129_student01_118072 | 15 s | 同上，另作 E1 连续采集反例 |
| E4 | 20261018_151420_student01_127667 | 15 s | 同上，另作 E3 同时长迁移检查 |

文件名仅作来源标识，不作为真实日期或硬件初始化的证据。E1/E3 连续采集由用户确认；E1/E2、E3/E4 的硬件会话关系不因时长相同而自动成立。四点有人记录与已确认空场不属同一硬件会话。重复的 A 点有人批次未作为空场使用。

空场复用既有有效窗口与状态缓存，每状态有效包数 142–2010，保留短状态窗口。第 3 轮指完整轮次层面的划分及该轮全部既有有效窗口缓存，不是随机拆包留出，也不代表所有原始包均通过前置质量处理。

四点每点使用两个既有批次的 run1，并非所有三轮、所有批次已重算。每状态缓存有 768 个确定性时间抽样包，前 64 包用于有限符号假设，其余 704 包用于方向统计。

{provenance}

0/12/21/30 曾为开发记录，3/15/24/33 曾为回顾性评价记录；它们此前均被查看，不能称为独立新测试集。完整 39 条四点记录清单保留在旧目录，本次未因结果差删除任何记录，未在门槛失败时扩大计算。

G1=[1,2,3]，G2=[4,5,6]，G3=[7,8,9]，G4=[10,11,12]，C1=[1,8,6]，C2=[10,8,6]。阵元间距 28 mm，按实际坐标构造导向矢量。HT40 中心频率 5.23 GHz，实际 30 组子载波索引 −58,−54,…,58，间隔基准 312.5 kHz。未联合拼接 12 阵元。

## 3. 方法与公共误差不变性

令同一包、同一频率的三链 CSI 为 \(\mathbf x_{p,k}\)。以该包全部 3×30 项范数归一化，计算未中心化二阶矩：

\[
\widetilde{\mathbf x}_{p,k}
=\mathbf x_{p,k}/\|\mathbf X_p\|_F,\qquad
\mathbf R_k=\frac1N\sum_p
\widetilde{\mathbf x}_{p,k}\widetilde{\mathbf x}_{p,k}^{H}.
\]

如果误差仅为同包三链共同的乘性项
\(g_p e^{j(\phi_p-2\pi f_k\delta_p)}\)，公共相位和公共定时在同频外积中逐包抵消，共同幅度增益由归一化消除。因此无需拟合逐包 CPE/定时斜率或选取首包相位参考。

这不是物理 CFO 估计，不处理 CFO 引起的非公共失真或 ICI。不同链路固定响应仍存在：
\[
\mathbf R_k^{obs}=\mathbf D_k\mathbf R_k^{channel}\mathbf D_k^H.
\]
未知 \(\mathbf D_k\) 会改变空间相位。本方法舍弃跨频相位，**不能单独恢复绝对传播时延或给出 AoA–ToF 二维定位**。多径间相对结构可能保留在频率变化中，但本实现未据此估计时延。

本次没有直接复数相减空场 CSI，也没有套用此前未通过预测验证的连续相位拟合。

每状态、每频率由源 run1 建立 \(\mathbf B_k\)，固定正则化：
\[
\mathbf B_{\alpha,k}=0.9\mathbf B_k+
0.1\frac{\operatorname{tr}\mathbf B_k}{3}\mathbf I,\quad
\mathbf W_k=\mathbf B_{\alpha,k}^{-1/2}.
\]
数据与导向矢量一致变换：
\[
a_{k,m}(\theta)=e^{j2\pi f_k y_m\sin\theta/c},\quad
\mathbf b_k=\mathbf W_k\mathbf a_k,
\]
\[
S(\theta)=\frac1{30}\sum_k
\frac{\mathbf b_k^H\mathbf W_k\mathbf R_k\mathbf W_k^H\mathbf b_k}
{\mathbf b_k^H\mathbf W_k\mathbf B_k\mathbf W_k^H\mathbf b_k}.
\]
当 \(R=B\) 时评分为 1。这是背景参照的方向功率评分，不是标准 MUSIC 谱或已标定物理角度。

矩阵变化用全部同频矩阵堆叠后的距离：
\[
D(R,B)=\frac{\|R-B\|_F}{(\|R\|_F+\|B\|_F)/2}.
\]

## 4. 冻结门槛与候选规则

1. 源 run2 按时间切成 4 段，\(T_D\) 为各段距离最大值；\(T_S\) 为 1 与各段所有角度评分最大值中的较大者。低/高各 15 组频率另有源门槛。
2. 网格 −90°:3°:+90°。\(D>T_D\)，且 \(S>T_S\) 的 5 格邻域局部最大值为初步候选。边界峰保留。
3. 至少 3/4 时间段有 ±9° 内的超阈峰，两个频率半区也各有 ±9° 内的超阈峰，记为“状态内稳定候选”。
4. 至少 3 个 G 子阵稳定候选距簇中心不超过 9°，才形成跨 G 方向簇。C1/C2 仅诊断，不重复投票。
5. 同站位两记录方向簇相差不超过 9°，才进入重复候选；E1/E2 两种参考还须相容。9° 是固定三个网格步长，未按真值误差调参。

时间段与频率半区是相关诊断，不能当作独立实验。四段门槛对整轮统计可能较保守；有限空场 0 个假候选不是已标定虚警概率，也不证明真实人体检测灵敏度。正则化和门槛均不声称最优。

### π 符号及参考坐标

沿用此前冻结的 E1→E2、E3→E4 RF 链符号 [1,−1,−1]；E1→E3 的 [1,−1,+1] 作为失败假设保留。它们不等同于每个物理天线的连续初始相位标定。

有人分析的 E2 建模、门槛、验证数据先统一到 E1 的 RF 符号坐标；E2 自身 run3 仍无候选。每条有人记录的离散符号只由前 64 包、G1/G2、限定频点从四种假设选取，随后在所有状态及后 704 包冻结。没有用人体坐标/理论角度选峰，未逐点拟合连续相位。

有限符号选择仍可能吸收有人场景变化，因此保留 raw 对照；“更像空场”不等于校准正确。连续硬件相位未知，方向参数不具有绝对物理标定。

## 5. 空场结果

自身第 3 轮：

{within}

E4 的 G1/G3 距离门槛比分别约 1.20/1.14，但无正向方向候选。算法保留矩阵变化，不强制输出方向；功率降低/遮挡不一定产生正向峰。

跨记录第 3 轮：

{transfer}

raw 为无符号修正；fixed_pi 为源/目标 run1 冻结的离散符号。第 2 轮迁移也保留：E1→E2、E3→E4 修正后均 0 个稳定方向假候选，E1→E3 仍为 15 个。

**观察 → 解释 → 边界 → 决策：**

- 自身和两对同时长空场的方向假候选减少，支持同频统计与有限符号处理在这些对照中有用；不能推出跨记录绝对硬件标定已完成。E1/E2 可作后续受限诊断的两种参考，不能与所有空场自动合池。
- E1→E3 全 6 状态仍明显变化，说明尚有共同标量误差之外的差异。现有结果不能独立区分状态映射、链响应、环境/信道变化等原因，不否定用户确认的连续采集事实。
- E1→E3 两轮各有 1 个跨 G 方向簇，构成“无人也能出现一致方向”的反例。三子阵一致性不能单独证明人体路径。

![空场稳定性](<{emptyfig}>)

左侧 \(D/T_D>1\) 表示矩阵变化超过源开发轮次波动；右侧是满足时间/频率支持的状态内方向峰总数，16 个峰不代表 16 个目标。

## 6. 四点有人结果

两种参考下全部 48 个“记录 × 状态”均超过各自 \(T_D\)，但跨硬件会话差异本身也可能造成这一现象。下表为符号假设分支，raw 对照与初步峰保留在 CSV。

{occupied}

E1 下各三阵元 G 子阵的**状态内稳定方向参数**（度；逗号代表多个候选，none 为未通过稳定性检查）：

{peaks}

这些不是人体真实 AoA。−90°/+90° 是网格端点，不能当作可靠端射到达角；未为降低误差而删去。

![四点各子阵方向评分](<{occfig}>)

曲线为 \(10\log_{10}(S/T_S)\)，虚线 0 dB 为源空场门槛。超阈不等于通过全部规则。C 状态结果保存在 CSV。

- 多个站位反复表现为 G1 约 −90°、G2 约 0°、G4 约 +90°，G3 多为 −33°到 −24°及附加候选。这更像共同参考/固定子阵差异占据输出的表现，尚非随站位变化的同一人体路径；具体原因未独立识别。
- 同站位确有重复结构，但异站位也共享类似结构。协方差距离的探索性比较：同站位重复中位数 0.167，异站位 0.227，范围高度重叠。这不是分类准确率；112 个状态配对不等于 112 个独立实验。
- 两种参考都没有记录通过跨 G 相容规则，应停止人体 AoA/位置计算，而不是每点选最接近真值的峰。
- 未用人体真值调整门槛、选峰、模型阶数或偏置。点位名称仅用于冻结输出后的记录解释，未计算角度误差。

即使硬件不变，静止人体也会与背景形成相干交叉项。单频静态示意为：
\[
R_{on}-R_{off}
=h_hh_h^H+h_bh_h^H+h_hh_b^H.
\]
差值不必是纯“人体功率协方差”，也未必半正定。人体遮挡和跨会话硬件差异会继续混入。因此最大变化、稳定变化乃至一致方向，都不能自动归为人体反射。

## 7. 复核与复现

- 20 个原始 CSI 文件、120 个唯一状态缓存 SHA256 均核对一致；原始路径和轮次见 original_sources.csv。
- 45 条冻结模型/门槛/预测哈希一致。96 条空场状态检查、192 条有人状态检查（含两参考和 raw/符号两分支）的峰计数与明细一致。
- 包索引严格递增，有时间字段的缓存时间非递减；输入均有限值。
- 随机公共相位/定时扰动后协方差最大数值差约 \(3.9\times10^{-18}\)；一致变换数据/导向矢量后的评分差约 \(6.7\times10^{-16}\)。
- 理想标定正对照：固定种子、每轮 2048 包、每阵元 0 dB 信噪比、合成 21° 平面波。四个 G 状态均得到 21° 稳定峰并通过跨 G 筛选，合成空场 0 稳定峰。仅验证实现没有拒绝所有理想输入，不能替代真实灵敏度或硬件标定。
- Python 语法检查通过；PNG 已人工查看，无裁切或图例遮挡；矢量 PDF 同时保留。

复现顺序（在本目录，使用既有缓存而非重新解析全体 CSI）：

    python validate_empty.py
    python validate_occupied.py
    python plot_results.py
    python synthetic_control.py
    python verify_artifacts.py
    python build_report.py

依赖 NumPy、SciPy、pandas、Matplotlib、threadpoolctl；核心定义在 spatial_model.py。依赖既有 geometry.json、状态缓存、validate_sign.py，路径及哈希在 dependency_hashes.json。迁移路径时不能静默更换输入。

## 8. 下一步与停止条件

保留此分支作为**不依赖逐包公共相位/定时参考的空间变化诊断基线**。它改善了部分空场的方向稳定性，尚未解决固定连续链响应和人体归因。本次不扩大到全部记录、不相干拼接 12 阵元、不输出虚构精度或位置。

继续使用现有资料时，优先核查 E1/E3 的状态映射、链间幅相响应和实际包序号对应的时段。任何新连续链相位拟合都要另立方案和留出预测验证，不能把有人相位逐点“拉到”预期角度。

无法绕过的物理验证缺口仍为：可隔离直达径的已知源/链路响应标定，以及与有人记录同硬件会话的空场—有人—空场配对。前者验证物理导向模型，后者区分人体出现相关变化与背景漂移；它们不是本次已经完成的工作。

## 9. 交付文件

- 本报告：REPORT_同频空间协方差验证.md
- 冻结规则：PROTOCOL.md、CANONICAL_REFERENCE_NOTE.md
- 核心代码：spatial_model.py、validate_empty.py、validate_occupied.py
- 每状态结果：empty_validation.csv、occupied_validation.csv
- 所有峰及失败：empty_false_candidates.csv、occupied_candidates.csv
- 跨子阵及重复：empty_cross_G_clusters.csv、occupied_subarray_clusters.csv、occupied_repeat_candidates.csv、reference_robust_candidates.csv；空表保留表头，表示无通过候选。
- 三阵元便览：stable_G_peaks_E1.csv
- 图：empty_stability.png/pdf、occupied_subarray_scores.png/pdf
- 来源：original_sources.csv、training_sources.csv、validation_sources.csv、occupied_sources.csv
- 可恢复模型：model_*.npz、occupied_model_*.npz、spectra/*.npz
- 数值与模拟：numerical_checks.json、synthetic_end_to_end.json
- 审计：verification.json、dependency_hashes.json、artifact_hashes.json

旧结果、原始数据、论文和 Overleaf 均保持原状。
"""
 for name,value in {
  'provenance':table(provenance),'within':table(w),'transfer':table(tr),'occupied':table(ot),'peaks':table(peaks),
  'emptyfig':(R/'empty_stability.png').as_posix(),'occfig':(R/'occupied_subarray_scores.png').as_posix()
 }.items():text=text.replace('{'+name+'}',value)
 (R/'REPORT_同频空间协方差验证.md').write_text(text,encoding='utf8')
 from verify_artifacts import h
 paths=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.csv','.json','.md','.png','.pdf'] and p.name!='artifact_hashes.json']
 (R/'artifact_hashes.json').write_text(json.dumps({p.name:h(p) for p in sorted(paths)},indent=2,ensure_ascii=False),encoding='utf8')
 print('report_written',len(text),'characters; artifacts',len(paths))
if __name__=='__main__':main()
