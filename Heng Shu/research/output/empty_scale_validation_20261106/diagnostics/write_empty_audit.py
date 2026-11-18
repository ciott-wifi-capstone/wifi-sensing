from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent
d=pd.read_csv(R/'empty_scalar_decomposition.csv')
q=pd.read_csv(R/'empty_model_predictions.csv')
t=pd.read_csv(R/'empty_tone_eigenvalues.csv')
q=q[(q.window=='R3_held')&(q.frequency_split=='held_odd')]
six=q[q.state!='G2_control'];g2=q[q.state=='G2_control']
names=['unchanged','projected_scalar_per_tone','projected_scalar_shared_frequency','affine_aR1_bI_shared_frequency','full_empty_anchor_per_tone']
labels=['不修正','逐频标量投影底','共享频率标量投影底','共同增益＋白底 aR1+bI','完整R3开发块协方差锚点']
def line(v):return '| '+' | '.join(str(x) for x in v)+' |'
def fmt(x):return f'{x:.3f}'
out=['# 空场跨轮背景与噪声尺度：独立诊断','',
'本审计只处理最新A–E五批的R1、R3空场，完全不计算R2有人协方差、不读取人体坐标、不估计人体角度。A–E在本文中只代表采集批次。原始缓存哈希与旧冻结清单一致。','',
'## 划分和尺度','',
'每批每状态均使用缓存中的24个周期/批。R1前8建立参考，后16留出；R3前8作开发，后16作时间留出。六态每态每周期6包，G2控制每批36包，因此完整每轮均864个不同包，R1/R3开发各288包、相应留出各576包。实际运行只选npz中索引0和2的run切片，未使用索引1。','',
'并列两种表示：（1）原来每包共同正实数归一化，三链30频点总能量固定90；（2）每状态由R1前8的平均数字总能量计算一次共同常数，使该均值为90，之后所有R1/R3包沿用这个常数。这是原始数字CSI的敏感性对照，不是已校准RSSI或物理热噪声。','',
'## 检验模型与留出含义','',
'固定R1主特征向量h和B=I−hhᴴ，D=B(R−R1)B。空场投影变化的最优标量分量为β=tr(D)/2，剩余D−βB衡量各向异性。该逐窗口最优分解是oracle诊断，不是可直接套到人体数据上的校正。','',
'比较五个预定小模型：不修正；R3开发逐频βB；R3开发偶数频点拟合一个共同β；偶数频点拟合R3≈aR1+bI两个实系数；完整R3开发矩阵作为后空场锚点。系数在R3后16周期保持冻结。aR1+bI为无约束诊断拟合，可能非PSD，不能自动作为合法协方差使用。','',
'**频率留出区别：** 共享β和aR1+bI使用偶数频点拟合，奇数频点是真正的目标频率留出；逐频β与完整矩阵锚点使用过奇数频点的R3开发数据，因此只提供时间留出，不能称为频率外推。CSV的frequency_role已显式标注。','',
'## 全五批的时间留出预测','',
'下表先对每条状态记录取奇数频点中位，再对30条六态记录或5条G2控制记录取中位。残余比=修正后投影残余范数/未修正变化范数；相对R1held倍数=残余范数/R1留出投影变化范数。逐频范数为Frobenius范数。','',
line(['表示','模型','六态残余比','六态/R1held','G2残余比','G2/R1held']),line(['---']*6)]
for rep in ['unit_packet','fixed_R1_scale_raw']:
 for name,label in zip(names,labels):
  a=six[(six.representation==rep)&(six.model==name)];b=g2[(g2.representation==rep)&(g2.model==name)]
  out.append(line([rep,label,fmt(a.projected_residual_ratio.median()),fmt(a.projected_residual_over_R1held.median()),fmt(b.projected_residual_ratio.median()),fmt(b.projected_residual_over_R1held.median())]))
out += ['', '改成固定R1尺度raw并没有使标量模型解释掉大部分空场变化，说明问题不能仅归咎于每包归一化。上述比较是诊断，没有据此搜索正则参数或放大验收阈值。','',
'## B、C与各批的结构','',line(['批次','oracle各向异性残余/原变化','次/主特征值绝对比','逐频β预测残余比','完整锚点预测残余比']),line(['---']*5)]
for pt in 'ABCDE':
 a=d[(d.point==pt)&(d.state!='G2_control')&(d.representation=='unit_packet')&(d.window=='R3_held')]
 b=t[(t.point==pt)&(t.state!='G2_control')&(t.representation=='unit_packet')]
 c=six[(six.point==pt)&(six.representation=='unit_packet')]
 out.append(line([pt,fmt(a.oracle_anisotropic_residual_fraction.median()),fmt((abs(b.small_eigenvalue)/abs(b.large_eigenvalue)).median()),fmt(c[c.model=='projected_scalar_per_tone'].projected_residual_ratio.median()),fmt(c[c.model=='full_empty_anchor_per_tone'].projected_residual_ratio.median())]))
out += ['',
'B：逐频标量只能把投影变化范数降到约0.724，仍是R1held波动的8.36倍。完整R3开发矩阵锚点降至约0.530，仍是5.02倍。B并非只缺一个共同噪声方差参数；开发后仍有时间变化。','',
'C：逐频标量残余约0.691，完整锚点约0.721，后者没有更好。这说明记录内参考变化不能简单用一个早期完整后空场矩阵消除。G2控制的aR1+bI在时间/频率留出时投影残余比约1.053，甚至略高于不修正。','',
'单位包表示下，无约束aR1+bI在A/G3、B/G3、E/G1、E/G3产生负特征值预测，不能当作有效协方差校正。该问题应保留为模型不相容的诊断；不能根据有人输出好坏再选择裁剪或约束。','',
'## 为什么“50%标量能量”不是“50%噪声”','',
'二维投影空间中，即使变化只有一个定向模态D=λuuᴴ（||u||=1），tr(D)/2=λ/2，其标量部分(λ/2)I天然占D平方Frobenius范数的50%，剩余范数比为1/√2≈0.707。实际B等批与这一高度各向异性形态相近。因此oracle标量占比约0.5不能证明噪声占了一半。','',
'直接对每个人体目标矩阵扣掉tr(D)/2或最小特征值，可能误消真实定向变化，或代数上强制获得rank1。当前空场证据不支持这样做。共用包相位已由同频外积消除；这里仍混合链间响应、方向性信道改变、噪声/归一化效应。','',
'## 可支持的结论','',
'1. 存在明显的方向性/各向异性跨轮空场变化，简单标量底或共同增益＋白底不足以解释。','2. 新的完整后空场锚点也有显著时间外推误差，不能将同一R3既拟合又当未见检验。','3. 本审计不能唯一分离真实硬件响应漂移与环境变化；不证明所有数据不可用。','4. 若空场模型未通过主流程预先固定的稳定性门槛，应保持PCD受限/不输出，并只把有人结果作为失败诊断，不能为得到更多有效频点而放宽门槛。','',
'文件：audit_empty_scales.py；empty_scalar_decomposition.csv；empty_model_predictions.csv；empty_tone_eigenvalues.csv；empty_source_audit.csv；audit_scope.json。']
(R/'空场尺度独立审计.md').write_text('\n'.join(out)+'\n',encoding='utf8')
print('Empty-only report written.')
