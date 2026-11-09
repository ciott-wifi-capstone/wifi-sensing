from export_blind import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.bbox':'tight'})
def table(d):
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def main():
 df=pd.read_csv(R/'blind_peak_truth_comparison.csv');states=['G1','G2','G3','G4','C1','C2']
 gt=pd.read_csv(R/'theoretical_angles.csv');com=pd.read_csv(R/'compact_all_subarrays.csv')
 fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
 colors=['#2b6cb0','#c26c23','#278166'];markers=['o','s','^']
 for ax,first,label in zip(axes.flat,[0,12,21,30],['+1 m','-1 m','+2 m','-2 m']):
  q=df[(df.reference=='E1')&(df.batch_first==first)]
  truth=q.drop_duplicates('state').set_index('state').loc[states,'theory_subarray_deg']
  ax.plot(range(6),truth,'kD--',ms=5,lw=1,label='Subarray-centre theory')
  for r in range(3):
   rr=q[q['index']==first+r].set_index('state').loc[states]
   for k,(s,row) in enumerate(rr.iterrows()):
    ax.scatter(k+(r-1)*.15,row.spectrum_top1_deg,marker=markers[r],s=48,
      edgecolors=colors[r],facecolors=colors[r] if row.spectrum_top1_is_stable else 'none',zorder=3)
  ax.set_xticks(range(6),states);ax.set_ylim(-100,100);ax.grid(axis='y',alpha=.2);ax.axhline(0,color='gray',lw=.6);ax.set_title(label+(' (run 3 truncated)' if first==0 else ''));ax.set_ylabel('Direction parameter / theory (deg)')
 from matplotlib.lines import Line2D
 handles=[Line2D([0],[0],color='black',marker='D',linestyle='--',label='Subarray-centre theory')]
 handles += [Line2D([0],[0],color=c,marker=m,linestyle='none',label=f'Run {i+1}') for i,(c,m) in enumerate(zip(colors,markers))]
 fig.legend(handles=handles,ncol=4,loc='lower center')
 fig.suptitle('Blind strongest peaks vs geometric bearings | E1 reference\nFilled: stable candidate; hollow: fails stability. No physical AoA calibration.')
 fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(R/'blind_peaks_vs_theory.png',dpi=180);fig.savefig(R/'blind_peaks_vs_theory.pdf');plt.close(fig)
 primary=com[com.reference=='E1'].copy()
 primary.theory_deg=primary.theory_deg.map(lambda x:f'{x:+.2f}')
 primary=primary[['point','state','theory_deg','raw_peak_R1_R2_R3','raw_absdiff_R1_R2_R3','raw_stability_R1_R2_R3']]
 primary.columns=['点位','子阵','理论角度','最强盲峰 R1/R2/R3','绝对偏差 R1/R2/R3','稳定检查 R1/R2/R3']
 stable=com[com.reference=='E1'][['point','state','theory_deg','stable_peak_R1_R2_R3','stable_absdiff_R1_R2_R3']].copy()
 stable.theory_deg=stable.theory_deg.map(lambda x:f'{x:+.2f}')
 stable.columns=['点位','子阵','理论角度','稳定盲选峰 R1/R2/R3','绝对偏差 R1/R2/R3']
 e2=com[com.reference=='E2'][['point','state','theory_deg','raw_peak_R1_R2_R3','raw_absdiff_R1_R2_R3']].copy()
 e2.theory_deg=e2.theory_deg.map(lambda x:f'{x:+.2f}')
 e2.columns=['点位','子阵','理论角度','E2最强盲峰 R1/R2/R3','绝对偏差 R1/R2/R3']
 allp=pd.read_csv(R/'all_candidates_truth_annotations.csv')
 evidence=[]
 for idx in [12,13,14]:
  for s,angle in [('C1',18.),('C2',24.)]:
   q=allp[(allp.reference=='E1')&(allp['index']==idx)&(allp.state==s)]
   row=q[np.isclose(q.angle_parameter_deg,angle)].iloc[0]
   rank=int((q.score>row.score).sum()+1)
   evidence.append(dict(record=idx,state=s,frozen_candidate_deg=angle,score_rank=rank,ratio_to_strongest=row.score/q.score.max(),abs_difference_deg=row.absolute_difference_deg,stable=row.stable))
 ev=pd.DataFrame(evidence);ev.to_csv(R/'minus1_secondary_candidate_evidence.csv',index=False)
 text=r"""# 四点各子阵盲选峰、理论角度与偏差

2026-11-04。复用最近的同批次三轮结果，不重跑大规模实验。方法为同频空间协方差背景评分，不是新算的二维 MUSIC。

## 1. 选择规则与证据边界

- 最强盲峰：直接取整个评分谱的最大值，完全不看真值；同分取较小角度。
- 稳定盲选峰：只在此前通过时间/频率稳定性检查的候选中取最高评分；没有候选则 NA。
- 先导出144行盲选预测并保存 SHA256，再由独立脚本读取人体坐标计算偏差。不是“从真值附近挑峰”。
- 主表固定使用 E1；E2 敏感性结果同时给出，不能按哪个背景误差小就选择哪个背景。
- 12轮×6状态×2参考。数据此前已被查看，故是回顾性诊断，不是独立盲测性能。
- 以下角度仍是未完成物理校准的方向参数。数值偏差不是已验证的人体 AoA 精度，也不等于定位成功。

## 2. 理论角度定义

全局 TX=(0,0)，RX=(3.65,0)，目标=(1.64,Y)。RX局部坐标为 \((x_\ell,y_\ell)=(2.01,-Y)\)，0° 指向 TX，正角朝局部 +y 侧。四点的阵列中心理论角为：

{center}

每个三阵元子阵还要考虑自身中心：
\[
\bar y_s=\frac{1}{3}\sum_{m\in s}(6.5-m)0.028,\qquad
\theta_s^{GT}=\operatorname{atan2}(-Y-\bar y_s,2.01).
\]
G1/G2/G3/G4 中心分别为 +0.126/+0.042/−0.042/−0.126 m；C1/C2 中心分别为 +0.042/−0.042 m。偏差为 \(\Delta\theta=\hat\theta-\theta_s^{GT}\)，主表列绝对值，CSV同时保留有符号偏差与阵列中心理论角。

这是站位中心点的几何方向；人体散射中心可能与站位参考中心不同。C1/C2 为非等距三阵元，中心方位不代表其谱能唯一反演方向。未用符号翻转、180°折返或人为偏置减少误差。

## 3. E1参考：最强盲峰（不是人体判定）

单位：度；三项依次为 R1/R2/R3。+1 m 的 R3 只记录约3.15秒，其他轮次完整；截短结果只作诊断。yes/no 指最强峰是否也是已通过稳定性检查的候选。

{primary}

其中 −2 m/G1 的 R1 实际偏差约8.005°，若只显示1位小数会变成8.0°，不能据此认定≤8°；下表/CSV保留足够精度。没有计算或宣称人体总体MAE。

![盲峰与理论角度](<{fig}>)

实心点为通过稳定性检查，空心点为未通过。黑色标记是各子阵几何理论角，不参与计算谱峰。

## 4. E1参考：通过稳定性门槛后再盲选

{stable}

这张表解释了为何“原谱某个最大峰很接近真值”不等于最终候选通过。例如 −2 m/G1 最强峰在三轮为+51/+48/+42°，几何偏差约8.01/5.01/0.99°，但三轮均未通过原有频率/时间稳定性组合门槛；稳定候选是另一分支或为空，不能换用更接近真值的峰来报告成功。

+1 m/G3 的原谱最强峰为−33/−30/−30°，几何偏差约7.52/4.52/4.52°；只有R1通过稳定性门槛。这个局部现象也不足以证明三轮人体方向恢复。

## 5. 现有数据中有哪些值得继续核查的候选

在 −1 m 记录的完整、已经冻结的候选集合中，C1 的 +18°、C2 的 +24°均在三轮稳定出现，且 E1/E2 两参考下都存在。几何对照分别为：
- C1 理论 +25.483°，候选 +18°，偏差7.483°；
- C2 理论 +27.403°，候选 +24°，偏差3.403°。

这是对**全部已冻结候选的事后标注**，不是估计器使用真值选出的结果。这两条分支只能称为“待验证候选”，不能直接改名为人体反射。

{evidence}

C2 的 −39°与+24°分支都反复存在，R1中+24°的评分约为最高峰的99.6%，之后发生排名交换。因此只跟踪全局最大值会出现63°跳变；保留多候选能保留实际稳定的分支，但并不自动完成人体归因。

阻止确认的证据缺口：
1. 不足3个互不重叠的 G 子阵提供相容的稳定方向；现有跨 G 方向簇仍为0。
2. C1/C2共享物理天线6、8，不是两个完全独立观测。
3. E1→E3的纯空场对照在 C2 也有+30°附近稳定峰，与+24°相近；不能仅凭接近真值和重复出现排除背景/旁瓣。该空场对照仍有跨初始化问题，因此也不能反过来证明+24°绝不可能含人体分量。
4. 现有有人记录没有同一驱动初始化段的配对空场，固定链/支路响应仍未物理标定。

所有原始与稳定候选、对应理论偏差均在 all_candidates_truth_annotations.csv。within8_evaluation_only 只用于事后核查；未加入选峰规则，也未用最近真值峰替换盲选结果。

## 6. 如何找到对应站位的人体候选

下一步应验证“某条候选为何属于人体”，而不是用真值直接指定某个峰。

1. **先恢复可信的角度参考。** 仅在可独立识别同记录直达径/已知源时估计固定链响应，同时补偿真实传播相位。不能用人体站位逐点拟合初始相位。现有 E1/E3 统一相位、斜率与增益修正均未通过，不能直接移植到有人批次。
2. **保留所有合格局部峰，做跨轮关联。** 以角度连续性、频率支持、时间支持和评分稳定性连接分支，不强制每轮取最大值。分支可以保持存在而强弱排名交换，−1 m/C2就是具体例子。关联容差必须用开发数据/空场确定后冻结。
3. **按真实阵元坐标做跨子阵检验。** 用 G1–G4 的互不重叠子阵形成共识；C1/C2检验稀疏阵歧义和共享天线关系，不能重复投票。若近场差异显著，采用实际子阵中心/球面波预测，而不是把各峰强行视作完全相同角度。几何搜索范围须来自房间/场地定义，不能只放四个真值点作为候选位置。
4. **用空场零假设验证人体关联。** 候选变化需超过相同参考条件下空场自身波动，最好与人体出现/离开相关；区分人体反射、遮挡导致的背景变化及硬件变化。当前无配对记录时，可以保留候选，不能证明归因。
5. **冻结输出后才与真值匹配。** 检验每站位每轮是否有唯一、可重复且被多子阵支持的候选，并报告漏检/多解。一个峰落入真值±8°只是几何接近，不是人体身份验证。

本分支是同频空间统计，未提供经验证的物理 ToF；即使方向候选通过，也不能据此补出二维位置。当前不需要为了“找到真值”放宽门槛或删除困难轮次。

## 7. E2参考敏感性

{e2}

同样的站位与真值、不同冻结背景。完整门槛状态、稳定盲选和有符号偏差见 CSV，不以 E1/E2 中较小偏差作为最终结果。

## 8. 文件与复现

- blind_predictions.csv / blind_predictions_frozen.json：不读取人体坐标的预测及哈希。
- blind_peak_truth_comparison.csv：144行，两参考、12轮、6状态；含原最强峰、稳定盲选峰、理论角度、有符号/绝对偏差、完整性。
- compact_all_subarrays.csv：所有子阵的三轮便览。
- all_candidates_truth_annotations.csv：所有既有候选的事后几何标注。
- minus1_secondary_candidate_evidence.csv：−1 m 的两个待验证分支及评分排名。
- theoretical_angles.csv：各子阵中心和几何方向。
- blind_peaks_vs_theory.png/pdf：盲峰对照图。
- export_blind.py → evaluate_truth.py → make_report.py：依次导出、冻结、评价及生成报告。依赖前一轮冻结谱，不重新修改相位或计算大规模实验。
"""
 centers=gt.drop_duplicates('batch_first')[['point','theory_array_center_deg']].copy()
 centers.theory_array_center_deg=centers.theory_array_center_deg.map(lambda x:f'{x:+.3f}°')
 centers.columns=['站位','RX阵列中心理论角']
 evt=ev.copy()
 evt.ratio_to_strongest=evt.ratio_to_strongest.map(lambda x:f'{100*x:.2f}%')
 evt.abs_difference_deg=evt.abs_difference_deg.map(lambda x:f'{x:.3f}')
 evt.columns=['记录','子阵','已有候选/度','谱内排名','相对最大评分','几何偏差/度','稳定门槛通过']
 for k,v in {'center':table(centers),'primary':table(primary),'stable':table(stable),'e2':table(e2),'evidence':table(evt),'fig':(R/'blind_peaks_vs_theory.png').as_posix()}.items():text=text.replace('{'+k+'}',v)
 (R/'REPORT_四点盲选峰与真值对照.md').write_text(text,encoding='utf8')
 print('report characters',len(text))
 print(ev.to_string(index=False))
if __name__=='__main__':main()
