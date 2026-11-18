from branch_analysis import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

POINTS={0:(1.64,1.),12:(1.64,-1.),21:(1.64,2.),30:(1.64,-2.)}
LABELS={0:'+1 m',12:'-1 m',21:'+2 m',30:'-2 m'}

def table(df,cols=None):
    if cols is not None:df=df[cols]
    vals=df.fillna('NA').astype(str)
    return '| '+' | '.join(vals.columns)+' |\n| '+' | '.join(['---']*len(vals.columns))+' |\n'+'\n'.join('| '+' | '.join(v)+' |' for v in vals.values)

def main():
    verify_freeze(R/'input_freeze.json');verify_freeze(R/'development_freeze.json');verify_freeze(R/'predictions_frozen.json')
    v=pd.read_csv(R/'branch_evidence.csv');dec=pd.read_csv(R/'release_decisions.csv')
    empt=pd.read_csv(R/'empty_control_scores.csv');null=pd.read_csv(R/'empty_repeated_branches.csv')
    rows=[]
    for p in v.to_dict('records'):
        x,y=POINTS[p['batch_first']]
        yc=float(np.mean(np.array(sm.GEO['array_y_local_m'])[np.array(sm.MAP[p['state']])-1]))
        theta=float(np.degrees(np.arctan2(-y-yc,3.65-x)))
        p.update(point=f'({x:.2f},{y:+.0f})',theory_deg=theta)
        for j in [1,2,3]:p[f'abs_difference_r{j}']=abs(p[f'angle_r{j}']-theta)
        p['center_absolute_difference_deg']=abs(p['center_dev_deg']-theta)
        p['within8_after_freeze_only']=p['center_absolute_difference_deg']<=8
        rows.append(p)
    ev=write(rows,'all_branch_truth_evaluation.csv')
    primary=ev[ev.reference=='E1']
    summary=[]
    for first in GROUPS:
        q=primary[primary.batch_first==first]
        release=dec[(dec.reference=='E1')&(dec.batch_first==first)].iloc[0]
        summary.append(dict(point=LABELS[first],development_branches=len(q),r3_supported=int(q.r3_supported.sum()),
            complete_repeat=int(q.complete_repeat.sum()),complete_nonboundary=int((q.complete_repeat & ~q.boundary).sum()),
            complete_nonboundary_G_states=release.repeat_nonboundary_G_states,three_G_gate=release.three_G_gate,
            human_output='UNCONFIRMED',physical_AoA_MAE='NA',position_error='NA'))
    summ=write(summary,'point_summary.csv')
    stat=empt.groupby(['reference','target','run','mode','role']).agg(states=('state','size'),stable_false_peaks=('stable_candidates','sum'),raw_false_peaks=('raw_candidates','sum')).reset_index()
    stat.to_csv(R/'empty_control_summary.csv',index=False)

    # Display all developed branches, not only branches close to truth.
    colors=dict(zip(ST,plt.get_cmap('tab10').colors[:6]))
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.5),sharex=True,sharey=True)
    for ax,first in zip(axes.ravel(),GROUPS):
        q=primary[primary.batch_first==first]
        for p in q.itertuples():
            ang=np.array([p.angle_r1,p.angle_r2,p.angle_r3],float)
            col=colors[p.state]
            ax.plot([1,2,3],ang,color=col,lw=1.4,alpha=.7,ls=':' if p.boundary else '-')
            ax.scatter([1,2,3],ang,facecolors='none' if not p.complete_repeat else col,edgecolors=col,s=27,zorder=3)
        ax.axhspan(87,95,color='.92');ax.axhspan(-95,-87,color='.92')
        ax.set_title(f'{LABELS[first]}: {int(q.r3_supported.sum())}/{len(q)} branches supported in R3'+ (' (partial R3)' if first==0 else ''),fontsize=11)
        ax.set_ylim(-95,95);ax.set_xlim(.85,3.15);ax.set_xticks([1,2,3]);ax.set_yticks([-90,-60,-30,0,30,60,90]);ax.grid(alpha=.2)
        ax.set_xlabel('Capture round');ax.set_ylabel('Uncalibrated angle parameter (deg)')
    fig.legend([Line2D([0],[0],color=colors[s],lw=2) for s in ST],ST,ncol=6,loc='lower center',bbox_to_anchor=(.5,.02))
    fig.suptitle('All E1-reference multi-peak branches: development R1/R2, retrospective check R3',fontsize=13)
    fig.text(.5,.007,'Open markers: incomplete or failed complete-repeat check. Grey bands: boundary peaks. No human-path attribution.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(R/'all_branch_tracks.png',dpi=180);fig.savefig(R/'all_branch_tracks.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),sharey=True)
    es=np.load(R/'empty_control_spectra.npz')
    for ax,state in zip(axes,['C1','C2']):
        for j,idx in enumerate([12,13,14]):
            z=np.load(A/'occupied_spectra'/f'E1_{idx}_{state}.npz');score=z['scores'][0]
            ax.plot(z['angles'],score/score.max(),label=f'Occupied -1 m R{j+1}',lw=1.6)
        for run in [2,3]:
            score=es[f'E1_E3_{run}_fixed_pi_{state}'][0]
            ax.plot(es['angles'],score/score.max(),ls='--',lw=1.2,label=f'Empty E3 R{run}')
        theta=primary[(primary.batch_first==12)&(primary.state==state)].theory_deg.iloc[0]
        ax.axvline(theta,color='black',ls=':',lw=1.5,label='Theory (evaluation only)')
        ax.set_title(state);ax.set_xlabel('Uncalibrated angle parameter (deg)');ax.set_xlim(-90,90);ax.set_ylim(0,1.05);ax.grid(alpha=.2)
    axes[0].set_ylabel('Within-curve normalized background score')
    axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('Repeated occupied branches and empty-control alternatives',fontsize=13)
    fig.text(.5,.01,'Each curve normalized separately: compare angular structure, not absolute signal strength or human likelihood.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/'occupied_empty_branch_comparison.png',dpi=180);fig.savefig(R/'occupied_empty_branch_comparison.pdf');plt.close(fig)

    full=[]
    for p in primary.itertuples():
        full.append({'点位':LABELS[p.batch_first],'状态':p.state,'三轮角度': '/'.join('NA' if not np.isfinite(a) else f'{a:g}' for a in [p.angle_r1,p.angle_r2,p.angle_r3]),
                     '完整三轮重复':p.complete_repeat,'边界':p.boundary,'E2支持':p.alternative_reference_support,
                     '空场邻近反例':p.empty_nearby_examples if isinstance(p.empty_nearby_examples,str) else '无',
                     '理论角度':f'{p.theory_deg:.2f}','开发中心偏差':f'{p.center_absolute_difference_deg:.2f}'})
    compact=write(full,'all_E1_branches_readable.csv')
    nulltab=stat[(stat.reference=='E1')&(stat['mode']=='fixed_pi')][['target','run','role','stable_false_peaks','raw_false_peaks']]
    examples=primary[(primary.batch_first==12)&(((primary.state=='C1')&(primary.center_dev_deg==18))|((primary.state=='C2')&(primary.center_dev_deg==24))|((primary.state=='G3')&(primary.center_dev_deg==36)))]
    exrows=[]
    for p in examples.itertuples():
        exrows.append({'子阵':p.state,'三轮角度':f'{p.angle_r1:g}/{p.angle_r2:g}/{p.angle_r3:g}','理论':f'{p.theory_deg:.2f}',
                      '偏差':f'{p.center_absolute_difference_deg:.2f}','E2支持':p.alternative_reference_support,'无人邻近':p.empty_nearby_examples if isinstance(p.empty_nearby_examples,str) else '无'})
    close=primary[(primary.complete_repeat)&(~primary.boundary)&primary.within8_after_freeze_only]
    # Extra LP bound diagnostic; never a range estimate.
    geo=pd.read_csv(R/'geometry_combinations.csv')
    intervals=[]
    for z in geo[geo.feasible].itertuples():
        chosen=v[v.validation_id.isin(z.branches.split(';'))].to_dict('records');aa=[];bb=[]
        for p in chosen:
            yc=float(np.mean(np.array(sm.GEO['array_y_local_m'])[np.array(sm.MAP[p['state']])-1]))
            lo=max(-90.,p['lower_deg']);hi=min(90.,p['upper_deg'])
            if lo>-90:aa.append([np.tan(np.deg2rad(lo)),-1]);bb.append(-yc)
            if hi<90:aa.append([-np.tan(np.deg2rad(hi)),1]);bb.append(yc)
        mn=linprog([1,0],A_ub=aa,b_ub=bb,bounds=[(1e-6,None),(None,None)],method='highs')
        mx=linprog([-1,0],A_ub=aa,b_ub=bb,bounds=[(1e-6,None),(None,None)],method='highs')
        intervals.append(dict(reference=z.reference,batch_first=z.batch_first,states=z.states,min_local_x=mn.x[0] if mn.success else np.nan,
                              max_local_x=mx.x[0] if mx.success else np.inf,not_estimated_range=True))
    intervaldf=write(intervals,'geometry_interval_diagnostics.csv')
    alias=pd.read_csv(R/'branch_steering_similarity.csv')
    aliaspair=alias[(alias.reference=='E1')&(alias.batch_first==12)&(alias.state=='C2')].iloc[0]
    report=f'''# 四点多峰分支验证报告

2026-11-05。只复用已有数据及缓存，保留旧结果，未修改论文或 Overleaf。所有选择先冻结，再计算人体真值偏差；由于此前已经查看这些记录，本次属于回顾性验证，不是新独立测试。

## 结论

多候选关联确实找回了被最大峰策略丢掉的持续分支，但没有找到足以确认人体的独立证据。E1 主背景产生 {len(primary)} 条开发分支，{int(primary.r3_supported.sum())} 条获得第3轮支持，{int(primary.complete_repeat.sum())} 条有完整三轮支持，其中 {int((primary.complete_repeat & ~primary.boundary).sum())} 条不在谱边界。分支数不是独立人体数，也不是定位成功次数。

四点、两种背景均未通过至少三个独立 G 子阵的几何支持。人体 AoA MAE、位置误差继续为 NA；没有把未确认的峰强制输出为人体方向。

## 实现和冻结

1. 保留原有局部峰及其时间/频段稳定门槛，E1 为主背景、E2 为敏感性背景，不重新拟合阈值或相位。
2. 第1/2轮稳定峰相差不超过9度建立分支，平均角度冻结为开发中心；第3轮稳定峰须在中心9度以内。全部合法关联保留，不按真值选择，也不只保留最强峰。
3. 独立 G1–G4 按物理阵列中心构造角度楔形区域，线性规划检验共同前半平面是否存在；C1/C2 只作辅助，不能重复计票。没有将单组角度或数学可行点当作距离估计。
4. 模型、代码、输入和开发分支先哈希冻结；分支、空场检验与几何输出再次冻结，然后独立评价真值。全部候选（包括不稳定、未关联和边界峰）写入 candidate_ledger.csv。

空场新评分只处理192个缓存状态-模式比较，未重解码原始CSI。目标缓存去掉首64包，与有人评分一致，故部分假候选数与旧报告略不同；旧结果未覆盖。原始模式与有限 pi 模式均保留。E4 的符号组合是假设性扩展，E3 反例不依赖这一扩展。

## 各点结果

{table(summ)}

+1 m 第3轮只有约3.15秒，持续分支可以列出，但不计完整重复。完整非边界分支中，事后中心偏差<=8度的 E1 分支数为 {len(close)}；该计数只是描述性真值对照，不能作为人体检出率。未通过稳定性的强峰没有因为接近真值而重新纳入。

## -1 m 的候选及其反证

{table(pd.DataFrame(exrows))}

C1的+18度与C2的+24度均可跨三轮、跨E1/E2背景保留。C2中-39度和+24度分支都存在，最强峰跳变来自评分排序交换，不能仅凭最大峰变化断言人体运动。

但 C1/C2 共享天线6、8，不是两个独立阵列证据。C2的+24度附近分支在四点都出现（+1 m 的第三轮不完整），C1的+18/+21度分支也跨站位存在；它们缺少站位特异性。E3无人对照中的C1约+9度、C2约+30度分支在两轮复现，分别在上述候选9度、6度内。9度是事先固定的关联容差，不是经过验证的物理角度置信区间。这些是背景/硬件/稀疏阵列歧义尚未排除的反例，不能直接证明是同一物理路径，也不能直接证明有人分支不是人体。

新增追踪还保留了G3的+36度分支，其事后偏差约8.60度，但E2没有对应的完整稳定分支。不可因为它比最强的负角度峰接近真值就给它额外权重。

## C2的固有方向歧义

C2=[10,8,6]的阵元位置为-0.098、-0.042、+0.014 m，是56 mm等距子阵，不是不等距子阵；C1才是不等距的。56 mm在5.23 GHz约为0.977个波长。

对冻结的-39度和+24度两个开发分支，逐频率理想导向矢量归一化内积的中位数为 **{aliaspair.coherence_median:.6f}**，30个频率组的范围为 {aliaspair.coherence_min:.6f}–{aliaspair.coherence_max:.6f}。它们对C2形成几乎相同的空间响应，是稀疏间距产生的强方向歧义。此数值是模型相似度，不是实测CSI相关系数或人体概率；也不独立证明两峰来自同一条真实路径。

所以，仅凭C2最大峰、重复性或接近真值不能在这两个分支之间作出可靠选择。固定初始相位校准可以处理硬件偏置，却不能消除阵列几何本身的这一歧义；需要独立G子阵或其他可信观测来消歧。该诊断不改变已经冻结的候选和阈值，所有同状态开发分支对都保存在 branch_steering_similarity.csv。

## 几何检验

对每个分支，三轮角度容差交集为[l_s,u_s]，子阵中心为 y_s。局部候选点满足 y_s+x tan(l_s)<=y<=y_s+x tan(u_s)，x>0。用全部前半平面验证，不以四个测试点建立搜索网格。

E1下只有-1 m的G2零度与G3正36度形成一个两G的数学相交区域：

{table(intervaldf)}

这是未校准角度楔形的可行区间，不是估计距离。它只存在于阵列附近，x>=0.25 m即不再相交；没有第三个独立G支持。0.25/0.5/1 m是数值敏感性下界，不是测得的房间或人体先验。即使完整半平面里出现可行点，也不能将其作为人体位置。C1/C2的附加相容性不弥补独立G证据缺失。

## 空场反例

E1背景、冻结符号假设下的假候选计数如下：

{table(nulltab)}

空场自己的run2属于模型开发数据；对应run3不参与该背景门槛拟合。E1/E2间相对稳定与E1/E3间明显变化并存。因此“空场能否稳定”取决于使用哪个参考，不可把E1/E3的差异直接当成人体变化。E3和E4仅有两轮用于这里的分支对照，不声称其证据强度等于三轮有人重复。

样本量限制：E1/G2 第2、3轮在去掉首64包后分别只有78、79包，低于有人缓存的样本量。结果保留在总表中，额外标记在 empty_sample_quality.csv；这两条记录的零假候选不能独立证明背景稳定。验证时发现的样本量假设及处理说明见 VERIFICATION_NOTES.md，未调整峰值或关联门槛。

原始模式的假候选同样记录在 empty_control_summary.csv；有限符号变化降低部分假候选，却未完成连续硬件响应校准。这里的计数不是独立、校准后的误报概率。无邻近空场峰也不代表人体归因成功。

## 仍缺的证据与停止条件

- 未有独立验证的每条硬件支路空间相位响应；当前角度仅为未校准谱参数。
- 四点有人与空场没有配对的同初始化背景记录，也没有可靠的同记录人体出现/离开对照。
- 无分支通过三个不重叠G子阵的完整重复和几何支持。
- 同频评分没有恢复物理ToF，不能据此生成双基地距离或二维位置。

因此停止位置误差计算。当前已有资料能支持“持续多峰分支及其失败机制”，不能唯一决定哪个分支是人体。要跨过这一证据缺口，至少需要独立已知源的固定响应标定（补偿真实传播相位），以及同初始化下的无人—有人—无人或等价目标变化证据。此处只列缺口，不要求用户现在补采，也不重新索取已确认不存在的文件。

## 全部E1分支

下面列出所有开发分支，未只挑接近真值者；理论只用于冻结后的对照。

{table(compact)}

## 文件

- PROTOCOL.md：冻结协议。
- branch_analysis.py：多分支、几何和空场验证；不加载人体真值。
- evaluate_and_report.py：冻结后的真值对照与报告。
- candidate_ledger.csv：全部局部峰及未关联/失败状态。
- development_branches.csv、validated_branches.csv、branch_evidence.csv：开发、完整验证和失败原因。
- all_branch_truth_evaluation.csv：全部分支的理论角度及逐轮绝对偏差。
- empty_control_scores.csv、empty_control_candidates.csv、empty_repeated_branches.csv：空场反例与重复性。
- geometry_combinations.csv、geometry_interval_diagnostics.csv、release_decisions.csv：几何可行性和停止决定。
- all_branch_tracks.png/pdf、occupied_empty_branch_comparison.png/pdf：分支轨迹及代表性空场对照。
- geometry_alias.py、branch_steering_similarity.csv、array_alias_diagnostic.png/pdf：全部分支对的阵列模型相似度及几何歧义。

图中空场/有人曲线各自归一化，只用于比较角度结构，不能比较绝对人体反射强度。主图全部展示E1分支，C1/C2特写为事后解释示例，不参与规则调整。
'''
    (R/'REPORT_多峰分支与人体候选验证.md').write_text(report,encoding='utf8')
    print(summ.to_string(index=False));print('complete nonboundary near truth retrospective:',len(close));print(intervaldf.to_string(index=False))
if __name__=='__main__':main()
