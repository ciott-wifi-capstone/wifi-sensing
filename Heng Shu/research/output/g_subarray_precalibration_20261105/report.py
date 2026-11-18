from run_check import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABEL={0:'+1 m',12:'-1 m',21:'+2 m',30:'-2 m'}
TRUTH={0:(1.64,1),12:(1.64,-1),21:(1.64,2),30:(1.64,-2)}
NAMES={'identity':'Existing baseline','rf_constant':'RF-chain constant','antenna_constant':'Antenna constant','antenna_tone':'Antenna per-frequency'}
def tab(df):
    q=df.fillna('NA').astype(str)
    return '| '+' | '.join(q.columns)+' |\n| '+' | '.join(['---']*len(q.columns))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in q.values)
def main():
    verify(R/'model_freeze.json');verify(R/'spectral_predictions_frozen.json');verify(R/'decisions_frozen.json')
    v=pd.read_csv(R/'validated_branches.csv');ph=pd.read_csv(R/'phase_prediction.csv');rel=pd.read_csv(R/'release_decisions.csv')
    inv=pd.read_csv(R/'occupied_raw_invariants.csv');rep=pd.read_csv(R/'within_batch_phase_stability.csv');cl=pd.read_csv(R/'shared_pair_closure.csv')
    cp=pd.read_csv(R/'occupied_candidates.csv');null=pd.read_csv(R/'empty_scores.csv')
    # Diagnose why the strongest peak cannot become a stable candidate; no peak rules change.
    rejection=[]
    stats=pd.read_csv(R/'occupied_scores.csv')
    for p in stats.itertuples():
        arr=np.load(R/'spectra'/f'{p.model}_{p.index}_{p.state}.npz');angle=float(sm.ANG[np.argmax(arr['scores'][0])])
        match=cp[(cp.model==p.model)&(cp['index']==p.index)&(cp.state==p.state)&np.isclose(cp.angle_parameter_deg,angle)]
        reasons=[];time_ok=False;freq_ok=False;stable=False
        if not p.matrix_change:reasons.append('background_distance_gate')
        if p.max_score<=p.score_threshold:reasons.append('background_score_gate')
        if len(match):
            h=match.iloc[0];time_ok=bool(h.temporal_blocks>=3);freq_ok=bool(h.frequency_halves_consistent);stable=bool(h.stable)
            if not time_ok:reasons.append('time_blocks')
            if not freq_ok:reasons.append('frequency_halves')
        if abs(angle)>=87:reasons.append('boundary')
        rejection.append(dict(model=p.model,index=p.index,batch_first=p.batch_first,state=p.state,strongest_angle=angle,
                              temporal_pass=time_ok,frequency_pass=freq_ok,stable=stable,boundary=abs(angle)>=87,reasons=';'.join(reasons)))
    rd=save(rejection,'strongest_peak_failure_reasons.csv')
    fail=rd[rd.state.isin(GS)].groupby(['model','state']).agg(total=('index','size'),stable=('stable','sum'),frequency_pass=('frequency_pass','sum'),temporal_pass=('temporal_pass','sum'),boundary=('boundary','sum')).reset_index()
    save(fail,'strongest_peak_failure_summary.csv')
    # Coordinates enter only after candidate and decision freezes have been verified.
    rows=[]
    for p in v.to_dict('records'):
        x,y=TRUTH[p['batch_first']];yc=Y[np.array(sm.MAP[p['state']])-1].mean()
        angle=np.degrees(np.arctan2(-y-yc,3.65-x));p['point']=LABEL[p['batch_first']];p['theory_deg']=angle
        for j in [1,2,3]:p[f'absolute_difference_r{j}']=abs(p[f'angle_r{j}']-angle)
        rows.append(p)
    annotated=save(rows,'all_branch_truth_comparison.csv')
    roots=inv.groupby('state').agg(dominant_fraction=('dominant_eigen_fraction_median','median'),coherence=('pair_coherence_median','median'),unmodelled_phase_RMS=('single_plane_phase_rms_deg','median')).join(rep.groupby('state').phase_median_deg.median().rename('between_round_phase')).reset_index()
    save(roots,'root_cause_state_summary.csv')
    phase=ph[(ph.run==3)&(ph.frequency_scope=='withheld_frequency')&ph.state.isin(GS)].groupby(['model','capture']).agg(passed_G=('phase_prediction_gate','sum'),phase_median=('phase_median_deg','median'),worst_state_p90=('phase_p90_deg','max')).reset_index()
    save(phase,'full_run_phase_summary.csv')
    cal=pd.read_csv(R/'development_model_ranking.csv')
    nullsm=null.groupby(['model','capture']).agg(stable_false_candidates=('stable_candidates','sum'),raw_false_candidates=('raw_candidates','sum'),low_sample_states=('low_sample','sum')).reset_index()
    save(nullsm,'empty_false_candidate_summary.csv')

    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout='constrained')
    cols=plt.get_cmap('tab10').colors
    for i,m in enumerate(MODES):
        q=phase[phase.model==m].set_index('capture').loc[['E1','E2','E3','E4']]
        axes[0].plot(range(4),q.phase_median,'o-',color=cols[i],label=NAMES[m])
        axes[1].plot(range(4),q.passed_G,'o-',color=cols[i],label=NAMES[m])
    for ax in axes:ax.set_xticks(range(4),['E1 R3','E2 R3','E3 R3','E4 R3']);ax.grid(alpha=.25)
    axes[0].set_ylabel('Median CSI phase residual under LoS model (deg)');axes[0].set_title('Unfitted frequencies, full third round');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('G subarrays passing phase-prediction check');axes[1].set_ylim(-.2,4.2);axes[1].set_yticks(range(5));axes[1].set_title('Passing is reference prediction, not hardware validation')
    fig.suptitle('Fixed templates trained only on E1 R1; no human coordinates used',fontsize=12)
    fig.savefig(R/'empty_phase_transfer.png',dpi=180);fig.savefig(R/'empty_phase_transfer.pdf');plt.close(fig)

    display=['identity','antenna_constant','antenna_tone'];fig,axes=plt.subplots(5,3,figsize=(12.5,11.5),sharex=True,sharey=True)
    for row,s in enumerate(GS+['C2']):
        for col,m in enumerate(display):
            ax=axes[row,col]
            for j,idx in enumerate([12,13,14]):
                z=np.load(R/'spectra'/f'{m}_{idx}_{s}.npz');score=z['scores'][0];score=score/score.max()
                ax.plot(sm.ANG,score,color=cols[j],lw=1.4,label=f'R{j+1}')
                qq=cp[(cp.model==m)&(cp['index']==idx)&(cp.state==s)&cp.stable]
                for a in qq.angle_parameter_deg:
                    k=np.where(sm.ANG==a)[0][0];ax.plot(a,score[k],'o',ms=3,color=cols[j])
            ax.axvline(-39,color='black',lw=.7,ls=':');ax.axvline(24,color='black',lw=.7,ls='--')
            ax.grid(alpha=.2);ax.set_ylim(0,1.07);ax.set_xlim(-90,90)
            if row==0:ax.set_title(NAMES[m],fontsize=11)
            if col==0:ax.set_ylabel(s+' normalized score')
            if row==4:ax.set_xlabel('Angle parameter (deg)')
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('-1 m example: fixed phase hypotheses and all three capture rounds',fontsize=13)
    fig.text(.5,.012,'Dots: existing stability rule passed. Vertical lines: old -39 / +24 hypotheses, not validated physical angles. Each curve normalized separately.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,.96]);fig.savefig(R/'G_and_C2_precalibration_comparison.png',dpi=170);fig.savefig(R/'G_and_C2_precalibration_comparison.pdf');plt.close(fig)

    # Frequency-dependence magnitude of the fixed response candidate, not hardware group delay truth.
    tm=np.load(R/'fixed_templates.npz');freq=[]
    for s in ST:
        for k,(i,j) in enumerate(PA):
            z=qp(tm[f'antenna_tone_{s}'])[k];phasecurve=np.unwrap(np.angle(z))
            freq.append(dict(state=s,chain_pair=f'{j+1}-{i+1}',empirical_phase_span_deg=float(np.degrees(np.ptp(phasecurve))),hardware_verified=False))
    save(freq,'frequency_dependence.csv')
    # Connectivity reports effective phase constraints; shared antennas never move to another RF chain here.
    edges=[];ports={}
    for s in ST:
        ids=np.array(sm.MAP[s])-1
        for j,a in enumerate(ids):ports.setdefault(int(a)+1,set()).add(j+1)
        for i,j in PA:
            vv=np.zeros(12);vv[ids[j]]=1;vv[ids[i]]=-1;edges.append(vv)
    topology=dict(equations=len(edges),incidence_rank=int(np.linalg.matrix_rank(edges)),gauge_free_unknown_phases=11,
                  repeated_antennas_switch_RF_chain=any(len(v)>1 for v in ports.values()),hardware_scene_separation_verified=False)
    (R/'graph_identifiability.json').write_text(json.dumps(topology,indent=2))
    closemed=cl.median_deg.median();closemax=cl.p90_deg.max()
    roots_show=roots[roots.state.isin(GS)].round(3)
    angle_votes=pd.read_csv(R/'nominal_C2_vote_counts.csv');angle_votes=angle_votes[angle_votes.batch_first==12]
    samepattern=annotated[(annotated.model=='antenna_constant')&annotated.state.isin(['G3','G4'])][['point','state','angle_r1','angle_r2','angle_r3','complete_repeat']]
    samepattern.to_csv(R/'constant_model_position_specificity.csv',index=False)
    report=f'''# G1–G4不一致、固定初相候选与C2消歧

2026-11-05。有界、回顾性验证，旧数据与结果保留，未修改论文或Overleaf。人体真值只在所有谱和判断冻结后用于描述性对照。

## 结论

现有数据支持稳定的有效相位/信道参考，但不支持一个简单的共同常数初相模型。空场训练的逐频点物理支路模板改善了E1/E2的相位预测，却不能迁移到E3/E4，且没有改善为四点一致的G子阵人体候选。C2原先-39/+24两个候选仍不能可靠区分。

本轮比较无连续校正、两个RF链常数、11个物理支路常数、逐频点共享物理支路四种模型。所有连续模板只由E1空场run1前半段的20个频率槽得到，没有为人体点拟合相位，也没有使用人体坐标选择模型。计算量：288个有人状态-模型比较，96个空场第三轮状态-模型比较；只读既有缓存，没有重解码大批原始CSI。

## G1–G4为什么不一致

按各状态跨记录中位数汇总：

{tab(roots_show)}

dominant_fraction是每频率3x3空间协方差最大特征值占比的中位数，再跨记录取中位数；coherence为链间相干度。between_round_phase是同批次后续轮相对第一轮的CSI相位差，不是AoA误差。unmodelled_phase_RMS是未校准观测与最佳单一平面波相位模型的加权RMS差，也不是人体角度误差。

- G子阵的同批次相位变化通常不大，但其空间频率响应与理想单平面波并不相容；因而“静态、重复”不等于“可由一个真实方向解释”。
- 最大特征值占比高不能证明只有一条传播路径：相干的静态多径复数和也可产生近似秩一的协方差。
- C1/C2对同一物理天线对6/8的相位闭合中位数约{closemed:.3f}度，逐轮P90最大约{closemax:.3f}度。结合已有状态/时间审计，这不支持将主要问题归为普遍状态错位。它仅验证此共享对的相对一致性，不代表12根天线的硬件相位已知。
- 每个重复物理天线始终接同一RF链，无法借重复观测把RF链响应与该天线/线缆/场景响应单独分开。图的18条相位方程秩为11，能给出一套有效相位，但没有独立传播路径参照时，仍不能确定其中多少是硬件、多少是场景。

已有状态映射、解析和时间审计作为来源复用，本次没有声称重新审计全部原始帧。详见相邻e1_e3_mapping_phase_timing_audit_20261104目录。

既有基线的最强峰为何不形成稳定跨组候选：

{tab(fail[fail.model=='identity'])}

统计覆盖每个G的12条记录（含已标记的截断轮）。temporal_pass与frequency_pass分别表示最强峰是否满足时间分块和双频段检查；这两个计数不是独立检验样本数。失稳峰的具体原因逐条保存在strongest_peak_failure_reasons.csv。稳定G峰之间角度不一致时，也不能把它们当作同一人体路径强行融合。

## 初始相位候选是否改善

训练前先用已知TX精确距离d_m=sqrt(3.65^2+y_m^2)扣除真实直达差分传播相位；绝不把所有天线传播相位强制设为相同。但CSI是多径复数和，因此该操作只产生直达主导假设下的有效响应候选。

开发选择仅看E1 run2的未拟合频点，四个G门槛全过才保留候选；此处的度为CSI相位度。

{tab(cal.round(3))}

逐频点模板被选为参考预测候选，随后完整第三轮结果如下：

{tab(phase.round(3))}

门槛为可靠支持>=0.5、中位相位残差<=15度、P90<=30度；它们是预测检查，不是硬件校准或人体路径证明。RF链模型仅用G1/G2拟合，G3/G4没有参与其拟合；支路模型因各组含独有天线，需要全部状态，不能把同一组拟合后靠近TX称为跨状态独立验证。

E1/E2通过、E3/E4失败，说明当前模板是可迁移范围有限的参考。不能把这一现象唯一归为硬件故障或环境变化，也不能只从有人记录的相位偏离判断模型错误，因为人体本身可能改变信道。

## 四点谱和独立G支持

{tab(rel[['model','batch_first','complete_nonboundary_G_states','three_G_geometry','human_output']])}

批次0/12/21/30分别对应+1/-1/+2/-2 m。+1 m第3轮不完整，完整重复条件不满足；没有删除这条记录。谱边界峰保留但不作为有效方向。四种模型均未让任一点通过三个独立G的完整重复和几何相容性。特别是相位预测最好的逐频点模板，在四点上没有得到完整、非边界的G分支。

所有图中的谱是同频空间协方差背景评分，不是新二维MUSIC或人体概率。每种相位模型同时变换有人数据与背景，导向矢量保留理想几何，这检验的是“候选相位确为硬件”的条件假设。另将导向矢量也一致变换的控制保持评分不变，排除了仅换坐标造成的伪改进。

## C2的-39/+24消歧尝试

以原始两个角度数值为条件假设，统计-1 m中各G是否在三个完整轮次均有稳定峰位于其9度内：

{tab(angle_votes)}

常数支路模型使G3三轮为+24度、G4三轮为+21度，数值上给+24假设两票。但这不能接受为消歧成功：

1. 该模型的E1开发相位检查是0/4个G通过，不能因人体点看起来较好就改选它。
2. G1/G2没有提供支持，未通过至少三个独立G门槛；G3和G4还分别与C2共享天线8和10。只有G1与C2完全不重叠。
3. 校正后C2没有保留完整、非边界的原双分支，不能把旧的未校准-39/+24直接当作新物理角度。

更关键的是，常数支路模型对四点的G3/G4都输出相似正角度，并非只在-1 m出现：

{tab(samepattern)}

这是重复的谱形归一化现象，与“候选方向跟随四点人体站位变化”的要求不相符。+1 m第三轮截断仍保留标记。不能只抽取-1 m这一行，再把其接近真值称为定位准确度改善。

同时枚举各新谱自身的C2重复分支及其理想响应相似度，不只检查旧两个数值。所有可检出的高相似分支对均未获得唯一、合格的G支持，见reestimated_C2_alias_pairs.csv。

从理想阵列几何看，G1–G4对-39/+24的导向矢量相似度约0.333，而C2约0.998。也就是说，G子阵在可靠校准后本来可以提供消歧信息；当前困难不是28 mm子阵也具有同样的近完全混叠，而是实际响应尚未被可靠标定、路径归属未确认。该相似度是理想模型量，不是实测CSI相关系数或人体概率。

## 空场反例与样本量

{tab(nullsm)}

各模型在E1/E2第三轮没有稳定假候选，但在E3/E4仍有许多；只靠减少跨场空场假候选数不能证明人体灵敏度。E1/G2第三轮去掉首64包后只有79包，必须保留低样本标记，零假候选不能单独证明稳定。E4相对E1的有限pi符号由既有两段变换组合，属于诊断假设；主要失败结论在E3已成立，不依赖E4。

## 当前可保留的实现与停止条件

- 保留所有模型和逐频点模板作为有效参考诊断；不把它写成已经完成硬件标定。
- 保留同批次多峰追踪，不改用某个“接近人体真值”的校正模型。
- C2旧双解及新谱中的双解都输出UNCONFIRMED；不计算已验证人体AoA MAE或位置误差。
- 最小缺口仍是独立可识别的已知源/直达径标定，以及同初始化的人体出现/离开或配对背景证据。用户已确认没有额外记录，本次没有等待补采，也没有虚构标定。

## 复现与证据

run_check.py → disambiguate.py → report.py → verify.py。先冻结模型，再冻结候选与消歧判断，最后才在report.py加载人体坐标。全部输入缓存哈希写入source_manifest.csv，旧结果不覆盖。

- phase_parameters.csv / fixed_templates.npz：有效相位候选，非已测定硬件相位。
- within_batch_phase_stability.csv / shared_pair_closure.csv：相对稳定性与共享天线证据。
- phase_prediction.csv / full_run_phase_summary.csv：逐状态、完整轮次及未拟合频点验证。
- occupied_candidates.csv / validated_branches.csv：所有局部候选及跨轮分支，不按真值筛选。
- all_branch_truth_comparison.csv：冻结后所有分支的真值对照，不是选峰规则。
- nominal_C2_hypotheses.csv / reestimated_C2_alias_pairs.csv：旧数值假设和重新估计分支的分别检验。
- coordinate_controls.csv：一致变换的数据/背景/导向矢量不改变评分。
- empty_phase_transfer.png/pdf：空场相位参考转移。
- G_and_C2_precalibration_comparison.png/pdf：-1 m代表性三轮全谱，其他点数据完整保留CSV/NPZ。

## 检查记录

模型、候选与消歧决策哈希已验证；identity逐谱完全复现前一轮基线；共享图已知相位合成控制、21度平面波方向控制、公共包相位/定时不变性、一致坐标变换不变性均通过。相干静态多径的合成反例说明：即使真实硬件相位为零，经验直达主导模板也可反推出非零“初相”，所以拟合良好不等于硬件校准。详细数值在verification.json。两幅PNG已检查标注、图例和裁切，矢量PDF同时保留。
'''
    (R/'REPORT_G子阵初相预校准与C2消歧.md').write_text(report,encoding='utf8')
    print('report saved; all four models: no validated human or position outputs')
if __name__=='__main__':main()
