"""Join surveyed geometry only after prediction_freeze.json exists."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
POINTS={'A':(.8,.6),'B':(2.8,1.2),'C':(.8,-1.2),'D':(2.4,-1.6),'E':(1.64,-2.01)}
ORDER=['G2_BG','SIX_BG','SIX_BG_R1','G2_POS','SIX_POS','G2_SIGNED','SIX_SIGNED','G2_WBG','SIX_WBG','SIX_WBG_R1']
LABEL={'G2_BG':'G2 / BG','SIX_BG':'12 noncoh / BG','SIX_BG_R1':'12 noncoh / BG + R1 weights','G2_POS':'G2 / positive change','SIX_POS':'12 noncoh / positive change','G2_SIGNED':'G2 / signed change','SIX_SIGNED':'12 noncoh / signed change','G2_WBG':'G2 / whitened BG','SIX_WBG':'12 noncoh / whitened BG','SIX_WBG_R1':'12 noncoh / whitened BG + R1 weights'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def angle(x):return f'{x:+.0f}°' if np.isfinite(x) else '无输出'
def fmt(x):return f'{x:.2f}' if np.isfinite(x) else 'NA'
frozen=json.loads((R/'prediction_freeze.json').read_text('utf8'))
assert all(sha(R/n)==h for n,h in frozen.items())
pred=pd.read_csv(R/'blind_predictions.csv')
truth={p:np.degrees(np.arctan2(-y,3.65-x)) for p,(x,y) in POINTS.items()}
pred['theory_rx_deg']=pred.point.map(truth)
pred['theory_G2_deg']=pred.point.map({p:np.degrees(np.arctan2(-y-.042,3.65-x)) for p,(x,y) in POINTS.items()})
for col,name in [('prediction','error'),('held','held_error'),('whole','whole_error')]:
    pred[name]=np.where(pred.run==2,abs(pred[col]-pred.theory_rx_deg),np.nan)
pred['G2_local_error']=np.where((pred.run==2)&pred.arm.str.startswith('G2_'),abs(pred.prediction-pred.theory_G2_deg),np.nan)
save('scored_predictions.csv',pred)
human=pred[pred.run==2].copy();empty=pred[pred.run==3].copy()
rows=[]
for arm in ORDER:
    g=human[human.arm==arm];null=empty[empty.arm==arm];n=int(g.error.notna().sum());ok=g[g.paired_candidate]
    rows.append(dict(arm=arm,numerical_outputs=n,all5_MAE=g.error.mean() if n==5 else np.nan,
      available_MAE=g.error.mean(),median=g.error.median() if n==5 else np.nan,
      maximum=g.error.max() if n==5 else np.nan,whole_MAE=g.whole_error.mean() if g.whole_error.notna().sum()==5 else np.nan,
      time_pass=int(g.pass_time.sum()),frequency_pass=int(g.pass_frequency.sum()),unique_pass=int(g.pass_unique.sum()),
      pre_return_candidates=int(g.pre_return_candidate.sum()),paired_candidates=int(g.paired_candidate.sum()),
      candidate_conditional_separation=ok.error.mean(),empty_false_candidates=int(null.pre_return_candidate.sum()),physical_release=int(g.physical_release.sum())))
summary=pd.DataFrame(rows);save('method_summary.csv',summary)
primary=human[human.arm.isin(['G2_BG','G2_WBG','SIX_BG','SIX_WBG_R1'])].copy();save('primary_five_point.csv',primary)

# Read-only diagnostics from frozen spectra and existing packet ledgers.
# No new candidate rule or fitted parameter is introduced at scoring time.
import predict_uniform as diagnostic
frozen_spectra=np.load(R/'spectra.npz');state_rows=[];time_rows=[]
for pt in POINTS:
    for state in (*diagnostic.ST,'G2_control'):
        for block in ['R2_first','R2_last','R3_full']:
            sp=frozen_spectra[f'{pt}_state_{state}_WBG_{block}']
            theta,cs,margin=diagnostic.peaks(sp[0])
            state_rows.append(dict(point=pt,state=state,block=block,peak=theta,
                low=diagnostic.peaks(sp[1])[0],high=diagnostic.peaks(sp[2])[0],
                margin_db=margin,candidates=json.dumps(cs)))
    ledger=pd.read_csv(diagnostic.OLD/f'{pt}_packets.csv')
    for (run,arm),g in ledger.groupby(['run','arm']):
        first=g[g['sample']<12];held=g[g['sample']>=12]
        assert first.packet.is_unique and held.packet.is_unique
        assert not set(first.packet)&set(held.packet)
        assert first.time_s.max()<held.time_s.min()
        time_rows.append(dict(point=pt,run=run,arm=arm,
            first_min=first.time_s.min(),first_max=first.time_s.max(),
            held_min=held.time_s.min(),held_max=held.time_s.max(),chronological_disjoint=True))
save('state_WBG_diagnostics.csv',pd.DataFrame(state_rows))
save('temporal_split_audit.csv',pd.DataFrame(time_rows))
gates=[]
for base in ['G2_BG','G2_WBG']:
    p=human[human.arm=='SIX_WBG_R1'].set_index('point').loc[list(POINTS)];b=human[human.arm==base].set_index('point').loc[list(POINTS)]
    s=summary.set_index('arm');a=s.loc['SIX_WBG_R1'];bb=s.loc[base]
    c=dict(all5_numerical=int(a.numerical_outputs)==5,lower_mean=bool(a.all5_MAE<bb.all5_MAE),
      no_worse_median=bool(a['median']<=bb['median']),no_worse_maximum=bool(a.maximum<=bb.maximum),
      improve_at_least4=int((p.error<b.error).sum())>=4,
      coverage_nonzero_and_no_worse=bool(a.paired_candidates>0 and a.paired_candidates>=bb.paired_candidates),
      no_more_empty_false_candidates=bool(a.empty_false_candidates<=bb.empty_false_candidates))
    gates.append(dict(baseline=base,improved_points=int((p.error<b.error).sum()),all_criteria_pass=all(c.values()),**c))
save('predeclared_gain_checks.csv',pd.DataFrame(gates))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def figsave(n):
    plt.savefig(R/(n+'.pdf'),bbox_inches='tight');plt.savefig(R/(n+'.png'),dpi=170,bbox_inches='tight');plt.close()
fig,axs=plt.subplots(1,3,figsize=(11,4.1),sharex=True,sharey=True)
for ax,arm in zip(axs,['G2_BG','G2_WBG','SIX_WBG_R1']):
    g=human[human.arm==arm].set_index('point').loc[list(POINTS)]
    for i,(_,x) in enumerate(g.iterrows()):
        ax.plot([x.prediction,x.held],[i,i],c='#9db0c1',lw=2)
        ax.plot(x.prediction,i,'o',c='#276c9e');ax.plot(x.held,i,'s',mfc='none',mec='#ce7626');ax.plot(x.theory_rx_deg,i,'kx',ms=8)
        ax.annotate(angle(x.prediction).replace('°',''),(x.prediction if np.isfinite(x.prediction) else 0,i),xytext=(0,9),textcoords='offset points',fontsize=8,ha='center')
    title={'G2_BG':'G2: original BG','G2_WBG':'G2: whitened BG','SIX_WBG_R1':'12 antennas: noncoherent\nwhitened BG + R1 weights'}[arm]
    ax.set(title=title,yticks=range(5),yticklabels=list(POINTS),xlim=(-95,95),ylim=(-.6,4.7),xlabel='Bearing parameter / deg');ax.grid(alpha=.2)
axs[0].invert_yaxis();axs[0].set_ylabel('Standing point')
fig.text(.5,.0,'Blue circle: first-half prediction    Orange square: held half    Black cross: geometry (evaluation only)',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));figsave('primary_angle_comparison')

fig,ax=plt.subplots(figsize=(9,5.7));y=np.arange(len(summary));colors=['#2576a4' if x.startswith('G2') else '#e5a45d' for x in summary.arm];colors[-1]='#c74a43'
for i,row in summary.iterrows():
    if np.isfinite(row.all5_MAE):
        ax.barh(i,row.all5_MAE,color=colors[i]);ax.text(row.all5_MAE+1,i,f'{row.all5_MAE:.2f} | candidates {row.paired_candidates}/5',va='center',fontsize=8)
    else:ax.text(1,i,f'NA: {row.numerical_outputs}/5 numerical outputs; candidates {row.paired_candidates}/5',va='center',fontsize=8)
ax.set(yticks=y,yticklabels=[LABEL[x] for x in summary.arm],xlabel='Mean geometric separation / deg (all 5 predictions)',xlim=(0,82));ax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout();figsave('all_uniform_methods')

weights=pd.read_csv(R/'R1_frozen_weights.csv');ww=weights[weights.state!='G2_control'].pivot(index='point',columns='state',values='weight').loc[list(POINTS),list(['G1','G2','G3','G4','C1','C2'])]
fig,ax=plt.subplots(figsize=(7,3.5));im=ax.imshow(ww,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set(xticks=range(6),xticklabels=ww.columns,yticks=range(5),yticklabels=ww.index,xlabel='State (all weights remain positive)',ylabel='Standing point')
for i in range(5):
    for j in range(6):ax.text(j,i,f'{ww.iloc[i,j]:.3f}',ha='center',va='center',color='white' if ww.iloc[i,j]>.55 else 'black')
fig.colorbar(im,ax=ax,label='R1-only unnormalized reliability weight');fig.tight_layout();figsave('R1_reliability_weights')

z=np.load(R/'spectra.npz');fig,axs=plt.subplots(5,2,figsize=(11,12),sharex=True)
for i,p in enumerate(POINTS):
    for j,arm in enumerate(['G2_BG','SIX_WBG_R1']):
        ax=axs[i,j]
        for b,k,label,style in [('R2_first',0,'First half','-'),('R2_last',0,'Held half','--'),('R2_last',1,'Held low',':'),('R2_last',2,'Held high','-.')]:
            cost=z[f'{p}_{arm}_{b}'][k];finite=np.isfinite(cost)
            if finite.any():
                val=-10*np.log10(np.maximum(cost,1e-12));val-=np.nanmax(val);ax.plot(z['angles'],val,style,lw=1,label=label)
        ax.axvline(truth[p],color='black',lw=.9,alpha=.6,label='Geometry');ax.set(title=f'{p}: {LABEL[arm]}',ylim=(-25,1),xlim=(-90,90));ax.grid(alpha=.15)
        if i==0:ax.legend(fontsize=7,ncol=2)
        if j==0:ax.set_ylabel('Normalized spectrum / dB')
        if i==4:ax.set_xlabel('RX-local bearing parameter / deg')
fig.tight_layout();figsave('primary_spectral_stability')

def line(vals):return '| '+' | '.join(str(x) for x in vals)+' |'
report=['# A–E：统一方法、相同包数的十二天线非相干融合验证','',
'本轮没有逐点选择最好方法。先冻结方案，再统一计算 A–E。原始数据、旧结果和论文未改动；本报告是已反复查看数据上的探索性验证，不是新的独立盲测。','',
'## 1. 本轮固定了什么','',
'- 每点、每轮、每种阵列均为 **864 个不同 CSI 包**。十二天线方案使用六个状态，每态 144 包；G2 使用 864 个 G2 包，覆盖同一采集时段。',
'- 每点均用自己的 R1 前 288 包建立空场模型与权重；其余 R1 留作检查。R2 前 432 包给出预测，后 432 包、四个时间块、两半频段验证。R3 全部保留为空场对照。',
'- 预先指定主方案：**空场重复性加权的正则化白化背景投影 MUSIC（WBG-R1，本研究定义的实验方案）**。正则化系数固定为 0.1，不搜索参数，不按人体真值设置权重。',
'- G2 使用相同白化 MUSIC 内核作控制，同时保留原 G2 背景投影 MUSIC，避免通过换弱基线制造收益。',
'- 其余均为预先声明的统一敏感性比较：正变化一阶、正负变化自适应阶数、普通背景投影，以及不加权／空场加权。每个方案处理全部五点。',
'- 非相干方案从六态的归一化 MUSIC 残差融合，不拼接复数 CSI；C1/C2 有共享天线，不能视为互相独立的验证器。','',
'## 2. 预先指定主方案的五点结果','',
'角度与偏差都采用统一 RX 阵列中心方向。偏差是候选峰与几何方向的差，不是已验证的人体 AoA 误差。',
line(['点','理论','G2 原 BG：预测 / 偏差','G2 同内核 WBG：预测 / 偏差','十二天线 WBG-R1：预测 / 偏差','十二天线后半段','候选通过']),
line(['---']*7)]
for p in POINTS:
    g=human[human.point==p].set_index('arm');vals=[p,f'{truth[p]:+.2f}°']
    for arm in ['G2_BG','G2_WBG','SIX_WBG_R1']:vals.append(f'{angle(g.loc[arm,"prediction"])} / {fmt(g.loc[arm,"error"])}°')
    vals += [angle(g.loc['SIX_WBG_R1','held']),'是' if g.loc['SIX_WBG_R1','paired_candidate'] else '否']
    report.append(line(vals))
report+=['','## 3. 全部统一方案，不能按点换方法','',line(['方案','数值输出','五点平均偏差','中位','最大','整轮描述性平均','候选通过','R3 假候选']),line(['---']*8)]
for _,x in summary.iterrows():report.append(line([x.arm,f'{x.numerical_outputs}/5',fmt(x.all5_MAE),fmt(x['median']),fmt(x.maximum),fmt(x.whole_MAE),f'{x.paired_candidates}/5',f'{x.empty_false_candidates}/5']))
report+=['','整轮结果不能替换前半段预测。数值输出不足五点时，五点平均写 NA，不丢弃失败点后冒充全五点平均。R3 假候选是本次五段空场上的内部检查，不是可推广的假警率。','',
'## 4. 是否满足预先写定的稳定收益条件','']
for g in gates:
    report.append(f'- 相对 {g["baseline"]}：改善 {g["improved_points"]}/5 点；全部条件通过：**{g["all_criteria_pass"]}**。失败项：'+', '.join(k for k,v in g.items() if isinstance(v,bool) and not v and k!='all_criteria_pass')+'。')
report+=['','条件要求：五点均有数值输出，平均偏差降低，中位与最差点不退步，至少四点改善，通过候选验收的数量不低于基线且大于零，R3 假候选不增加。即使这些条件通过，也还需要独立校准与新重复采集才能声称人体定位收益。','',
'## 5. 主方案失败记录和双解','',line(['点','失败检查','第一候选','第二候选','候选间隔 dB']),line(['---']*5)]
for _,x in human[human.arm=='SIX_WBG_R1'].iterrows():
    cs=json.loads(x.alternatives);second=angle(cs[1]['angle']) if len(cs)>1 else '无'
    report.append(line([x.point,x.failure or '无数值门槛失败',angle(x.prediction),second,fmt(x.margin_db)]))
report+=['','失败词：time=前后半段不一致；frequency=半频段不一致；quarters=少于三个时间块支持；unique=第二候选在 3 dB 内；above_R1held=变化未超过留出空场；R3_control=未超过后空场或后空场有同方向稳定峰；nonedge=无输出或边界峰。',
'','## 6. 证据、复现与边界','',
'- `protocol.json`：计算前的主方案、敏感性方案、门槛与停止条件。',
'- `predict_uniform.py`：预测代码，不读取人体坐标；`R1_weight_freeze.json` 在 R2 计算前固定权重；`prediction_freeze.json` 在真值评分前固定所有预测与谱。',
'- `score_uniform.py`：唯一加入人体坐标的评分及作图步骤。`scored_predictions.csv` 保留全部方法和空场记录。',
'- `packet_budget.csv`、`source_manifest.csv`、`input_cache_hashes.json`：包数、文件和哈希。复用的 G2 背景投影谱与此前基线逐值一致。',
'- `tone_support.csv`、`reference_transfer.csv`、`leave_G_state_out.csv`：缺失频点、空场参考迁移、留一 G 状态敏感性，均未按真值筛除。',
'- 本轮没有独立已知源的硬件相位标定；没有新的完整人体重复实验。物理人体方向、ToF 和二维定位验收仍未完成。','',
'![统一预测与留出对照](primary_angle_comparison.png)','',
'![全部统一方法](all_uniform_methods.png)','',
'![仅空场权重](R1_reliability_weights.png)','',
'![时间与频段稳定性](primary_spectral_stability.png)']
report[2:2]=[(R/'INTERPRETATION.md').read_text(encoding='utf8')]
report += ['', '## 7. 方法与软件复核补充', '',
  '- 公式和复现步骤见 [METHOD.md](METHOD.md)。所有比较均来自预先声明的方法。',
  '- `state_WBG_diagnostics.csv`：六态与等包数 G2 的分频段、前后半段和后空场候选。',
  '- `temporal_split_audit.csv`：每点、每轮、每方案的前后半段时间范围；包无重叠且严格按时间顺序划分。',
  '- `endpoint_fix_audit.json` 与 `before_endpoint_fix/`：纠正候选代码把非极小端点作为次峰的问题。修复前后谱数组、权重及全部主峰完全相同，单解和候选通过数均不变；未修改阈值。',
  '- 图中实线与虚线只是同记录内部的重复性证据，不是独立测试会话。四幅图已检查文字、图例和坐标，提供 PNG 与矢量 PDF。']
(R/'统一非相干融合验证报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
verification={'prediction_hashes_verified':True,'previous_G2_spectra_reproduced':True,'points':5,'arms':10,'primary':'SIX_WBG_R1','primary_gain_checks':gates,'per_point_method_switch':False,'independent_new_test':False,'physical_release':0}
(R/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
print(summary.to_string(index=False));print('\nPRIMARY FIVE POINTS');print(primary[['point','arm','prediction','error','held','whole','paired_candidate','failure']].to_string(index=False));print('\nGAIN CHECKS',gates)
