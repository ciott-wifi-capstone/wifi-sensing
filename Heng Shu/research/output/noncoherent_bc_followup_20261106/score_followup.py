"""Join survey geometry only after the new prediction artifacts are frozen."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'noncoherent_uniform_20261106'
POS={'A':(.8,.6),'B':(2.8,1.2),'C':(.8,-1.2),'D':(2.4,-1.6),'E':(1.64,-2.01)}
TRUTH={p:np.degrees(np.arctan2(-y,3.65-x)) for p,(x,y) in POS.items()}
OLD=['G2_BG','SIX_BG','SIX_BG_R1','G2_POS','SIX_POS','G2_SIGNED','SIX_SIGNED','G2_WBG','SIX_WBG','SIX_WBG_R1']
NEW=['G2_PCD','SIX_PCD_POOL','SIX_PCD_BAL','SIX_PCD_R1','SIX_SIGNED_BAL','SIX_SIGNED_R1']
PRIMARY='SIX_PCD_R1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,df):df.to_csv(ROOT/n,index=False,encoding='utf-8-sig')
frozen=json.loads((ROOT/'prediction_freeze.json').read_text('utf8'))
assert all(sha(ROOT/n)==h for n,h in frozen.items())
prior_frozen=json.loads((PREV/'prediction_freeze.json').read_text('utf8'))
assert all(sha(PREV/n)==h for n,h in prior_frozen.items())
previous=pd.read_csv(PREV/'blind_predictions.csv');previous['source']='previous frozen'
new=pd.read_csv(ROOT/'blind_predictions.csv');new['source']='new predeclared'
pred=pd.concat([previous,new],ignore_index=True)
pred['theory_rx_deg']=pred.point.map(TRUTH)
for col,out in [('prediction','error'),('held','held_error'),('whole','whole_error')]:
    pred[out]=np.where(pred.run==2,abs(pred[col]-pred.theory_rx_deg),np.nan)
save('all_scored_predictions.csv',pred)
h=pred[pred.run==2];e=pred[pred.run==3];rows=[]
for arm in OLD+NEW:
    g=h[h.arm==arm];z=e[e.arm==arm];n=int(g.error.notna().sum());accepted=g[g.paired_candidate]
    rows.append(dict(arm=arm,source=g.source.iloc[0],numerical=n,mean=g.error.mean() if n==5 else np.nan,
      median=g.error.median() if n==5 else np.nan,maximum=g.error.max() if n==5 else np.nan,
      available_only_mean=g.error.mean(),whole_mean=g.whole_error.mean() if g.whole_error.notna().sum()==5 else np.nan,
      accepted=int(g.paired_candidate.sum()),conditional_separation=accepted.error.mean(),time_pass=int(g.pass_time.sum()),
      frequency_pass=int(g.pass_frequency.sum()),unique_pass=int(g.pass_unique.sum()),empty_false=int(z.pre_return_candidate.sum()),physical_release=int(g.physical_release.sum())))
s=pd.DataFrame(rows);save('all_method_summary.csv',s)
checks=[]
p=h[h.arm==PRIMARY].set_index('point').loc[list(POS)]
for baseline in ['G2_BG','G2_PCD','SIX_WBG_R1']:
    b=h[h.arm==baseline].set_index('point').loc[list(POS)];a=s.set_index('arm').loc[PRIMARY];bb=s.set_index('arm').loc[baseline]
    conditions=dict(all5=bool(a.numerical==5 and bb.numerical==5),lower_mean=bool(a['mean']<bb['mean']),median_not_worse=bool(a['median']<=bb['median']),max_not_worse=bool(a.maximum<=bb.maximum),improve_at_least4=bool((p.error<b.error).sum()>=4),coverage_positive_not_worse=bool(a.accepted>0 and a.accepted>=bb.accepted),false_not_more=bool(a.empty_false<=bb.empty_false))
    checks.append(dict(baseline=baseline,improved_points=int((p.error<b.error).sum()),all_gain_criteria=all(conditions.values()),**conditions))
save('gain_checks.csv',pd.DataFrame(checks))
focus=h[h.arm.isin(['G2_BG','SIX_WBG_R1','G2_PCD',PRIMARY])].copy();save('five_point_comparison.csv',focus)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def figsave(name):
    plt.savefig(ROOT/(name+'.pdf'),bbox_inches='tight');plt.savefig(ROOT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close()
fig,axs=plt.subplots(1,4,figsize=(13,4.3),sharex=True,sharey=True)
for ax,arm in zip(axs,['G2_BG','SIX_WBG_R1','G2_PCD',PRIMARY]):
    g=h[h.arm==arm].set_index('point').loc[list(POS)]
    for i,(_,x) in enumerate(g.iterrows()):
        ax.plot([x.prediction,x.held],[i,i],color='#aab5c4',lw=2)
        ax.plot(x.prediction,i,'o',c='#276c9e');ax.plot(x.held,i,'s',mfc='none',mec='#cc7626');ax.plot(x.theory_rx_deg,i,'kx',ms=8)
        if not np.isfinite(x.prediction):ax.text(-90,i-.17,'No first-half output',ha='left',color='#b33a3a',fontsize=8)
    ax.set(title=arm.replace('SIX_','12 noncoh\n').replace('_',' '),yticks=range(5),yticklabels=list(POS),xlim=(-95,95),ylim=(-.6,4.6),xlabel='Bearing parameter / deg');ax.grid(alpha=.15)
axs[0].invert_yaxis();fig.text(.5,0,'Blue: first half    Orange: later half    Black: evaluation geometry; no human-path release',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));figsave('uniform_comparison')
oldspec=np.load(PREV/'spectra.npz');newsp=np.load(ROOT/'spectra.npz')
fig,axs=plt.subplots(2,3,figsize=(12,6.5),sharex=True,sharey=True)
for i,pt in enumerate('BC'):
    for j,arm in enumerate(['G2_BG','SIX_WBG_R1',PRIMARY]):
        data=newsp if arm in NEW else oldspec;ax=axs[i,j]
        has_spectrum=False
        for block,b,label,style in [('R2_first',0,'First','-'),('R2_last',0,'Later','--'),('R2_first',1,'Low',':'),('R2_first',2,'High','-.')]:
            cost=data[f'{pt}_{arm}_{block}'][b]
            if np.isfinite(cost).any():
                has_spectrum=True
                spec=-10*np.log10(np.maximum(cost,1e-12));spec-=np.nanmax(spec);ax.plot(data['angles'],spec,style,lw=1,label=label)
        ax.axvline(TRUTH[pt],c='black',lw=.8,label='Geometry');ax.set(title=f'{pt}: {arm}',ylim=(-25,1),xlim=(-90,90));ax.grid(alpha=.15)
        if not has_spectrum:ax.text(.5,.45,'No admissible spectrum\n(rank / tone-support gate)',transform=ax.transAxes,ha='center',color='#b33a3a')
        if i==0:ax.legend(fontsize=7,ncol=3)
        if i==1:ax.set_xlabel('Bearing parameter / deg')
        if j==0:ax.set_ylabel('Normalized score / dB')
fig.tight_layout();figsave('BC_spectra')
fig,ax=plt.subplots(figsize=(10,7))
show=s[s.arm.isin(['G2_BG','SIX_WBG_R1']+NEW)].reset_index(drop=True)
for i,r in show.iterrows():
    if np.isfinite(r['mean']):ax.barh(i,r['mean'],color='#377f9d' if r.arm.startswith('G2') else '#ca8c53');ax.text(r['mean']+1,i,f'{r["mean"]:.2f}°; {r.accepted}/5 accepted',va='center',fontsize=9)
    else:ax.text(2,i,f'All-five mean NA; {r.numerical}/5 numerical',va='center',fontsize=9)
ax.set(yticks=range(len(show)),yticklabels=show.arm,xlim=(0,95),xlabel='Mean candidate-to-geometry separation, not validated human AoA accuracy');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout();figsave('method_summary')

def fmt(x):return f'{x:.2f}' if np.isfinite(x) else 'NA'
def angle(x):return f'{x:+.0f}°' if np.isfinite(x) else '无输出'
def line(xs):return '| '+' | '.join(map(str,xs))+' |'
report=['# A–E 十二天线非相干融合：B/C 原因与统一改进复核','',
'本次在已反复查看的开发数据上开展一次有界试验，没有独立测试集。所有方案统一处理五点，没有逐点挑最好；保留旧结果、失败输出及空场对照。主方案和六种新比较在评分前写入 protocol.json 并冻结。','',
'## 本轮方法','',
'主要改法为投影后协方差差分 MUSIC（PCD-MUSIC，本项目实验方案）：D=B(R−R1)B。旧 BG 拟合 B R B，R1 投影残余只用于门槛；新方法实际扣除残余背景，然后拟合显著正变化模态。只有一个显著正模态才可用于三链投影后的 MUSIC；两个会耗尽噪声子空间，明确拒绝。负变化也记录，不直接解释为人体。',
'','PCD 的 pool / 等权状态均衡 / R1 权重版本使用相同的逐状态角度有效域，隔离汇总方式和权重影响。另对既有 signed covariance-change MUSIC 加入等权状态均衡和 R1 加权。主方案预先指定为 SIX_PCD_R1；其余是敏感性比较，不事后替换主方案。',
'','每点每轮两种阵列均为864个不同包。R1前288包建模，其余留出；R2前432包预测、后432包和半频段验证；R3完全不用于拟合。数学检查验证代数和退化拒绝，不证明现实人体识别。',
'','## 五点统一对照','',line(['点','理论','G2 BG角/偏差','旧12 WBG-R1角/偏差','G2 PCD角/偏差','新12 PCD-R1角/偏差','新12后半段','候选通过']),line(['---']*8)]
for pt in POS:
    g=h[h.point==pt].set_index('arm');row=[pt,f'{TRUTH[pt]:+.2f}°']
    for arm in ['G2_BG','SIX_WBG_R1','G2_PCD',PRIMARY]:row.append(f'{angle(g.loc[arm,"prediction"])} / {fmt(g.loc[arm,"error"])}°')
    row +=[angle(g.loc[PRIMARY,'held']),'是' if g.loc[PRIMARY,'paired_candidate'] else '否'];report.append(line(row))
report+=['','偏差统一使用 RX 阵列中心的几何方向。它不是已验证人体 AoA 精度；未接受的峰不计作成功定位。整轮结果仅为描述性结果，不能替换前半段预测。','',
'## 所有固定方法','',line(['方法','数值输出','五点平均','中位','最大','整轮平均','时间/频段通过','候选通过','空场假候选']),line(['---']*9)]
for _,r in s.iterrows():report.append(line([r.arm,f'{r.numerical}/5',fmt(r['mean']),fmt(r['median']),fmt(r.maximum),fmt(r.whole_mean),f'{r.time_pass}/{r.frequency_pass}',f'{r.accepted}/5',f'{r.empty_false}/5']))
report += ['', '### 固定 SIGNED-R1 敏感性方案的五点值', '',
  '该方案在计算前已声明，统一应用于五点。其首段选峰与旧 SIX_SIGNED 完全相同；不能把相对旧 WBG-R1 的降低归功于本次新增权重。它没有替代本轮 PCD 主方案，也不是独立验证的获胜方案。', '',
  line(['点','G2 BG角/偏差','SIGNED-R1角/偏差','后半段','失败']),line(['---']*5)]
for pt in POS:
    g=h[h.point==pt].set_index('arm');v=g.loc['SIX_SIGNED_R1'];b=g.loc['G2_BG']
    report.append(line([pt,f'{angle(b.prediction)} / {fmt(b.error)}°',f'{angle(v.prediction)} / {fmt(v.error)}°',angle(v.held),v.failure]))
report+=['','NA 表示全五点指标无法定义，不能删除无输出的点再报告较好均值。所有输出和失败原因见 all_scored_predictions.csv。','',
'## 主方案验收与歧义','']
for c in checks:report.append(f'- 相对 {c["baseline"]}：改善 {c["improved_points"]}/5 点；整体收益验收={c["all_gain_criteria"]}；未通过项：'+', '.join(k for k,v in c.items() if isinstance(v,bool) and not v and k!='all_gain_criteria')+'。')
report +=['',line(['点','第一候选','第二候选','间隔dB','失败']),line(['---']*5)]
for pt,r in p.iterrows():
    alts=json.loads(r.alternatives);report.append(line([pt,angle(r.prediction),angle(alts[1]['angle']) if len(alts)>1 else '无可用次峰',fmt(r.margin_db),r.failure or '无内部失败']))
report+=['','## B/C 原因与证据','',
'详细诊断见 diagnostics/BC原因审计.md。B 的共同近零偏置伴随明显 R1→R3 空场变化，不能单归因于 G1。C 的子阵方向和半频段不相容；C2 的 56 mm 间距产生近等价角度分支，空场相干权重不能保证正确消歧。未标定的支路相位、相干环境多径和人体遮挡仍可能混淆。','',
'## 复现和边界','',
'`predict_followup.py` 仅加载已选缓存；`pcd_kernel.py` 定义新内核；`score_followup.py` 是加入人体坐标的评分步骤。`pre_execution_freeze.json` 和 `prediction_freeze.json` 保存前后冻结哈希。旧 G2 BG、G2 SIGNED 和十二 WBG-R1 谱逐值复现通过；同一包公共相位/定时问题仍由同频协方差消除，固定链路响应并未因此标定。',
'','`tone_support.csv` 保留各窗口正模态0/1/2及有效频点；`state_diagnostics.csv` 为逐状态候选；`projection_retention.csv` 给出完整角度域的背景投影保留率。全状态正权重参与，不称为完整12阵元相干MUSIC。没有对人体真值拟合相位或选择局部峰。',
'','![统一比较](uniform_comparison.png)','','![B/C谱](BC_spectra.png)','','![统一方法汇总](method_summary.png)']
(ROOT/'统一改进验证报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
verification=dict(prediction_hashes_verified=True,prior_hashes_verified=True,packets_per_arm_run=864,prediction_packets=432,all_points_retained=True,old_spectra_reproduced=True,per_point_method_selection=False,independent_new_test=False,physical_release=0,primary=PRIMARY,gain_checks=checks,changed_raw_data=False,thesis_modified=False)
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
print(s.to_string(index=False));print('\nFOCUS\n',focus[['point','arm','prediction','held','error','failure']].to_string(index=False));print('\nGAIN',checks)
