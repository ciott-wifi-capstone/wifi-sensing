"""Interpret frozen predictions and explain missing outputs; no new estimation."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((R/'prediction_freeze.json').read_text('utf8'))
assert all(sha(R/n)==h for n,h in frozen.items())
pred=pd.read_csv(R/'blind_predictions.csv');t=pd.read_csv(R/'tone_support.csv')
rows=[]
for x in pred[pred.arm.str.contains('PCD')].itertuples():
    states=['G2_control'] if x.arm.startswith('G2') else ['G1','G2','G3','G4','C1','C2']
    z=t[(t.point==x.point)&(t.kernel=='PCD')&(t.block==f'R{x.run}_first')&t.state.isin(states)]
    failed=z[z.valid_tones<15]
    rows.append(dict(point=x.point,run=x.run,arm=x.arm,numerical_output=np.isfinite(x.prediction),
      primary_cause='insufficient_single_positive_change_tones' if len(failed) else 'none',
      failed_states='/'.join(failed.state),valid_tones_by_state=json.dumps(dict(zip(z.state,z.valid_tones))),
      downstream_angle_checks='not evaluable: no spectrum' if not np.isfinite(x.prediction) else 'see frozen checks',
      empty_direction_excluded=False if not np.isfinite(x.prediction) else 'see checks'))
pd.DataFrame(rows).to_csv(R/'missing_output_attribution.csv',index=False,encoding='utf-8-sig')
st=['G1','G2','G3','G4','C1','C2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(11.5,4),sharey=True)
for ax,block,title in zip(axs,['R1_held','R2_first','R3_full'],['Held R1 empty','R2 first-half occupied','R3 empty return']):
    z=t[(t.kernel=='PCD')&(t.block==block)&t.state.isin(st)].pivot(index='point',columns='state',values='q2').loc[list('ABCDE'),st]
    im=ax.imshow(z,vmin=0,vmax=30,cmap='YlOrRd',aspect='auto')
    ax.set(xticks=range(6),xticklabels=st,yticks=range(5),yticklabels=list('ABCDE'),title=title)
    for i in range(5):
        for j in range(6):ax.text(j,i,int(z.iloc[i,j]),ha='center',va='center',fontsize=9,color='white' if z.iloc[i,j]>22 else 'black')
fig.subplots_adjust(bottom=.20,right=.9,wspace=.15)
fig.colorbar(im,cax=fig.add_axes([.92,.22,.015,.64]),label='Two-positive-mode tones / 30')
fig.text(.45,.03,'Same R1-derived floor; two positive modes do not mean two human paths.',ha='center',fontsize=9)
fig.savefig(R/'rank_failure_controls.png',dpi=160,bbox_inches='tight');fig.savefig(R/'rank_failure_controls.pdf',bbox_inches='tight');plt.close(fig)

summary='''**本轮没有证明十二天线非相干融合稳定优于 G2。** 新 PCD 主方案在五点都因单正变化模态的有效频点不足而不输出；不是得到五个很差的角度，更不能把它称为人体未检出率。统一 SIGNED-R1 的五点首段平均候选偏差为19.13°，比旧 WBG-R1 的25.53°低，但首段选峰与旧 SIGNED完全相同，不能算新增权重的收益。相对 G2 BG的22.25°，它只改善D，A/B/C/E均更差；平均值下降主要由G2在D首段的错误分支造成。

**B的主要证据是跨run参考变化叠加共同方向偏置。** B前后空场的协方差变化为R1内部波动的2.46–5.55倍；有人变化仅为空场返回变化的1.00–1.59倍。多个子阵同时偏向0°附近，降低G1权重不足以解决。SIGNED-R1在B从前半段−2°跳到后半段−74°；不能用更接近真值的那一段替代预测。

**C的主要证据是相互矛盾的空间/频段分支。** G1/G2支持正角，G3支持负角，G4/C1接近0°；C2的+37°与−25°是56mm间距的近等价栅瓣分支。SIGNED-R1把融合首峰从旧WBG-R1的−24°改为+6°，候选偏差由46.83°降至16.83°，仍明显差于G2 BG的+21°/1.83°，并未解决半频段和双解问题。

**为什么PCD没有输出：** 六态有人首段638/900个频点出现两个超过训练底的正变化模态，G2控制为123/150；但R3空场也有603/900，R1留出空场有256/900。因此当前“单正变化模态+R1两块训练底”组合不适用，不能推断实测有两个等强人体径或原数据不可用。两个R1半块与共享参考只给出一个半差尺度；噪声/背景方差和逐包归一化效应可能使次模态也超过很低的底。此轮保留失败，不看真值后放宽门槛。

后续最有依据的方向是**跨轮噪声/背景尺度建模**，先在空场对空场上检验，而非继续微调角度权重。若使用R3建立双空场模型，需要另行固定R3训练/留出划分，并给G2相同信息预算；不能将同一R3既用于拟合，又当作完全未见的验证。固定支路响应仍需独立已知源验证。这是下一步研究方案，本轮没有宣称已经完成。
'''
report=R/'统一改进验证报告.md';text=report.read_text('utf8');title,body=text.split('\n',1)
text=title+'\n\n'+summary+'\n'+body
text+='''

## PCD 缺失输出的正确解释

原通用评估器对NaN会同时返回nonedge/time/frequency/unique未通过，这不等于观测到了边界峰、时间跳变或双峰。PCD实际第一失败原因为有效单正模态频点不足，尚未形成谱，后续角度检查应读作“无法检验”。`missing_output_attribution.csv`给出逐点、逐轮、逐方案的明确归因；冻结原始CSV保持不变。R3同方向标记在无输出时为false也不表示背景方向已被排除。

PCD代数、门槛与输出归因的独立复核见`PCD_failure_review.md`；映射和参考残余审计见`diagnostics/`。本轮停止后没有对照真值调阈值、补选首峰或剔除失败站位。

![有人和空场中的双正模态门槛失败](rank_failure_controls.png)
'''
report.write_text(text,encoding='utf8')
v=json.loads((R/'verification.json').read_text('utf8'));v.update(new_arms=6,previous_arms_retained=10,no_new_parameters_after_scoring=True,PCD_missing_outputs_explained=True,numerical_NA_not_detection_misses=True,old_frozen_outputs_preserved=True)
(R/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
print('Frozen predictions verified; missing-output attribution and evidence interpretation saved.')
