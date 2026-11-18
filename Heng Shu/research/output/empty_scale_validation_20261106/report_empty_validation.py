"""Summarize frozen empty validation and gate the occupied branch. No R2 estimation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen=json.loads((R/'empty_stage_freeze.json').read_text('utf8'))
assert all(sha(R/n)==v for n,v in frozen.items())
g=pd.read_csv(R/'state_model_gate.csv');c=pd.read_csv(R/'held_empty_checks.csv');p=pd.read_csv(R/'model_parameters.csv')
e=pd.read_csv(R/'human_branch_eligibility.csv')
assert not e.empty_model_pass.any(), 'A qualified model exists: a separate frozen occupied runner is required before reporting.'
# Reuse earlier frozen numerical results only as labeled historical baselines.
old=R.parent/'noncoherent_bc_followup_20261106'
prior=pd.read_csv(old/'all_scored_predictions.csv');prior=prior[prior.run==2]
rows=[]
for x in e.itertuples():
    baseline='G2_BG' if x.arm=='G2' else 'SIX_WBG_R1'
    b=prior[(prior.point==x.point)&(prior.arm==baseline)].iloc[0]
    rows.append(dict(point=x.point,arm=x.arm,model=x.model,empty_model_pass=x.empty_model_pass,
      updated_prediction=np.nan,updated_geometric_error=np.nan,updated_numerical_outputs=0,occupied_estimation_executed=False,
      status='blocked_by_empty_model',passed_states=x.passed_states,required_states=x.required_states,
      historical_baseline=baseline,historical_prediction=b.prediction,historical_error=b.error,
      historical_packet_budget=864,historical_prediction_packets=432,physical_release=False))
pd.DataFrame(rows).to_csv(R/'five_point_conditional_comparison.csv',index=False,encoding='utf-8-sig')
summary=[]
for model in ['NSI','NSD']:
    d=g[g.model==model];checks=c[c.model==model]
    for cohort,states in [('six_states',d.state!='G2_control'),('G2_control',d.state=='G2_control')]:
        dd=d[states];cc=checks[checks.state.isin(dd.state.unique())]
        summary.append(dict(model=model,cohort=cohort,states=len(dd),valid_parameter_fits=int(dd.fit_valid.sum()),
            states_passed=int(dd.pass_model.sum()),envelope_only_passed=int(dd.envelope_failures.eq(0).sum()),
            checks=len(cc),envelope_pass=int(cc.gate_envelope.sum()),
            raw_crossrun_p90_over_envelope=float((cc[cc.block.str.startswith('R3')].raw_p90/cc[cc.block.str.startswith('R3')].envelope_p90).median()),
            corrected_crossrun_p90_over_envelope=float((cc[cc.block.str.startswith('R3')].corrected_p90/cc[cc.block.str.startswith('R3')].envelope_p90).median())))
pd.DataFrame(summary).to_csv(R/'model_summary.csv',index=False,encoding='utf-8-sig')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
def figsave(n):
    plt.savefig(R/(n+'.png'),dpi=160,bbox_inches='tight');plt.savefig(R/(n+'.pdf'),bbox_inches='tight');plt.close()
st=['G1','G2','G3','G4','C1','C2','G2_control']
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,model in zip(axs,['NSI','NSD']):
    mat=g[g.model==model].pivot(index='point',columns='state',values='envelope_failures').loc[list('ABCDE'),st]
    im=ax.imshow(mat,vmin=0,vmax=18,cmap='YlOrRd',aspect='auto')
    ax.set(title=model+': empty envelope failures / 18',xticks=range(7),xticklabels=['G1','G2','G3','G4','C1','C2','G2 ctrl'],yticks=range(5),yticklabels=list('ABCDE'))
    for i in range(5):
        for j in range(7):ax.text(j,i,int(mat.iloc[i,j]),ha='center',va='center',color='white' if mat.iloc[i,j]>12 else 'black')
fig.subplots_adjust(right=.90,bottom=.16,wspace=.2);fig.colorbar(im,cax=fig.add_axes([.92,.18,.015,.65]),label='Failed block/band checks')
fig.text(.45,.01,'Same rules at every site; no state passes all empty checks.',ha='center');figsave('empty_gate_matrix')

diag=pd.read_csv(R/'diagnostics/empty_model_predictions.csv')
models=['unchanged','per_tone_beta','shared_beta','affine','full_anchor']
# Use actual recorded model labels rather than guessing output schemas.
actual=diag.model.unique().tolist()
print('Diagnostic models:',actual)
names={'unchanged':'No correction','projected_scalar_per_tone':'Per-tone scalar','projected_scalar_shared_frequency':'Shared scalar','affine_aR1_bI_shared_frequency':'Gain + white floor','full_empty_anchor_per_tone':'Full R3-dev anchor'}
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,rep in zip(axs,['unit_packet','fixed_R1_scale_raw']):
    z=diag[(diag.representation==rep)&(diag.window=='R3_held')&(diag.frequency_split=='held_odd')&(diag.state!='G2_control')]
    for model in actual:
        values=z[z.model==model].groupby('point').projected_residual_over_R1held.median().reindex(list('ABCDE'))
        ax.plot(range(5),values,'o-',label=names.get(model,model),lw=1)
    ax.axhline(1,color='black',ls='--',lw=1)
    ax.set(title='Per-packet normalization' if rep=='unit_packet' else 'One fixed R1 scale',xticks=range(5),xticklabels=list('ABCDE'),xlabel='Acquisition batch');ax.grid(alpha=.15)
axs[0].set_ylabel('Remaining empty drift / within-R1 drift')
axs[1].legend(fontsize=7,loc='upper right');fig.tight_layout();figsave('empty_drift_models')

def line(v):return '| '+' | '.join(map(str,v))+' |'
report=['# 空场跨轮背景与噪声尺度验证','',
'**结论：现有空场没有支持本轮两种噪声尺度校正。PCD 有效频点不足的问题尚未解决。按预先冻结的停止条件，本轮没有把未经验证的校正用于 R2 人体数据，也没有生成新的校正人体角度。** 这不是认定原数据不可用，更不是人体未检出率。','',
'## 1. 使用的数据与公平预算','',
'主分析仅使用 New A、Real New B、Real New C、Real New D、New E 五批的 R1/R3，共十份空场。旧 kong 空场与这些有人数据并非共同标定会话，不能混用作噪声参考。每种阵列每轮仍有864个不同包；六态各144包，G2独立控制864包。R2缓存没有进入模型计算。',
'','两阶段的时间角色如下：R1前8周期/批（288包）拟合参考；中8（288包）校准空场误差范围；后8（288包）检验。R3前8（288包）拟合跨轮参数，后16（576包）检验。全部检查使用连续4周期块；固定20个频点拟合尺度系数，另10个频点检验系数迁移。参考背景仍在每个频点由R1建立，因此并非整个频点的观测从未使用过。数据曾被多次研究，因此这些是内部开发检验，不是新的独立实验。','',
'## 2. 实际测试的模型','',
'设B=I−hhᴴ为R1背景投影，D=B(R−Rref)B；u是当前窗口每包归一化因子90/||CSI||²的均值相对R1的变化。','',
'- **NSI**：D≈u βB。每状态只有一个跨频共享的非负数字方差系数。',
'- **NSD（预先指定主模型）**：D≈u B diag(β1,β2,β3)B。允许三条链具有不同噪声尺度，但同一系数跨频共享。',
'- 两者均只用空场开发块和训练频点拟合；系数为负时明确拒绝，不裁剪成好看结果。不得从当前目标的最小特征值、trace或人体角度直接确定要扣除的噪声。',
'','这两个模型是投影后随机方差变化的假设，不是测得的物理热噪声。单位能量归一化令全频trace恒90，完整协方差还需要主信号功率补偿；投影模型不能自动解释链间相位或环境方向变化。','',
'## 3. 空场先验收的结果','',
line(['模型','六态通过','G2控制通过','系数可用（35条）','仅看误差范围通过']),line(['---']*5)]
for model in ['NSI','NSD']:
    d=g[g.model==model]
    report.append(line([model,f'{d[d.state!="G2_control"].pass_model.sum()}/30',f'{d[d.state=="G2_control"].pass_model.sum()}/5',f'{d.fit_valid.sum()}/35',f'{d.envelope_failures.eq(0).sum()}/35']))
report += ['',
'误差范围由R1中8周期的两个四周期块固定，未用R3留出结果扩大。每状态要通过六个时间块×全/低/高三组留出频点，共18项检查；要求残余中位数及P90不超校准范围、不劣于不修正，并且超过原PCD底的正变化频点数不增加。这里的范围是开发验收标准，不是95%统计置信界。',
'35条是30个六态记录加5个G2独立包预算控制，不是35个独立实验。时间块和全/半频段检验也不能当作相互独立的统计样本。正变化频点数不增加不等于空场方向误检率已验证：两个正模态变成一个时，频点数可能不变，但候选输出会改变。若未来模型通过，还需实际空场候选检验。',
'','**即使完全不考虑负系数、proxy外推等其他限制，仅看留出残余误差，仍没有任何状态全部通过。** 因此不是单靠一个符号/外推限制卡住输出。原 PCD 特征值底、恰好一个正模态、至少15/30频点、半频段至少8/15频点、3dB歧义和9°一致性条件保持原规则。空场模型误差范围只决定是否允许使用模型，绝不替换PCD特征值底。',
'独立审查逐值复现了105组误差范围，最大数值差9.8×10⁻¹⁷；并检查18,900个逐频矩阵。空场诊断使用固定1e−10数值容差，旧PCD使用按特征值尺度缩放的容差；本批两种写法造成的正模态计数及相关门槛差异均为0。冻结结果没有重写。','',
'## 4. 为什么不能只修一个噪声底','',
'独立诊断另外比较了逐频标量、跨频共享标量、gR1+δI以及完整后空场开发矩阵锚点。这些仅用于判断剩余变化；其中逐频标量和完整锚点只有时间留出，没有真正的新频点验证。','',
'六态30条记录上，逐频标量模型留出残余仍约为原变化的0.721，约为R1自身留出波动的5.03倍；完整后空场开发矩阵作锚点仍有0.531残余、3.38倍R1波动。固定一次R1数字尺度、保留包幅度的对照结果接近，不能单归咎于逐包归一化。',
'','**B：**逐频标量修正后还剩0.724原变化，约8.36倍R1内部波动。**C：**还剩0.691原变化，约3.27倍。说明空场变化具有明显方向性/非各向同性成分，当前资料不能唯一分离环境变化、接收支路响应和随机噪声。',
'','二维纯定向变化D=λvvᴴ本身也有trace(D)/2的标量分量，标量部分天然占一半平方范数。因此“标量拟合解释了一半”不能叫“噪声占一半”，更不能直接扣目标最小特征值来强制rank1。','',
'## 5. 五点与G2的条件比较','',
'下面保留原来相同包数的首段基线数值作为上下文。新校正是否允许进入有人处理，对全部五点使用同一个空场规则；本轮均未允许，新增角度及偏差为NA。没有逐点改选模型，也没有只保留某一状态冒充十二天线。',
'',line(['点','原G2 BG首峰','原12 WBG-R1首峰','NSI：G2/12允许','NSD：G2/12允许','新校正角度']),line(['---']*6)]
for site in 'ABCDE':
    z=prior[prior.point==site].set_index('arm')
    report.append(line([site,f'{z.loc["G2_BG","prediction"]:+.0f}°',f'{z.loc["SIX_WBG_R1","prediction"]:+.0f}°','否 / 否','否 / 否','NA（未进入人体估计）']))
report+=['','旧G2与12WBG-R1的五点平均候选偏差仍为22.25°、25.53°；它们是上轮冻结基线，不是本轮得到的新精度。两者历史完整候选验收都为0/5。当前不能声称噪声校正改善了五点结果。','',
'## 6. 下一步最小条件','',
'1. 优先采集同会话连续空—空—空重复，记录门的位置/开关、人员出入、TX/RX与线缆、网卡/驱动状态；先证明跨run空场可预测。RX附近有门是已知现场因素，但本次数据不能确定门导致了变化。',
'2. 用独立空场开发批次确定噪声和相对响应模型，另留完整批次检验，再进入人体；不把当前R3同时用作拟合和未见验证。',
'3. 若继续研究对角复响应/相位迁移模型，应只用空场或已知源拟合，并检验未用频点与后续空场；另做人体分量保真检查，避免把人体变化一起对齐掉。此分支尚未实现。',
'4. 固定硬件相位和理想导向的对应关系仍需已知源独立验证；不能用人体站位真值反算逐点偏置。','',
'## 7. 文件和复现','',
'- `protocol.json`、`pre_execution_freeze.json`：模型、划分、验收规则及执行前冻结。',
'- `validate_empty_models.py`：只处理run索引0和2，不处理R2。`empty_stage_freeze.json`在人体分支判定前锁定全部输出。',
'- `model_parameters.csv`、`held_empty_checks.csv`、`state_model_gate.csv`：保留全部拟合、逐块失败和拒绝。',
'- `human_branch_eligibility.csv`、`five_point_conditional_comparison.csv`：统一条件分支和历史等包数对照，明确新人体估计未执行。',
'- `diagnostics/空场尺度独立审计.md`：标量/全矩阵锚点、原始数字尺度对照及频率留出边界。',
'- `review/`：独立设计与代码审查。所有原始数据、旧分析和论文保留。','',
'![留出空场门槛失败](empty_gate_matrix.png)','','![跨轮背景残余对照](empty_drift_models.png)']
(R/'空场跨轮模型验证报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
verification=dict(empty_hashes_verified=True,R2_estimation_executed=False,new_human_angles_generated=False,empty_records=10,models=['NSI','NSD'],all_35_states_fail_each_model=True,PCD_eigenvalue_floor_unchanged=True,PCD_acceptance_thresholds_relaxed=False,per_point_best_selection=False,packets_per_arm_run=864,model_calibration_not_physical_noise_truth=True,independent_test_capture=False,raw_and_prior_outputs_preserved=True)
(R/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
print(pd.DataFrame(summary).to_string(index=False))
print('Human branch stopped by frozen empty gates. Baseline tables labeled historical.')
