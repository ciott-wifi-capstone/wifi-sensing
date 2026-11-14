"""Join surveyed geometry only after all prediction registries verify."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import validate_coherent as c
ROOT=Path(__file__).resolve().parent
POINTS={'A':(.80,.60),'B':(2.80,1.20),'C':(.80,-1.20),'D':(2.40,-1.60),'E':(1.64,-2.01)}
TRUTH={k:float(np.degrees(np.arctan2(-v[1],3.65-v[0]))) for k,v in POINTS.items()}

def md(df):return df.to_markdown(index=False,floatfmt='.2f')

def score(name):
    d=pd.read_csv(ROOT/name)
    d['truth_rx_deg']=d.point.map(TRUTH)
    d['error_deg']=abs(d.prediction-d.truth_rx_deg)
    if 'all_R2_peak' in d:d['all_R2_error_deg']=abs(d.all_R2_peak-d.truth_rx_deg)
    d.to_csv(ROOT/name.replace('blind_','scored_'),index=False,encoding='utf-8-sig')
    return d

def audit():
    for name in ('prediction_freeze.json','refinement_freeze.json','background_span_freeze.json','affine_freeze.json'):
        z=json.loads((ROOT/name).read_text())
        assert all(c.base.sha(ROOT/k)==v for k,v in z.items()),name
    rows=[];cross=[]
    for site in POINTS:
        p=pd.read_csv(ROOT/f'{site}_packets.csv')
        assert (p.groupby(['run','arm']).size()==864).all()
        assert not p.duplicated(['run','arm','packet']).any()
        cc=pd.read_csv(ROOT/f'{site}_closure.csv')
        packet=pd.read_csv(ROOT/f'{site}_packet_reference.csv')
        states=packet.groupby(['run','state']).pass_common.mean()
        for r in (1,2,3):
            q=cc[(cc.run==r)&(cc.tree=='tree6')&(cc.part=='all')]
            pv=packet[packet.run==r]
            cyc=pd.read_csv(ROOT/f'{site}_cycles.csv')
            rows.append(dict(point=site,run=r,all_tree_edges_pass_fraction=q.all_tree_edges_pass.mean(),
                closure_pass_count=int(q.pass_closure.sum()),cycles=len(q),closure_pass_fraction=q.pass_closure.mean(),
                closure_tone_p90_median_deg=q.closure_phase_tone_p90_deg.median(),
                packet_common_pass_fraction=pv.pass_common.mean(),minimum_state_packet_pass=states.loc[r].min(),
                representative_span_ms=cyc[cyc.run==r].within_cycle_span_ms.median()))
        a=np.load(ROOT/f'{site}_assembled.npz')['snapshots']
        cov=np.load(ROOT/f'{site}_complex_covariances.npz')
        for tree in ('tree6','tree8'):
            rr=cov[tree]
            assert np.max(abs(rr-rr.conj().swapaxes(-1,-2)))<1e-8
            assert np.linalg.eigvalsh(rr).min()>-1e-7
        for r in range(3):
            for variant,v1,v2 in [('tree6_vs_tree8',a[r,:,0,0],a[r,:,1,0]),
                                  ('early_vs_late',a[r,:,0,1],a[r,:,0,2])]:
                x=c.cov_vectors(v1);y=c.cov_vectors(v2)
                ds=np.linalg.norm(x-y,axis=(-2,-1))/np.maximum((np.linalg.norm(x,axis=(-2,-1))+np.linalg.norm(y,axis=(-2,-1)))/2,1e-15)
                cross.append(dict(point=site,run=r+1,test=variant,covariance_distance_median=float(np.median(ds)),
                                  covariance_distance_p90=float(np.quantile(ds,.9))))
        rr=cov['tree6'][1].mean(0)
        mask=np.ones((12,12),bool)
        for i in range(4):mask[i*3:(i+1)*3,i*3:(i+1)*3]=False
        cross.append(dict(point=site,run=2,test='actual_cross_G_blocks',cross_block_energy_fraction=float(np.sum(abs(rr[:,mask])**2)/np.sum(abs(rr)**2))))
    c.save('assembly_validation.csv',rows);c.save('assembly_sensitivity.csv',cross)
    # Phase identifiability at one tone, without pretending multipath is known LoS.
    designs=[]
    for state,s in enumerate(c.ST):
        for chain,ant in enumerate(c.base.MAP[s]):
            row=np.zeros(21);row[ant-1]=1;row[12+chain]=1;row[15+state]=1
            designs.append(row)
    A=np.array(designs);rank=int(np.linalg.matrix_rank(A))
    j=dict(observations=18,unknown_channel_phases=12,unknown_chain_phases=3,unknown_state_phases=6,
           matrix_rank=rank,nullity=int(21-rank),meaning='Contains a common gauge and channel/hardware ambiguity; same physical antenna never changes its receive chain.',
           source_validation='Same-direction empty-room repetition alone cannot isolate an individual propagation path.')
    (ROOT/'identifiability.json').write_text(json.dumps(j,indent=2),encoding='utf8')
    return pd.DataFrame(rows),pd.DataFrame(cross)

def summary(d,keys,name):
    out=[]
    for key,g in d.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        out.append(dict(zip(keys,key),points=len(g),available=int(g.prediction.notna().sum()),
            provisional_mae=float(g.error_deg.mean()),median=float(g.error_deg.median()),max_error=float(g.error_deg.max()),
            within8=int((g.error_deg<=8).sum()),spectral_gate_count=int(g.spectral_gate.sum()),physical_release=0,
            all_R2_mae=float(g.all_R2_error_deg.mean()) if 'all_R2_error_deg' in g else np.nan))
    z=pd.DataFrame(out);z.to_csv(ROOT/name,index=False,encoding='utf-8-sig');return z

def main():
    av,sen=audit()
    music=score('blind_music_predictions.csv');refine=score('blind_refinement.csv')
    span=score('blind_background_span.csv');aff=score('blind_affine.csv')
    sm=summary(music,['arm','calibration','smoothing','rank'],'music_method_summary.csv')
    sr=summary(refine,['arm','radius','weighting'],'refinement_method_summary.csv')
    sb=summary(span,['arm','calibration','background'],'background_method_summary.csv')
    sa=summary(aff,['arm','calibration'],'affine_method_summary.csv')
    primary=music[(music.calibration=='raw_phase')&(music.smoothing=='none')&(music['rank']==1)]
    tables=[]
    for site in POINTS:
        g=primary[(primary.point==site)&(primary.arm=='G2')].iloc[0]
        t=primary[(primary.point==site)&(primary.arm=='tree6')].iloc[0]
        h=primary[(primary.point==site)&(primary.arm=='tree8')].iloc[0]
        emp=music[(music.point==site)&(music.arm=='tree6')&(music.calibration=='R1_empirical')&(music.smoothing=='none')&(music['rank']==1)].iloc[0]
        cal=aff[(aff.point==site)&(aff.arm=='tree6')].iloc[0]
        tables.append(dict(point=site,truth=TRUTH[site],G2_first_half=g.prediction,G2_error=g.error_deg,
            twelve_first_half=t.prediction,twelve_error=t.error_deg,twelve_held=t.held_R2_peak,
            twelve_held_low=t.held_low,twelve_held_high=t.held_high,
            G2_all_R2=g.all_R2_peak,G2_all_error=g.all_R2_error_deg,twelve_all_R2=t.all_R2_peak,
            twelve_all_error=t.all_R2_error_deg,alternative_tree=h.prediction,
            empirical12=emp.prediction,empirical_error=emp.error_deg,affine12=cal.prediction,affine_error=cal.error_deg))
    tab=pd.DataFrame(tables);c.save('five_point_coherent_comparison.csv',tables)
    refs=pd.concat([pd.read_csv(ROOT/f'{s}_reference_stability.csv') for s in POINTS],ignore_index=True)
    c.save('all_reference_stability.csv',refs.to_dict('records'))
    r3=refs[(refs.arm=='tree6')&(refs.empty_run==3)]
    r1=refs[(refs.arm=='tree6')&(refs.empty_run==1)]
    calchecks=pd.read_csv(ROOT/'affine_reference_checks.csv')
    # Add a precise list of physical elements actually used by the weighted refinement.
    import sys
    sys.path.insert(0,str(ROOT.parent/'short_long_branch_20261106'))
    import predict_branches as old
    eff=[]
    for row in refine.itertuples():
        w=list(map(float,row.weights.split('/')))
        ants=sorted({a for wt,s in zip(w,old.ARMS[row.arm]) if wt>0 for a in c.base.MAP[s]})
        eff.append(dict(point=row.point,arm=row.arm,radius=row.radius,weighting=row.weighting,effective_elements=len(ants),physical_elements='/'.join(map(str,ants))))
    c.save('refinement_effective_elements.csv',eff)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    x=np.arange(5)
    for r,ofs,color in [(1,-.24,'#4885bd'),(2,0,'#df7e32'),(3,.24,'#3e9a78')]:
        z=av[av.run==r].set_index('point').loc[list(POINTS)]
        ax[0].bar(x+ofs,z.closure_pass_fraction*100,.23,label=f'Run {r}',color=color)
    ax[0].axhline(90,ls='--',c='gray');ax[0].set(xticks=x,xticklabels=list(POINTS),ylim=(0,110),ylabel='通过周期比例 / %',title='共享天线 8 的留出闭合检验');ax[0].legend(fontsize=8)
    ax[1].bar(x-.18,r1.set_index('point').loc[list(POINTS)].phase_pair_p90_deg,.35,label='R1 后段',color='#4885bd')
    ax[1].bar(x+.18,r3.set_index('point').loc[list(POINTS)].phase_pair_p90_deg,.35,label='R3 后空场',color='#df7e32')
    ax[1].axhline(15,ls='--',c='gray',label='15°诊断门槛');ax[1].set(xticks=x,xticklabels=list(POINTS),ylabel='阵元对相位变化 P90 / °',title='前空场参考能否保持');ax[1].legend(fontsize=8)
    ax[2].plot(x,tab.truth,'k*',ms=11,label='站位理论角')
    ax[2].plot(x,tab.G2_first_half,'o-',label='G2 前半段暂定')
    ax[2].plot(x,tab.twelve_first_half,'s-',label='12 阵元前半段暂定')
    ax[2].plot(x,tab.twelve_held,'x--',label='12 阵元后半段复核')
    ax[2].set(xticks=x,xticklabels=list(POINTS),ylabel='RX 中心角 / °',title='原相位分支，尚未物理标定');ax[2].legend(fontsize=8)
    fig.suptitle('十二阵元相干重建验证：共享约束、参考保持与候选角度',fontsize=14)
    fig.savefig(ROOT/'十二阵元相干验证.png',dpi=180);fig.savefig(ROOT/'十二阵元相干验证.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for site,axis in zip(POINTS,axes.flat):
        z=np.load(ROOT/f'{site}_music_profiles.npz')
        for arm,style,label in [('G2','-','G2'),('tree6','-','12 elements'),('tree8','--','alternative bridge')]:
            v=z[f'{site}_{arm}_raw_phase_full_q1_predict'][0]
            if np.isfinite(v).any():
                sp=-10*np.log10(np.maximum(v,1e-12));sp-=np.nanmax(sp)
                axis.plot(c.ANG,sp,style,label=label)
        axis.axvline(TRUTH[site],c='black',ls=':',label='survey (evaluation only)')
        axis.set(xlim=(-90,90),ylim=(-15,1),title=f'{site}: uncalibrated diagnostic',xlabel='Angle / deg',ylabel='Normalized MUSIC / dB');axis.grid(alpha=.2)
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=8)
    axes.flat[-1].text(0,.95,'24 switched cycles per run\n864 packets per arm/run\nR1 first 8: reference\nR2 first 12: prediction\nR2 last 12: temporal check\nR3: empty control\n\nNo human-geometry peak selection.\nNo independent hardware calibration.',va='top',fontsize=11)
    fig.savefig(ROOT/'五点完整阵列MUSIC诊断谱.png',dpi=170);fig.savefig(ROOT/'五点完整阵列MUSIC诊断谱.pdf');plt.close(fig)
    primary_summary=sm[(sm.arm.isin(['G2','tree6']))&(sm.calibration=='raw_phase')&(sm.smoothing=='none')&(sm['rank']==1)]
    selected_refinement=refine[(refine.arm=='short_long_12')&(refine.radius==9)&(refine.weighting=='reliability')]
    text=f'''# 十二阵元相干 MUSIC 与局部细化验证

日期 2026-11-06。**已实现真实的十二阵元复数拼接和 12×12 MUSIC 协方差，并完成现有 A–E 的有界验证。共享天线约束大部分可用，但独立空场参考保持未通过，尚未完成物理校准的人体 AoA 验证，也未证明五点整体精度收益。**

## 1. 数据和公平口径

只使用 New A、Real New B/C/D、New E。每点三轮空→人→空。TX=(0,0)、RX=(3.65,0)，E=(1.64,−2.01)。12 阵元间距 28 mm，六状态映射、接收链重排和 READY 事件均复核。

完整阵列试验每方案、每轮 **864 个不同包**。12 方案为 24 个完整周期，每状态每周期 6 包，C1/C2 开销已包含。G2 为同一 10 秒内均匀分布的 864 包。12 方案每周期平均成一个非同时虚拟快照，G2 保留包内三链空间协方差，因此二者原始包数相同，协方差样本构造不同，不能声称独立快照数相同。R1 前8个所选周期/包块训练，余16块留出。R2 前12块预测，后12块检查。R3 为主试验的负对照。各站位一次三轮记录，所有结果均为已反复查看数据上的回顾性研究。

此次与旧的 6.81°/30.41° 试验采用不同预算和严格时间划分。两项旧值保持不变，不能用此次结果替换，也不能交叉比较后把差值归因于天线数。

## 2. 已实际实现的相干处理

1. 在一次状态停留内，利用复数条件最大似然估计同包公共相位和频率线性相位。所有链施加相同修正。前两链、20 个频点拟合，未参与拟合的链/频点检查。独立链拟合仅作模型诊断。
2. 用共享天线构造树，以 C1 为根，G1/1、G2/6、G3/8、C2/6、G4/10 连接。每边只拟合公共相位、频率斜率和正实数幅度。共享天线8的 C1–C2 观测留出作闭合检验。备用树用8连接、用6检查。每条边20频点拟合、10频点预测。
3. 将 G1–G4 的复数代表向量拼成12维数据。其12×12外积保留跨G非对角块，**确实计算了完整矩阵，未用谱相乘冒充相干阵列**。同周期观测横跨约 {av.representative_span_ms.median():.1f} ms，仍需要该时间内静态和公共模型成立，未制造同时观测。
4. 构建前空场背景投影 B，目标协方差为 B R B，导向也使用 B a。主分支保留一个目标信号维度，另列 q=2 和8阵元前后向空间平滑敏感性。平滑仍使用全部12根组成的五个重叠子阵，但有效平滑孔径为8根，不能写成未损失孔径。
5. 保留原相位、R1逐频点经验相位、R1频率仿射相位三种校准假设，全部报告。经验相位假设先补偿 TX 到每根天线的真实球面传播相位，再估计残余响应；其前提是已提取直达波，而这个前提目前未独立证实。

主要公式

`tau_hat = argmax_tau |sum_mk conj(x_mk) h_ref,mk exp(-j 2pi f_k tau)|`

`z_g(k) = a_g exp(j[phi_g + 2pi f_k tau_g]) x_g(k)`

`z_12 = [z_G1; z_G2; z_G3; z_G4], R_k = mean_c z_12,c(k) z_12,c(k)^H`

`B_k = I - U_bg,k U_bg,k^H, P(theta) = sum_k ||B_k a_k(theta)||^2 / sum_k ||U_n,k^H B_k a_k(theta)||^2`

单包公共修正保持同频链间相位。组间修正通过共享物理天线约束，不把各原始天线强制拉成相同相位。条件最大似然改造参考 [Ratnam Eq.27](https://arxiv.org/html/2307.12126v2#S4.SS4)。组内/组间校准顺序参考 [WiCAL II-B](https://arxiv.org/html/2505.21408v1#S2.SS2)。本实现是线阵、公共定时与留出检验的适配，不复称原论文算法或性能。

## 3. 共享相位能否拼接

{md(av[['point','run','closure_pass_count','cycles','closure_tone_p90_median_deg','packet_common_pass_fraction','minimum_state_packet_pass']])}

closure_pass_count 为共享天线8留出的完整周期通过数。闭合包含相位、相关度、相对斜率和幅度检查。大多数约束通过，说明未发现普遍的随机组间相位破坏。它不检验已知人体方向，也不能独立验证树叶 G1、G4 的所有连接。每状态公共包模型通过率见 `*_packet_reference.csv`，不能只用整体均值掩盖较差状态。

合成无噪声复数12维恢复相对误差 {json.loads((ROOT/'synthetic_checks.json').read_text())['complex_12_recovery_relative_error']:.2e}。故意破坏共享天线或未拟合链的测试被识别。实际协方差均 Hermitian、半正定且跨G块非零。这些证明实现和约束检查确实工作，不等于实测角度精度。

## 4. 冻结前空场参考能否保持

{md(r3[['point','phase_pair_p90_deg','eigenvector_corr','pass_reference']])}

该值是在每个频点消除特征向量公共相位自由度后，统计66个阵元对空间相位变化的 P90，不是 AoA 误差。**五个后空场均未通过15°参考保持门槛。** B/C/E 在 R1 内部后段也未通过。没有把这些差异直接解释成网卡重启、硬件变化或人体反射，当前检查只能确认空间参考变化，不能独自确定其来源。

同一共享天线始终处于同一接收链。单频18条观测中，若同时估计12个未知真实信道相位、3个链响应和6个状态相位，设计矩阵秩为17、零空间为4维。共享约束不能把固定链响应与未知信道唯一拆开。允许每个天线支路独立响应时歧义更大。前空场已知无人不代表已分离纯直达波。

## 5. 五点结果与收益

以下是未完成硬件标定的原相位主分支，统一按 RX 中心理论角计分。前半段预测在看后半段结果前固定。

{md(tab[['point','truth','G2_first_half','G2_error','twelve_first_half','twelve_error','twelve_held','twelve_held_low','twelve_held_high']])}

{md(primary_summary[['arm','provisional_mae','median','max_error','within8','spectral_gate_count','all_R2_mae']])}

**spectral_gate 只检查谱的时间/频率/负对照现象，不包含物理硬件校准；通过也不能自动标为人体方向。** 例如 A/B/C 存在稳定但偏离人体的原相位谱峰。最终物理单解均未放行。

同时列出整轮描述性峰，防止早段误支使十二根看起来不公平地占优。

{md(tab[['point','G2_all_R2','G2_all_error','twelve_all_R2','twelve_all_error','alternative_tree']])}

E 的十二阵元候选为 +43°，前后半段同为 +43°，留出低/高频段为 +45°/+42°。它显示方向稳定性的潜力，但当前单空场分支的负对照/参考条件仍不齐备。D 的十二阵元 +35° 比 G2 早段的错误分支稳定，但整轮 G2 方向更接近真值，因此不能以早段对照单独宣布 D 精度获益。A/B/C 的错误方向限制了全体收益。

## 6. 初始相位校准与局部细化是否改善

{md(tab[['point','empirical12','empirical_error','affine12','affine_error']])}

逐频点经验校准并没有提供全体改进。仿射校准只用20频点拟合相位常数与斜率，并保留10频点和后空场验证，详见 `affine_reference_checks.csv`。即使拟合通过，也只能说明该响应近似可由仿射函数解释，不能排除多径混入校准。

另复用旧同预算数据，按 R2 Q1/Q2 确定局部可靠权重，再在 Q3/Q4 与 R3 检查。以下统一采用 ±9° 局部窗，未逐点选择方法。

{md(selected_refinement[['point','seed','prediction','error_deg','active_roles','long_roles_weight','spectral_gate']])}

候选边界、权重、后半段峰及 ±18°敏感性均保留。A 的额外谱检查可通过，但只启用了部分状态；有效物理阵元数单列，不能据此称为完整十二根相干增益。B/D 早段候选出现换支，局部窗无法保证包含人体方向。此次对时间划分的敏感性进一步说明旧整轮接近真值不能代替可重复预测。

## 7. 多背景子空间的结构性结果

追加试验用 R1 前8块和 R3 前12块建立两空场子空间，R1后16块与R3后12块保持留出。它是离线双空场方法。背景取两维时，三阵元若再保留一维人体信号，已没有独立噪声维度供 MUSIC 判别；十二阵元仍有9维噪声空间。

{md(span[(span.arm.isin(['G2','tree6']))&(span.calibration=='raw_phase')&(span.background=='two_empty_span')][['point','arm','prediction','error_deg','valid_noise_dimension_tones','spectral_gate']])}

这证实了多阵元在本实现里能够承受更高维的背景抑制；**可继续计算的自由度收益尚未转化为经过物理验证的正确人体方向**。G2 不可用的点保留 NA，未从总体中悄悄删除，也不拿有限输出的均值比较完整五点。

## 8. 当前结论与最小补充

- 实现层面已完成12维复数重建、全矩阵MUSIC、前后向平滑敏感性、两棵共享树、半频段/半时间及负对照检查。原始和失败结果完整保留。
- 数据层面可支持共享状态相对同步的大部分约束、E 的候选稳定性和更高维背景抑制的可用自由度。尚不支持五点整体AoA精度增益。
- 完整物理验证未通过的关键位置是固定响应与空场参考保持，不能用调峰、按站位选校准、修改符号或放宽门槛弥补。
- 最小补采是在同一次开机、接线、阵列位置与通道下，先取得可验证直达波占优的已知源校准，保留准确TX到各阵元距离，再至少用一个**未参与拟合的侧向已知源方向**检查12阵元流形。正前方源本身无法给出全角度泛化证据。测试时TX回到固定位置、静置，连续空→人→空重复完整批次。每次移动TX可能牵动线缆，要固定RX线缆并保存动作日志。已知源标定与静止人体检测分别验收。
- 若可调整切换表，增加能独立检查 G1/G4 叶节点的重叠状态，并保留 C1/C2。新状态须先验证开关物理可达性。新会话校准不能直接用于这些旧记录。
- 若上述标定和空场留出通过，再冻结全部规则，至少采独立重复批次，按相同总包数/时长比较 G2、四G非相干、12相干及稀疏三阵元长基线对照。报告全部点、输出率、错误放行和各分支结果。

## 文件与复现

`validate_coherent.py` 主验证；`validate_refinement.py` 局部细化；`validate_background_span.py` 双空场背景；`validate_affine_reference.py` 仿射参考；`score_and_report.py` 独立评价。预测代码不读取人体坐标，四份freeze文件先于评分。各点 `*_sources.csv`、`*_packets.csv` 追溯源文件哈希和原始包序号。`five_point_coherent_comparison.csv` 为主表，所有配置和失败见 `scored_*.csv` 与 `*_method_summary.csv`。

运行环境 `python -B -X utf8`，BLAS单线程。运行顺序按上面列出的脚本执行。前四脚本有独立缓存，旧目录、原始数据、论文和Overleaf均未改动。
'''
    (ROOT/'十二阵元相干MUSIC验证报告.md').write_text(text,encoding='utf8')
    print('PRIMARY',primary_summary[['arm','provisional_mae','all_R2_mae']].to_string(index=False))
    print('AFFINE',sa[sa.arm!='tree8'].to_string(index=False))
    print('REPORT READY',flush=True)

if __name__=='__main__':main()
