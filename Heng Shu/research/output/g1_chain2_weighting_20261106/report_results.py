"""Truth joins only after the frozen prediction files verify."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_g1 as a

R=a.ROOT;c=a.c
POINTS={'A':(.8,.6),'B':(2.8,1.2),'C':(.8,-1.2),'D':(2.4,-1.6),'E':(1.64,-2.01)}
TRUTH={s:float(np.degrees(np.arctan2(-y,3.65-x))) for s,(x,y) in POINTS.items()}
NAMES={'G2_original':'G2 三阵元','12_original':'十二阵元原始','12_R1_weighted_scan':'十二阵元仅加权谱匹配',
       '12_R1_weighted_fit_and_scan':'十二阵元加权拟合＋谱匹配',
       '12_fixed_w2_0.5_scan':'十二阵元固定权重0.5','12_fixed_w2_0.25_scan':'十二阵元固定权重0.25',
       '12_R1_weighted_fit_only':'十二阵元仅加权公共拟合','11_drop_antenna2':'十一阵元移除天线2'}

def md(d):return d.to_markdown(index=False,floatfmt='.2f')

def export(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(R/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    freeze=json.loads((R/'prediction_freeze.json').read_text())
    assert all(c.base.sha(R/k)==v for k,v in freeze.items())
    oldfreeze=json.loads((a.OLD/'prediction_freeze.json').read_text())
    assert all(c.base.sha(a.OLD/k)==v for k,v in oldfreeze.items())
    d=pd.read_csv(R/'blind_predictions.csv')
    d['truth_rx_centre_deg']=d.point.map(TRUTH)
    d['absolute_error_deg']=(d.prediction-d.truth_rx_centre_deg).abs()
    d['whole_R2_error_deg']=(d.all_R2_peak-d.truth_rx_centre_deg).abs()
    a.save('scored_predictions.csv',d)
    old=pd.read_csv(a.OLD/'blind_music_predictions.csv')
    for site in POINTS:
        for method,arm in [('G2_original','G2'),('12_original','tree6')]:
            for cal in ('raw_phase','R1_empirical'):
                n=d[(d.point==site)&(d.method==method)&(d.calibration==cal)].iloc[0]
                q=old[(old.point==site)&(old.arm==arm)&(old.calibration==cal)&(old.smoothing=='none')&(old['rank']==1)].iloc[0]
                for k in ['prediction','all_R2_peak','held_R2_peak','held_low','held_high','spectral_gate']:
                    assert n[k]==q[k] or (pd.isna(n[k]) and pd.isna(q[k])),(site,method,cal,k,n[k],q[k])
    summary=[]
    for (cal,method),g in d.groupby(['calibration','method'],sort=False):
        summary.append(dict(calibration=cal,method=method,method_zh=NAMES[method],numerical_outputs=int(g.prediction.notna().sum()),
            provisional_MAE_deg=g.absolute_error_deg.mean(),median_error_deg=g.absolute_error_deg.median(),
            max_error_deg=g.absolute_error_deg.max(),within8_count=int((g.absolute_error_deg<=8).sum()),
            spectral_gate_count=int(g.spectral_gate.sum()),physical_release_count=0,
            whole_R2_descriptive_MAE_deg=g.whole_R2_error_deg.mean()))
    sm=a.save('method_summary.csv',summary)
    raw=d[d.calibration=='raw_phase'];table=[];whole=[]
    for s in POINTS:
        z=raw[raw.point==s].set_index('method')
        row={'站位':s,'理论角':TRUTH[s]};wr={'站位':s,'理论角':TRUTH[s]}
        for key,title in [('G2_original','G2'),('12_original','12原始'),('12_R1_weighted_fit_and_scan','12降权'),('11_drop_antenna2','11移除2')]:
            row[title+'角度']=z.loc[key,'prediction'];row[title+'偏差']=z.loc[key,'absolute_error_deg']
            wr[title+'整轮角度']=z.loc[key,'all_R2_peak'];wr[title+'整轮偏差']=z.loc[key,'whole_R2_error_deg']
        table.append(row);whole.append(wr)
    tb=a.save('五点等包数角度对照.csv',table);wt=a.save('整轮描述性角度对照.csv',whole)
    states=pd.read_csv(R/'pair_phase_held.csv')
    phase=states.groupby(['state','pair'],sort=False)[['phase_abs_median_deg','phase_abs_p90_deg']].median().reset_index()
    a.save('pair_phase_summary.csv',phase)
    power=pd.read_csv(R/'chain_power.csv');tones=pd.read_csv(R/'g1_tone_profiles.csv')
    ch2=power[(power.state=='G1')&(power.chain==2)]
    psummary=[]
    for site in POINTS:
        q=ch2[ch2.point==site]
        psummary.append(dict(point=site,power_share_median_over_runs=q.median_power_share.median(),
            min_run_relative_db=q.relative_other_chains_db.min(),max_run_relative_db=q.relative_other_chains_db.max(),
            zero_fraction_mean=q.exact_zero_fraction.mean(),rail_fraction_max=q.signed8_rail_fraction.max()))
    ps=a.save('g1_power_summary.csv',psummary)
    refs=pd.concat([pd.read_csv(R/f'{s}_reference_stability.csv') for s in POINTS])
    a.save('all_reference_stability.csv',refs)
    fits=pd.concat([pd.read_csv(R/f'{s}_weighted_common_fit.csv') for s in POINTS])
    fitq=fits[fits.packet!=0][['common_delay_change_ns','common_phase_change_deg']].quantile([.5,.9,.99])
    a.save('common_fit_change_summary.csv',fitq.rename_axis('quantile').reset_index())
    weights=pd.read_csv(R/'frozen_R1_weights.csv')
    closure=pd.concat([pd.read_csv(R/f'{s}_weighted_closure.csv') for s in POINTS])
    a.save('weighted_closure_summary.csv',closure.groupby(['point','run']).agg(cycles=('pass_closure','size'),passes=('pass_closure','sum'),edge_passes=('all_tree_edges_pass','sum')).reset_index())
    # Documents contain descriptive, repeated-sample statistics, not pseudo-independent confidence intervals.
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,3,figsize=(13.4,7.5),layout='constrained')
    colors=['#487BB5','#D56832','#459781']
    for ax,s in zip(axs.flat,POINTS):
        for rr,name,color in zip((1,2,3),('Run 1 空场','Run 2 有人','Run 3 空场'),colors):
            q=tones[(tones.point==s)&(tones.run==rr)]
            ax.plot(q.frequency_mhz-5230,q.relative_chain2_db,label=name,color=color,lw=1.8)
        ax.axhline(-6,color='#888888',ls=':',lw=1)
        ax.set(title=f'{s}：G1 第二链',xlabel='相对中心频率 / MHz',ylabel='相对链1、3平均功率 / dB',ylim=(-25,4))
        ax.grid(alpha=.15)
    axs.flat[5].axis('off')
    axs.flat[5].legend(*axs.flat[0].get_legend_handles_labels(),loc='upper left',frameon=False)
    axs.flat[5].text(0,.60,'读图说明\n\n每个频点：逐包功率比的中位数。\n横向虚线：−6 dB，仅作读图参考。\n\n多处整段偏弱，也存在频率选择性变化。\n固定频率/位置下，不能据此区分\n硬件支路损耗与持续传播衰落。\n\n只统计此前冻结的864包/臂/轮中的G1包。',va='top',linespacing=1.65)
    fig.suptitle('G1 第二链：幅度偏弱及频率形状（原始 CSI 数字量，不是接收功率或 SNR）',fontsize=15)
    export(fig,'G1第二链频率响应')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for j,pair in enumerate(('1-2','1-3','2-3')):
        q=phase[phase.pair==pair].set_index('state').loc[c.ST]
        axs[0].bar(np.arange(6)+(j-1)*.24,q.phase_abs_p90_deg,width=.24,label='链 '+pair)
    axs[0].set_xticks(np.arange(6),c.ST);axs[0].set(ylabel='留出包相对链间相位差 P90 / °',title='各状态：15轮统计量的中位数')
    axs[0].legend(frameon=False);axs[0].grid(axis='y',alpha=.15)
    match=pd.read_csv(R/'g1_common_phase_invariant_amplitude_match.csv')
    bins=['[3.0, 6.0)','[6.0, 12.0)','[12.0, 24.0)']
    for chain,col,name in [(2,'#D56832','链2相对链1'),(3,'#459781','链3相对链1')]:
        q=match[match.target_chain==chain].set_index('amplitude_bin').loc[bins]
        axs[1].plot(np.arange(3),q.phase_median_deg,'o-',color=col,label=name,lw=2)
        for i,v in enumerate(q.phase_median_deg):axs[1].annotate(f'{v:.1f}°',(i,v),xytext=(0,7),textcoords='offset points',ha='center')
    axs[1].set_xticks(np.arange(3),['3–6','6–12','12–24']);axs[1].set(xlabel='目标链原始 CSI 幅度（参考与留出包同一档）',ylabel='留出相位差绝对值中位数 / °',title='相近幅度下，链2相位仍较不稳',ylim=(0,30))
    axs[1].legend(frameon=False);axs[1].grid(alpha=.15)
    fig.suptitle('同包共轭相位诊断：无需逐包公共相位或时延对齐',fontsize=15)
    export(fig,'G1第二链相位诊断')
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    methods=['G2_original','12_original','12_R1_weighted_fit_and_scan']
    for j,(method,color) in enumerate(zip(methods,['#487BB5','#D56832','#459781'])):
        q=raw[raw.method==method].set_index('point').loc[list(POINTS)]
        x=np.arange(5)+(j-1)*.22
        axs[0].scatter(x,q.prediction,color=color,s=45,label=NAMES[method],zorder=3)
        axs[1].bar(x,q.absolute_error_deg,width=.21,color=color,label=NAMES[method])
        for xi,ang,err in zip(x,q.prediction,q.absolute_error_deg):
            axs[0].annotate(f'{ang:+.0f}°',(xi,ang),xytext=(0,8 if j!=1 else -16),textcoords='offset points',ha='center',fontsize=9)
            axs[1].annotate(f'{err:.1f}',(xi,err),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    axs[0].scatter(np.arange(5),list(TRUTH.values()),marker='*',s=120,color='#444444',label='RX中心理论角',zorder=2)
    axs[0].legend(loc='lower right',frameon=False,ncol=2)
    axs[0].set(ylabel='角度 / °',ylim=(-100,80),title='前半段固定规则取峰；所有点均未通过物理验收')
    axs[1].set(ylabel='与理论角的数值偏差 / °',ylim=(0,95),title='同包数临时偏差：G2 22.25°；十二阵元原始/降权均27.33°')
    for ax in axs:ax.set_xticks(np.arange(5),list(POINTS));ax.grid(axis='y',alpha=.15)
    fig.suptitle('G1 第二链降权：五点十二阵元首选角度均未改变（1°网格）',fontsize=15)
    export(fig,'五点等包数降权对照')
    fig,axs=plt.subplots(2,3,figsize=(13.4,7.5),layout='constrained')
    for ax,s in zip(axs.flat,POINTS):
        spec=np.load(R/f'{s}_spectra.npz')
        for method,color,style in [('G2_original','#487BB5','-'),('12_original','#D56832','-'),('12_R1_weighted_fit_and_scan','#459781','--'),('11_drop_antenna2','#7556A0',':')]:
            cost=spec[f'{method}_raw_phase_predict'][0]
            db=-10*np.log10(np.maximum(cost,1e-12));db-=np.nanmax(db)
            ax.plot(c.ANG,db,color=color,ls=style,lw=1.5,label=NAMES[method])
        ax.axvline(TRUTH[s],color='#444444',lw=1,ls='--',label='理论角（计算后叠加）')
        ax.set(title=s,xlabel='RX局部角度参数 / °',ylabel='归一化谱 / dB',xlim=(-90,90),ylim=(-12,.8));ax.grid(alpha=.15)
    axs.flat[5].axis('off');axs.flat[5].legend(*axs.flat[0].get_legend_handles_labels(),loc='upper left',frameon=False)
    axs.flat[5].text(0,.46,'每条曲线各自归一化到0 dB，\n不能直接比较曲线间绝对强度。\n\n同一原始包，R1前8周期作背景。\nRun 2前12周期取峰，后12周期复核。\n\n曲线为未完成固定响应标定的\n诊断谱，不等于已识别的人体路径。',va='top',linespacing=1.7)
    fig.suptitle('原始、降权和移除天线2：MUSIC 子空间诊断谱',fontsize=15)
    export(fig,'降权前后MUSIC谱')
    # Values in prose are calculated from the saved result files.
    ph1=phase[(phase.state=='G1')].set_index('pair')
    refs3=refs[(refs.check=='R3')&refs.method.isin(['12_original','12_R1_weighted_fit_and_scan','11_drop_antenna2'])]
    reft=refs3.pivot(index='point',columns='method',values='phase_pair_p90_deg').reset_index()
    descriptive=sm[(sm.calibration=='raw_phase')&sm.method.isin(methods)][['method_zh','whole_R2_descriptive_MAE_deg']]
    primary=sm[(sm.calibration=='raw_phase')][['method_zh','numerical_outputs','provisional_MAE_deg','median_error_deg','max_error_deg','spectral_gate_count','physical_release_count']]
    empirical=sm[sm.calibration=='R1_empirical'][['method_zh','provisional_MAE_deg','spectral_gate_count','physical_release_count']]
    report=f'''# G1 第二链排查与降权对照

更新：2026-11-06。对象为最新 A–E 的空→人→空配对批次。仅复用现有数据，原始文件、旧结果、论文与 Overleaf 均未修改。

## 结论

1. **G1 第二链对应天线2的数据确实低可靠**：既偏弱，又存在明显的链间相位波动。状态日志、链重排、文件哈希和冻结样本均复核通过，未发现这次结果由缓存错配或明显时间窗错误造成。
2. **不能从这些记录把原因唯一判成“天线2损坏”或“传播深衰落”**。其余状态使用同一接收链2时正常；疑点在天线2对应的天线/馈线/开关支路及该位置传播条件。始终固定天线位置和连接，二者在现有数据中不可分离。相近幅度下仍有额外相位不稳，因此“仅数字幅度小或量化粗”也不是完整解释。
3. **本轮降权未改善十二阵元首选角度**。R1确定权重后，在1°网格上 A–E 仍为 −29°、+2°、−21°、+35°、+43°。仅加权、加权公共拟合、固定权重0.5/0.25乃至移除天线2均未改变主线五点首选角度。它不是现有整体方向偏差的唯一原因。
4. 十二阵元原始与降权的临时平均偏差均为 **27.33°**；同包数G2为 **22.25°**。这不是通过验收的人体AoA准确率。所有点的物理验收仍为 **0/5**；不报告位置精度。

## 输入与公平性

- 原始来源：A=New A；B=Real New B；C=Real New C；D=Real New D；E=New E。每批3轮，Run 1空场、Run 2有人、Run 3恢复空场。
- 每臂每轮 **864个不同原始包**。十二阵元：24个完整切换周期×6状态×6包。G2：同10秒中的864个G2包。两臂总包数相等，但不同时观测12阵元；每阵元包数及有效独立快拍数不同。
- R1前8周期/批（288包/臂）建立背景及权重；其余R1留作检查。R2前12周期/批（432包/臂）固定取峰，后12周期及半频段复核。R3不参与权重拟合。
- 原始数据15个CSI文件已重新解码。冻结缓存与原始选中包逐值相等，记录映射、接收链重排和实际包时间对应一致。这里的“映射验证”指**与既定映射及日志一致**，不替代现场接线审计。
- 15轮是五个站位各一组空→人→空流程，**不是每点15次或每点3次独立人体重复实验**。现有批次此前已反复查看，本轮为开发诊断，不是全新独立测试。
- 输入及逐包依据见 `source_audit.csv`、`packet_budget_audit.csv`、`input_hashes.json`；选包ID保留在旧工程的各点 `_packets.csv`，未重新抽样。

## G1到底哪里异常

### 幅度、零值与频率

在15轮中，链2包内功率占比中位数为 **{100*ch2.median_power_share.min():.2f}%–{100*ch2.median_power_share.max():.2f}%**；相对另外两链平均功率，逐轮中位比值低 **{-ch2.relative_other_chains_db.max():.2f}–{-ch2.relative_other_chains_db.min():.2f} dB**。这是原始CSI数字量的相对统计，不能当作绝对接收功率或SNR。

链2精确复数零值占比为 **{100*ch2.exact_zero_fraction.min():.2f}%–{100*ch2.exact_zero_fraction.max():.2f}%**，没有发现该链样本达到8位有符号分量的上下限。因此不是整条链断开成全零，也没有数字削顶的证据；不能据此排除模拟链路异常。

频率响应并非只坏一两个频点。例如C的Run 1全部30个频点的中位功率比均低于−10 dB；A则表现出明显频率选择性，部分频点并不很弱。不同轮次的频率形状也有变化。**观察支持“持续弱通路叠加频率选择性变化”，不能单凭这点在硬件频响和传播衰落之间作唯一判断。**

![频率响应](G1第二链频率响应.png)

### 同包链间相位：公共CFO/定时误差不能解释全部差异

对同一包、同一频点使用 `H_i conj(H_j)`，公共包相位与公共频率斜率会相消。每个状态访问中的包0/2/4作圆均值参考，包1/3/5作未参与拟合的检查。各轮的相位绝对偏差P90，再对15轮取中位数：

| G1链对 | 留出相位偏差P90的跨轮中位数 |
|---|---:|
| 1–2 | {ph1.loc['1-2','phase_abs_p90_deg']:.2f}° |
| 2–3 | {ph1.loc['2-3','phase_abs_p90_deg']:.2f}° |
| 1–3 | {ph1.loc['1-3','phase_abs_p90_deg']:.2f}° |

这不是AoA误差。它说明异常集中在涉及链2的空间相位关系，不能仅靠再扣一次公共相位来解决。

在目标链当前包及参考包幅度都为12–24、共同参考链1幅度≥12的对照中，链2相对链1的留出相位误差中位数 **13.75°**，链3相对链1为 **1.88°**。该对照不依赖逐包公共拟合。弱幅度与相位误差有关，但相同幅度并不保证相同噪声水平；该差异仍可能包含支路噪声/干扰/频响、不同信道及参考不稳，**不能直接把它写成已证实的硬件故障**。

状态访问的后3包也没有恢复：G1链1–2的三包圆标准差跨轮中位数，前3包 **9.82°**、后3包 **16.31°**；链1–3则约2.77°和2.76°。这不支持“仅移除刚切换的最早几个包即可解决”，但不是外部时钟或开关模拟瞬态的独立测量。

![相位诊断](G1第二链相位诊断.png)

### 可以排除或缩小什么

| 假设 | 本次证据与边界 |
|---|---|
| 单纯缓存、状态标签或解码重排错配 | 重新解码与逐包核验未发现；物理布线仍依赖已确认映射。 |
| 只是公共CFO或公共定时斜率 | 不足以解释，因为同包同频链间共轭量也不稳定。 |
| 整条RX接收链2普遍故障 | 其他状态的同一接收链正常，更应检查天线2所在通路。不能排除仅该开关状态相关的链路效应。 |
| 固定初始相位常量偏置 | 常量不能解释访问内快速相位变化；扣常量不是完整修复。 |
| 纯粹少数频点的窄带深衰落 | 多数记录存在宽频段偏弱；仅删几个频点不足。宽带衰落或硬件损耗仍可能。 |
| 天线/馈线/开关支路 vs 位置深衰落 | 缺少物理支路交换或已知TX位置改变的受控对照，现有数据不可唯一归因。 |

## 怎样降权：不把加权误当校准

方法名称：**R1可靠性加权的MUSIC子空间匹配（reliability-weighted subspace-fitting MUSIC）**。这是本次改造，未宣称复现某篇论文的完整算法。

每个站位仅用R1前8周期，根据链间共轭相位的留出圆误差计算 `V_ij=mean(1−cos(error_ij))`：

`w2 = clip(V13 / ((V12+V23)/2 + epsilon), 0.05, 1)`。

{md(weights[['point','weight2','unclipped_ratio','phase_valid_fraction']])}

这里 `w2` 是**平方残差权重**；阵元2在度量矩阵里的幅度系数为 `sqrt(w2)`。其他阵元权重为1。A–D触及预设下限0.05，E约0.107；保留非零权重，所以仍使用全部12个阵元。权重未利用人体坐标，其他轮次只作复核，未因结果差而重新选权重。

令 `h` 为R1背景主特征向量，`u` 为去背景后的前景主特征向量，`D=diag(sqrt(w))`。使用

`J_w(theta) = ||P_perp(D[h,u]) D a(theta)||² / ||P_perp(Dh) D a(theta)||²`。

子空间和物理导向矢量同时变换，原本的频点可用性、背景噪声底和阶数保持不变。**没有只压小CSI后仍把噪声当白噪声重新做普通MUSIC**。非均匀噪声文献区分噪声协方差估计和子空间加权，本次没有声称用相对幅度估出了真正的噪声协方差。[Esfandiari & Vorobyov](https://arxiv.org/abs/2109.15043)

另作一组加权公共相位/斜率拟合：仅降低G1链2在共同拟合目标中的权重，对三链施加同一个修正，**不单独把链2相位强行拉齐**。对非首参考包，相比原拟合，共同相位变化中位数 **{fitq.loc[.5,'common_phase_change_deg']:.2f}°**、P90 **{fitq.loc[.9,'common_phase_change_deg']:.2f}°**；定时变化中位数 **{fitq.loc[.5,'common_delay_change_ns']:.3f} ns**、P90 **{fitq.loc[.9,'common_delay_change_ns']:.3f} ns**。原拟合已受到强链主导，所以改动相对有限。

原始/降权使用同一共享天线拼接树。十二阵元协方差保留跨G块，是条件式复数拼接后的相干算法诊断，不等于同步采样或物理校准已通过。WiCAL的组内固定响应处理与组间对齐是不同步骤；降低坏链影响也不能替代组内标定。[WiCAL §II-B](https://arxiv.org/html/2505.21408v1#S2.SS2)

## 同包数五点结果

全局TX=(0,0)、RX=(3.65,0)；角度 `atan2(-y,3.65-x)`，统一以RX阵列中心为参考。E为新E=(1.64,−2.01)，不是早期绿色E。所有真值只在冻结代码、权重和预测后由独立脚本用于评分。

下面是 **Run 2前半段**固定规则取峰，不用理论角挑选次峰：

{md(tb)}

五点降权角度在1°网格上均未变化。D相对G2的数值偏差更小、E相同，**这些在原始十二阵元里已经存在，不是本次降权带来的收益**；A/B/C相对G2更差。仅移除天线2之后也是相同五个主线首选角，说明不能把整体错误分支简单归咎于这一条链。

![五点对照](五点等包数降权对照.png)

主线及全部预先声明的敏感性：

{md(primary)}

`numerical_outputs`只表示程序找到数值峰；`spectral_gate`表示后半段/半频段/空场对照的局部诊断门槛。它不包含全部物理条件。原始及降权十二阵元A–D虽然有稳定峰，B/C的稳定峰仍明显错误；E仍存在空场一致方向问题。G2五点均没有同时通过这次半频段复核。**不能把稳定错峰发布为人体方向。**

后空场与R1背景的阵元对相位差P90如下；15°门槛下，降权拟合及十一阵元均未恢复五点参考：

{md(reft)}

### 整轮描述性结果不能替换留出实验

{md(wt)}

{md(descriptive)}

例如D的G2前半段为−29°、后半段为+39°，用整轮会得到+38°。这正说明分支不稳。整轮G2的7.81°与前半段22.25°应并列说明，不能用前者覆盖失败复核；此前固定两个候选得到的6.81°则是另一套包预算和选峰规则，不能混为同一结果。

### R1经验相位参考仅作次要假设

同样的降权也在之前的R1经验相位处理假设下测试，并没有自动把R1当直达径硬件标定：

{md(empirical)}

该假设下十二阵元原始平均偏差62.33°、降权后65.53°，未改善。C从+17°变为+1°，不能拿个别更接近真值的旧谱峰替代新的固定规则输出。所有逐点值见 `scored_predictions.csv`。

![谱图](降权前后MUSIC谱.png)

## 下一步有依据的动作

- **保留链2低可靠标记及失败记录**，不要将其自动补齐、单链相位硬拉齐，或将十一阵元称为十二阵元成功。
- 现有数据已能判断：这条通路需要检查，但仅降低其影响不能恢复这五点的整体物理角度。继续围绕同一真值反复调权重没有独立验证意义。
- 最小物理区分试验：保持空场与RX位置，记录一个0°基线；在已标记的位置改变TX并返回0°，看弱通路是否随传播条件改变、回位能否复现。另独立建立一次受控端口交换试验：例如在开关输入端交换通向天线2和5的馈线接头，保持物理天线位置不动，准确记录新映射。若异常留在G1的原开关输入通路，支持该端口/支路问题；若跟随原天线2及其馈线进入G2，原因仍可能是天线/馈线或原空间位置，需要下一步交换天线部件与位置来区分。**交换连接后新建硬件会话和标定，不能沿用旧相位参考。**
- 完成独立已知方向源验证后，再检验完整十二阵元固定响应与静态人体路径。新的结果应在未参与调参的完整采集批次上与G2公平对照。

## 可复现文件

- `protocol.json`：事先写明的预算、权重、方法与验收。
- `analyze_g1.py`：原始核对、诊断、权重冻结、复数重组和MUSIC。
- `diagnose_g1_extended.py`：幅度匹配、前后访问窗口及参考稳定性辅助诊断；不修改预测。
- `report_results.py`：验证冻结哈希后才加入真值、制表和画图。
- `blind_predictions.csv` / `scored_predictions.csv`：全部方法、逐点角度、误差及门槛。
- `五点等包数角度对照.csv`、`method_summary.csv`：便于复查的汇总。
- `g1_common_phase_invariant_amplitude_match.csv`、`pair_phase_held.csv`、`g1_tone_profiles.csv`、`g1_switch_settling.csv`：诊断原表。
- `prediction_freeze.json`、`input_hashes.json`、`verification_receipt.json`：预测冻结、数据来源与核验。

执行：`python -B -X utf8 analyze_g1.py`，随后运行 `diagnose_g1_extended.py` 和 `report_results.py`。软件单元检查在已知34°合成数据上验证了3/11/12阵元及多种权重的一致性，**只是代码验证，不是实测性能证据**。
'''
    # Avoid allowing a hard-coded summary to silently disagree with the output tables.
    expected={'12_original':62.32700996118164,'12_R1_weighted_fit_and_scan':65.52700996118164}
    for meth,value in expected.items():
        actual=sm[(sm.calibration=='R1_empirical')&(sm.method==meth)].provisional_MAE_deg.iloc[0]
        assert abs(actual-value)<1e-5,(meth,actual,value)
    (R/'G1第二链排查与降权对照.md').write_text(report,encoding='utf8')
    dependencies=[Path(c.__file__),Path(c.prior.__file__),Path(c.base.__file__),Path(c.base.decoder.__file__),Path(c.rat.__file__),Path(c.rat.__file__).with_name('config.json')]
    a.dump('code_dependencies.json',{str(p):c.base.sha(p) for p in dependencies})
    a.dump('verification_receipt.json',dict(raw_files_verified=15,matched_budget_per_arm_run=864,
        prior_results_hashes_unchanged=True,new_prediction_freeze_verified=True,
        baseline_predictions_reproduced=True,methods=8,calibration_hypotheses=2,points=5,
        synthetic='3/11/12 elements, four weights, known direction; software only',
        physical_human_AoA_verified=False,independent_new_test=False,
        visual_review='pending model inspection of generated PNGs'))
    print(sm.to_string(index=False))

if __name__=='__main__':main()
