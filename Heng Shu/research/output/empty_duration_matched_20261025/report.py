from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    fig.savefig(R/'figures'/f'{name}.png',dpi=160);fig.savefig(R/'figures'/f'{name}.svg');plt.close(fig)
def main():
    df=pd.read_csv(R/'results/pair_comparisons.csv');summ=pd.read_csv(R/'results/summary.csv');ph=pd.read_csv(R/'results/phase_summary.csv');sq=pd.read_csv(R/'results/state_quality.csv')
    main=summ[(summ.variant=='full')&(summ.scope=='all_run_pairs')];groups=['within_E1','within_E2','E1_E2','within_E3','within_E4','E3_E4'];labels=['Within E1','Within E2','E1 vs E2\n10 s','Within E3','Within E4','E3 vs E4\n15 s']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,col,title in zip(axes,['amplitude_distance','spatial_distance','spectrum_rms'],['Normalized amplitude distance','Same-frequency spatial distance','MUSIC spectral RMS difference']):
        vals=main.set_index('pairing').reindex(groups)[col];ax.bar(range(6),vals,color=['#9ecae1','#9ecae1','#e6550d','#a1d99b','#a1d99b','#e6550d']);ax.set(xticks=range(6),xticklabels=labels,title=title);ax.tick_params(axis='x',rotation=40,labelsize=8)
    fig.suptitle('Equal-duration comparisons: 3 x 3 run pairs x 6 states; no phase correction');save(fig,'01_duration_matched_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax,pair,title in zip(axes,['E1_E2','E3_E4'],['E1 vs E2: 10 s vs 10 s','E3 vs E4: 15 s vs 15 s']):
        mat=ph[ph.pairing==pair].pivot(index='state',columns='chain_pair',values='median_abs_difference_deg').reindex(index=list(rat.MAP),columns=['chain1-chain2','chain1-chain3','chain2-chain3']).to_numpy()
        im=ax.imshow(mat,vmin=0,vmax=180,cmap='magma',aspect='auto');ax.set(xticks=range(3),xticklabels=['Chain 1-2','Chain 1-3','Chain 2-3'],yticks=range(6),yticklabels=list(rat.MAP),title=title)
        for i in range(6):
            for j in range(3):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',color='black' if mat[i,j]>100 else 'white')
    fig.colorbar(im,ax=axes,label='Median absolute circular phase difference (deg)',shrink=.8)
    fig.suptitle('Matched chain-pair pattern: near half-cycle offsets involve chain 1');save(fig,'02_chain_pair_phase_pattern')
    fig,axes=plt.subplots(6,4,figsize=(13,13),sharex=True,sharey=True,layout='constrained')
    for j,cap in enumerate(['E1','E2','E3','E4']):
        z=np.load(R/'cache'/f'{cap}_spectra.npz')['A'];sp=np.median(z,axis=0)
        for k,state in enumerate(rat.MAP):
            ax=axes[k,j];im=ax.imshow(10*np.log10(np.maximum(sp[k],1e-3)),origin='lower',extent=[-400,400,-90,90],aspect='auto',vmin=-30,vmax=0,cmap='viridis');ax.set_title(f'{cap} / {state}',fontsize=9)
            if j==0:ax.set_ylabel('Angle parameter (deg)',fontsize=8)
            if k==5:ax.set_xlabel('Delay parameter (ns)',fontsize=8)
            ax.tick_params(labelsize=7)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Median of three individually normalized baseline spectra (dB)',shrink=.55)
    fig.suptitle('E1/E2: 10 s | E3/E4: 15 s. Uncalibrated spectra; no human-path labels.');save(fig,'03_empty_spectra_overview')
    select=summ[(summ.variant=='full')&summ.pairing.isin(['E1_E2','E3_E4'])];sensitivity=summ[(summ.variant=='equal_packet_count_max512')&summ.pairing.isin(['E1_E2','E3_E4'])&(summ.scope=='all_run_pairs')]
    mini=ph.groupby(['pairing','chain_pair'])[['median_abs_difference_deg','circular_concentration']].median().reset_index()
    text=f'''# 按相同采样时长匹配 E1–E4

2026-10-25。根据用户要求，主比较改为E1–E2（10s对10s）和E3–E4（15s对15s）。旧E1–E3报告保留；两种配对回答不同问题，不能用改组删除原有证据。本轮没有更新论文或Overleaf。

## 1. 配对与计算范围

| 配对 | 第一组 | 第二组 | 轮次匹配 |
|---|---|---|---|
| 10 s对10 s | E1：20261003_130149_student01_6613，索引78–80 | E2：20261003_130256_student01_11355，81–83 | 3×3全轮次组合；另列run1↔run1等编号配对 |
| 15 s对15 s | E3：20261018_151129_student01_118072，84–86 | E4：20261018_151420_student01_127667，87–89 | 同上 |

时长从原始run.json的requested_seconds核对，不从目录日期推断。使用原有保护区状态选择；同编号并不表示同时观测或时间同步。每个时长组包含9个轮对×6状态=54项比较；编号配对为3×6=18项。另保留每个目录内部3个轮对×6状态作为参照。比较共享原始记录，不能当作54次独立实验。

已确认E1与E3为连续同会话；E1与E2、E3与E4是否同会话仍未确认。时长匹配不改变上述事实，也不改变空场与四点有人测量属于不同会话的确认。

原始哈希12个全部匹配；72个状态的重建基线协方差与旧缓存一致到1e-10以内。E1/E3复用已解析数据，E2/E4仅补充解析；原谱和旧A/C参数复用，不重新拟合相位，也不使用人体坐标。输入和缓存哈希见[source_hashes.csv](results/source_hashes.csv)。

## 2. 相同长度并没有使原始复数背景直接一致

距离为对称相对Frobenius距离，幅度先每包归一化；谱RMS比较每张归一化谱。数值越小表示对应指标越相近，不是定位米误差。下表为跨轮对与状态的中位数：

{select[['pairing','scope','state_pairs','spatial_distance','amplitude_distance','covariance_distance','spectrum_rms','pair_phase_median_deg']].to_markdown(index=False,floatfmt='.4f')}

同目录内部参照：

{main[main.pairing.str.startswith('within')][['pairing','spatial_distance','amplitude_distance','covariance_distance','spectrum_rms']].to_markdown(index=False,floatfmt='.4f')}

两种同长度配对的幅度距离只有约0.066和0.044，与组内幅度波动数量级接近；但空间距离仍约1.15和1.11，远大于组内约0.05–0.11。因此，配对结果更像是“幅度相近而空间相位参考不同”，不能简单称为信道完全不稳定，也不能说已经可以直接做复数背景相减。

为了排除包数不等，另对每个状态轮对按等间隔抽取相同包数（最多512），不按幅度或谱质量选包：

{sensitivity[['pairing','spatial_distance','amplitude_distance','covariance_distance','spectrum_rms']].to_markdown(index=False,floatfmt='.4f')}

![同长度比较](figures/01_duration_matched_comparison.png)

## 3. 新发现：相位差集中在涉及链1的链对

对每轮每状态，先平均同频链间共轭乘积得到链对相位，再计算记录之间的包裹圆差。仅在双方链对相干度>=0.5的频点描述相位；全部支持标记保存在[phase_traces.csv](results/phase_traces.csv)，未据此修改CSI。

{mini.to_markdown(index=False,floatfmt='.3f')}

逐状态结果显示：两组配对中，链1–2、链1–3通常接近180°差，而链2–3仍接近0°差。跨频率圆集中度较高。**这一模式与“重排后链1相对另外两链发生近似π相位翻转”的假设相容。** 因为给链1乘−1会使两个涉及链1的共轭乘积翻转，而链2–3保持不变；一个三链共享的公共相位不能产生该模式。

这是数据支持的新诊断线索，尚不是已识别的硬件原因。未确认的采集端符号约定、链相关响应或其他静态参考因素仍可能产生类似现象。图中的链号是解析和permutation重排后的接收链，不等于四组中某一根固定物理天线。不能自动把全部状态、全部会话或四点有人数据都按同一规则翻转。

![逐链对相位模式](figures/02_chain_pair_phase_pattern.png)

## 4. 对恢复路线的影响

这次同长度匹配确实找到了比“所有空场互不一致”更具体的结构：幅度相近，且两组都呈现链1相关的近半周差。它值得优先验证，而不是继续仅调公共CPE/定时参考。

下一项有界检验应只用一对开发轮次决定是否采用固定链符号模型，冻结后验证剩余完整轮次、其他状态及未参与拟合的频点；必须同时改善空间协方差和谱重复性。不能逐记录自由拟合三根链的相位，再以拟合后残差下降宣布标定完成。如果模型成立，也只得到这些空场之间的相对参考，不能自动迁移到另一会话的人体数据或获得绝对AoA/ToF。

本次按要求完成匹配与诊断，**没有将链1乘−1应用到结果，也没有宣称人体定位已经恢复**。旧Ratnam逐状态公共模型在这些空场中通过数为 {int(sq.old_common_model_pass.sum())}/{len(sq)}；这个门槛检验不包含链特异符号校正，不能替代上述新假设验证。

![四组空场谱概览](figures/03_empty_spectra_overview.png)

## 5. 复现与范围

从本目录按顺序执行 `analyze.py`、`phase_diagnostic.py`、`report.py`；运行环境为python，依赖NumPy、SciPy、Pandas、Matplotlib、threadpoolctl及tabulate。分析复用前一独立工程的冻结解析器与Ratnam模块，路径和SHA见[dependency_hashes.csv](dependency_hashes.csv)。四点原始数据没有重算、修改或用于挑峰。

本目录独立保存；原始数据、旧E1–E3报告及旧交付包未覆盖。数值完整表见[summary.csv](results/summary.csv)、[pair_comparisons.csv](results/pair_comparisons.csv)、[phase_summary.csv](results/phase_summary.csv)。CSV包含全部轮次与状态，不只展示最接近的配对。
'''
    (R/'REPORT.md').write_text(text,encoding='utf-8')
    deps=[PRE/'common.py',PRE/'provenance/ratnam/core.py',PRE/'provenance/ratnam/config.json',PRE/'provenance/analyze.py',PRE/'empty_manifest.csv',OLD/'results/state_summary.csv']
    pd.DataFrame([dict(path=str(p),sha256=sha(p)) for p in deps]).to_csv(R/'dependency_hashes.csv',index=False)
    print(mini.to_string(index=False));print('Report and three figures written')
if __name__=='__main__':main()
