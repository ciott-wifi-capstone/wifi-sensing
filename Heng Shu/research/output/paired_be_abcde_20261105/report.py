"""Make the five-point readout from the registered spectra and their empty controls."""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
from aggregate import ROOT, FOLDERS, PRIMARY, reviewer, validate_frozen

plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
INK="#222c39"; BLUE="#33709b"; ORANGE="#c06020"; GREEN="#238372"; PURPLE="#75518d"


def read(name):
    return pd.read_csv(ROOT/name)


def angle(x,precision=0):
    if np.isclose(x,0):
        return "0°"
    return f"{x:+.{precision}f}°".replace("-","−")


def overview(table):
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.35),sharey=True,sharex=True,layout="constrained")
    points=np.arange(len(table))
    for ax,three,four,title in zip(axes,["G2_raw_R2","G2_R1ref_R2"],["four_G_raw_R2","four_G_R1ref_R2"],
                                  ["原始空间 MUSIC","用前空场作逐频相位参考"]):
        ax.axvline(0,color="#8d9aa5",lw=.9)
        for i in points:
            ax.axhline(i,color="#e2e8ed",lw=.85,zorder=0)
        ax.scatter(table.theory_centre_deg,points,facecolor="white",edgecolor=INK,s=66,zorder=5,marker="D",lw=1.7,label="阵列中心理论方向")
        ax.scatter(table[three],points-.17,c=BLUE,s=50,marker="s",label="G2 三阵元主峰",zorder=4)
        ax.scatter(table[four],points+.17,c=ORANGE,s=58,marker="o",label="G1–G4 融合主峰",zorder=4)
        ax.set(title=title,xlim=(-68,68),xticks=[-60,-40,-20,0,20,40,60],xlabel="RX 方向角 / °")
        ax.set_yticks(points,table.point)
        ax.tick_params(axis="y",length=0,pad=8)
        ax.grid(axis="x",alpha=.12)
    axes[0].set_ylim(len(table)-.45,-.55)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("五个站位的理论方向与有人窗口盲选主峰",fontsize=14)
    fig.savefig(ROOT/"ABCDE_理论方向与实测主峰.png",dpi=180)
    plt.close(fig)


def db(s):
    z=s/np.max(s)
    return 10*np.log10(np.maximum(z,1e-12))


def fusion_figure(table):
    fig,axes=plt.subplots(2,2,figsize=(10.7,6.3),layout="constrained",sharex=True)
    for col,site in enumerate(["B","E"]):
        gt=table.set_index("point").loc[site,"theory_centre_deg"]
        with np.load(FOLDERS[site]/"angular_spectra.npz") as z:
            axis=z["angles"]
            ax=axes[0,col]
            for r,name,color,style in [(1,"前空场",BLUE,"--"),(2,"人在场",ORANGE,"-"),(3,"后空场",GREEN,":")]:
                curve=db(z[f"fusion_R{r}_raw"])
                idx=curve.argmax()
                ax.plot(axis,curve,c=color,ls=style,lw=1.8,label=name)
                ax.scatter(axis[idx],curve[idx],s=20,c=color,zorder=5)
            ax.axvline(gt,c=INK,ls=(0,(2,3)),lw=1.2,label="理论方向")
            ax.set_title(f"{site} 点三轮全频段融合谱")
            ax.legend(ncol=2,fontsize=8,frameon=False)
            ax=axes[1,col]
            bands=np.mean([z[f"G{i}_R2_raw"] for i in range(1,5)],axis=0)
            for band,name,color,style in [(0,"全频段",ORANGE,"-"),(1,"低半频段",BLUE,"--"),(2,"高半频段",GREEN,"-.")]:
                curve=db(bands[band])
                idx=curve.argmax()
                ax.plot(axis,curve,c=color,ls=style,lw=1.7,label=f"{name} {angle(axis[idx])}")
                ax.scatter(axis[idx],curve[idx],s=21,c=color,zorder=5)
            ax.axvline(gt,c=INK,ls=(0,(2,3)),lw=1.2)
            ax.set_title(f"{site} 点有人窗口的频段检查")
            ax.legend(ncol=2,fontsize=8,frameon=False)
    for ax in axes.ravel():
        ax.set(xlim=(-90,90),xticks=[-90,-60,-30,0,30,60,90])
        ax.grid(alpha=.15)
    for ax in axes[0]: ax.set_ylim(-2.85,.25)
    for ax in axes[1]: ax.set_ylim(-3.5,.3)
    for ax in axes[:,0]: ax.set_ylabel("相对各曲线主峰 / dB")
    for ax in axes[-1]: ax.set_xlabel("角度参数 / °")
    fig.savefig(ROOT/"BE_三轮融合谱与分频段检查.png",dpi=180)
    plt.close(fig)


def candidate_figure(bc):
    fig,axes=plt.subplots(1,2,figsize=(10.7,5.0),layout="constrained")
    ax=axes[0]
    for site,name,color,style in [("B_earlier","上次 B",BLUE,"--"),("B","本次 B",ORANGE,"-")]:
        with np.load(FOLDERS[site]/"exploratory_empirical_spectra.npz") as z:
            axis=z["angles"]
            v=db(z["C1_R2"])
            idx=v.argmax()
            ax.plot(axis,v,color=color,ls=style,lw=2,label=f"{name} 主峰 {angle(axis[idx])}")
            ax.scatter([axis[idx]],[0],color=color,s=28)
    ax.axvline(bc.geometry("B")[0]["C1"],color=INK,ls=":",label="B 的 C1 理论角")
    ax.set(xlim=(-90,90),ylim=(-11.4,.6),xticks=[-90,-60,-30,0,30,60,90],
           xlabel="角度参数 / °",ylabel="相对各自主峰 / dB",title="B · 同点两次采集的 C1 逐频参考谱")
    ax.legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2);ax.grid(alpha=.15)
    peaks=bc.read("E","exploratory_empirical_bg_peaks")
    peaks=peaks[peaks.stable&(peaks.state=="C1")]
    ax=axes[1]
    for ref,color,marker,offset in [(1,BLUE,"s",-0.8),(3,GREEN,"o",0.8)]:
        p=peaks.query("reference==@ref and test==2")
        norm=p.ratio.max()
        ax.scatter(p.angle+offset,p.ratio/norm,color=color,s=35,marker=marker,label=f"R{ref} 作背景")
        v=peaks.query("reference==@ref and test!=2")
        # Vertically separated empty controls are existence markers, not power readings.
        ax.scatter(v.angle+offset,np.full(len(v),-.09 if ref==1 else -.18),color=color,s=22,marker="x")
    ax.axhline(0,color="#aab3bb",lw=.8)
    gt=bc.geometry("E")[0]["C1"]
    ax.axvline(gt,color=INK,ls=":",label=f"E 的 C1 理论角 {angle(gt,1)}")
    ax.plot([],[],marker="x",ls="none",color=INK,label="对应空场也有的峰")
    ax.set(xlim=(-94,94),xticks=[-90,-60,-30,0,30,60,90],ylim=(-.25,1.16),yticks=[0,.25,.5,.75,1],
           xlabel="角度参数 / °",ylabel="各参考内，有人候选相对分数",title="E · C1 的 +42° 候选与空场对照")
    ax.legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2)
    ax.grid(alpha=.15)
    fig.savefig(ROOT/"B同点复验与E的C1候选.png",dpi=180)
    plt.close(fig)


def md(fields):
    return "| "+" | ".join(str(x) for x in fields)+" |"


def build_report(bc,t):
    angles=read("directions_by_state.csv").set_index(["point","state"])
    cs=read("paired_comparisons.csv").query('metric=="covariance_distance"')
    q=read("equal_visit_contrast.csv").set_index(["point","state"])
    baselines=[]
    for row in t.itertuples():
        baselines.append(md([row.point,f"({row.X_m:.2f}, {row.Y_m:+.2f})",row.source,angle(row.theory_centre_deg,2),
                             angle(row.G2_raw_R2),angle(row.four_G_raw_R2),f"{row.four_G_raw_abs_diff_deg:.2f}°",
                             angle(row.G2_R1ref_R2),angle(row.four_G_R1ref_R2)]))
    effects=[]
    for site in ["B","E"]:
        for s in bc.STATES:
            h=cs[(cs.point==site)&(cs.state==s)].iloc[0]
            effects.append(md([site,s,f"{h.empty_drift:.3f}",f"{h.appeared:.3f}",f"{h.departed:.3f}",f"{h.ratio:.3f}",
                               f"{h.passing_quarters}/4",f"{q.loc[(site,s),'ratio']:.3f}"]))
    subarray=[]
    for site in PRIMARY:
        p=angles.loc[site]
        subarray.append(md([site,*[f"{angle(p.loc[s,'raw_r2'])} / {angle(p.loc[s,'per_tone_r2'])}" for s in bc.STATES]]))
    r2=read("raw_fusion_sensitivity.csv")
    segments=[]
    for site in ["B","E"]:
        rows=r2.query('point==@site and run==2 and weight=="packet"')
        full=rows.query('window=="full"').set_index("band")
        blocks=rows.query('window!="full" and band=="full"')
        segments.append(md([site,angle(full.loc["full","peak_deg"]),angle(full.loc["low","peak_deg"]),angle(full.loc["high","peak_deg"]),
                            " / ".join(angle(a) for a in blocks.peak_deg),angle(full.loc["full","next_local_deg"]),f"{-full.loc['full','next_local_db']:.2f} dB"]))
    mean=t.four_G_raw_abs_diff_deg.mean()
    near=int((t.four_G_raw_abs_diff_deg<=8).sum())
    same_g3=[]
    for row in t.itertuples():
        sequence=" → ".join(angle(v) for v in (row.G3_R1ref_R1,row.G3_R1ref_R2,row.G3_R1ref_R3))
        same_g3.append(md([row.point,angle(row.theory_G3_deg,2),sequence,
                           f"{row.G3_R1ref_abs_diff_deg:.2f}°",f"{row.G3_reference_pairs_passed}/2"]))

    # Look up specific previously reported A candidates, without re-ranking directions by truth.
    a_background=pd.read_csv(FOLDERS["A"]/"background_peaks_vs_geometry.csv")
    a_c1=a_background.query('state=="C1" and reference==1 and test==2 and angle_parameter_deg==-12 and stable').iloc[0]
    empty=a_background.query('state=="C1" and reference==1 and test==3 and angle_parameter_deg==-15 and stable').iloc[0]
    reverse=a_background.query('state=="C1" and reference==3 and test==2 and stable')
    assert not reverse.angle_parameter_deg.sub(a_c1.angle_parameter_deg).abs().le(9).any()
    prior_a=pd.read_csv(ROOT.parent/"kong1_kong1a_20261105"/"A_peak_geometry_comparison.csv")
    old_c2=prior_a.query('batch=="Kong1A" and run==1 and state=="C2" and mode=="raw" and rank==2').iloc[0]
    matched=read("two_reference_including_empty_edges.csv")
    empirical=matched.query('site=="A" and mode=="empirical_background" and state=="C1" and pre_angle==-9 and post_angle==-9').iloc[0]
    assert empirical.pre_empty_control_near_including_edges and not empirical.clears_empty_controls
    return [
        "# A–E 五点的理论方向和 CSI 实测方向",
        "",
        "相同的盲选规则在 D、E 的原始四组融合谱上给出了靠近理论方向的主峰。D 的谱峰偏差为 4.00°，E 为 9.00°。B 重测时出现了可辨认的信道变化，但最高峰落在人体的另一侧。E 另外有一个 C1 的 +42° 候选，同时通过前、后空场对照，可在独立进场记录中复验。",
        "",
        "**A 点已有两项不同的主峰结果。** 同一份 New A 记录，G3 用前空场逐频参考后得到 −6°，距 G3 理论方向 5.08°。四组原始谱融合得到 −30°，距阵列中心理论方向 18.11°。两项都保留，按各自的方法和角度原点比较。",
        "",
        "## 五点实测",
        "",
        "左端 TX=(0,0)，右端 RX 阵列中心=(3.65,0)。以 RX 指向 TX 为 0°，图上方为负角，下方为正角。人体在 (X,Y) 时，阵列中心理论方向为 atan2(−Y, 3.65−X)。此次 E 对应粉色的 (1.64,−2.01)。站位取操作者标签，每一行使用该站位最新的一次完整“前空场 R1 → 有人 R2 → 后空场 R3”，旧 B、旧 C 另作重复观测。",
        "",
        "| 点 | 坐标 / m | 采用的采集 | 中心理论方向 | 原始 G2 主峰 | 原始四组融合主峰 | 融合谱峰绝对差 | 逐频参考 G2 主峰 | 逐频参考融合主峰 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",*baselines,
        "",
        "以上实测栏均取 R2 全频段同频空间 MUSIC 谱的盲选最大峰，角度步长 3°。G2 是第 4、5、6 根天线。四组融合把 G1–G4 各自求得的归一化谱等权相加，共使用十二根天线，只在组内使用相位。逐频参考以本批 R1 前 5 秒拟合，保留按真实阵元位置计算的直达传播相位。相位参考仍包含室内多径。表中融合偏差相对 RX 中心，单子阵理论角另按子阵中心算。",
        "",
        f"固定原始融合规则在五点中的 {near}/5 点落在理论方向 ±8° 内，五个主峰与站位方向的平均绝对差为 {mean:.2f}°。这个数描述所选谱峰。人体路径的身份尚未独立验证，人体 AoA 的 MAE 和二维位置误差留空。",
        "",
        "### 按原先 G3 的方法对照五点",
        "",
        "各批用自己的前空场建立逐频参考，再在全角域选主峰。偏差按 G3 自己的阵列中心计算。该方法在 A 点形成假设，随后在其他点复验。",
        "",
        "| 最新批次站位 | G3 理论方向 | 前空场 → 有人 → 后空场主峰 | 有人主峰绝对偏差 | 留出后空场参考检查通过链对 |",
        "|---|---:|---|---:|---:|",*same_g3,
        "",
        "之前截图采用较早的 New B 和 New C。其 G3 三轮主峰分别为 0° → +6° → +3°、0° → −3° → −3°，有人峰偏差为 59.72°、26.55°。上表采用 Real New B 和 Real New C，A 仍为原来的 New A。",
        "",
        "### A 点还有哪些更接近站位的峰",
        "",
        f"- New A 的 C1 用前空场作协方差白化背景，在有人段保留了 {angle(a_c1.angle_parameter_deg)} 候选，C1 理论为 {angle(a_c1.theory_deg,2)}，绝对差仅 {a_c1.posthoc_difference_deg:.2f}°。该候选得到两个半频段和四个时间块支持，但同一参考下的后空场也有 {angle(empty.angle_parameter_deg)} 稳定峰。改用后空场为背景，原方向 ±9° 内没有通过筛选的稳定候选。这个数字来自多个候选与真值的事后对照。",
        f"- 在相同的 New A 再加逐频参考，C1 在前、后两种背景下均保留 {angle(empirical.pre_angle)} 候选，绝对差 {empirical.pre_difference:.2f}°。前空场作背景时，后空场在同一角度也有稳定峰。",
        f"- 更早的 Kong1A 三轮均有人。第一轮 C2 的第二强局部峰为 {angle(old_c2.peak_deg)}，C2 理论为 {angle(old_c2.theory_deg,2)}，绝对差 {old_c2.retrospective_absolute_difference_deg:.2f}°。第二轮 −6° 未通过两半频段检查，第三轮没有复现邻近分支。该批的有人、空场分属两次接收机初始化。",
        "",
        "[New A 原始报告](../new_a_paired_20261105/New_A_空场有人空场_实测分析.md) · [New A 白化候选与空场对照](../new_a_paired_20261105/background_peaks_vs_geometry.csv) · [Kong1A 原始候选](../kong1_kong1a_20261105/A_peak_geometry_comparison.csv)。",
        "",
        "![五点角度对照](ABCDE_理论方向与实测主峰.png)",
        "",
        "## 新 B 和新 E 的原始数据与空场对照",
        "",
        "两批各有三个 10 秒 HT40 窗口，每批 36 个源清单项目全部核验。B 的 R1/R2/R3 目标包数为 9684 / 9620 / 9654，E 为 9568 / 9551 / 9069。B 为 phy12/ifindex 39，E 为 phy10/ifindex 33。各批内部三轮的接口身份一致，continuity_gate 全部通过。链重排、READY 状态映射与 CSI 头信息均经核对。所有完整驻留窗口都有目标包。E 的后空场第一秒仅 739 包，全窗口最长包间隔 28.92 ms，首尾都有数据。按每次开关驻留等权重算，B、E 三轮融合主峰和六状态的全窗口对照判断均保持一致。",
        "",
        "逐包归一化后计算同频三链空间协方差。表中的比值为 min(D12,D23)/D13，D13 为前后空场的相对协方差距离，D12 和 D23 分别描述入场和离场。比值大于 1 表示两次有人/空场变化均超过这一次空场自身的变化。",
        "",
        "| 点 | 状态 | 前后空场 D13 | 入场 D12 | 离场 D23 | 比值 | 超过空场的 2.5 秒分块 | 驻留等权比值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",*effects,
        "",
        "B、E 均为 5/6 状态全窗口超过空场，均有 4/6 状态在四个分块全部超过。B 的 C2 比值约 0.999，E 的 G1 约 0.920。换用 5 ms 切换保护间隔，判断相同。每批四个时间块属于同一次人体进出，无法用作四次独立试验。前后夹空场支持把部分变化与人的出现联系起来。遮挡改变已有多径也会产生这种变化。",
        "",
        "## B 的复验和 E 的方向线索",
        "",
        "- **B。** 先前 B 的 C1 逐频参考主峰为 0° → −63° → 0°。新 B 为 0° → 0° → 0°。−63° 在本次有人段仍是第二强的局部峰，只比主峰低 0.07 dB。前、后空场在 −63° 的谱值也只分别低于各自主峰 0.94、0.12 dB。R2 的低、高半频段分别选 +60° 和 −63°，后空场的部分时间块也选出 −63°。两次 B 都能看到这条角分支，但后空场也支持它，人体归属仍缺少证据。原始 G3 二维 MUSIC 的 R2 主峰为 −60°，与 G3 理论 −53.72° 在同侧，前后空场也在 −69° 和 −81°。固定的 G2 与四组融合均落在正角度一侧。",
        "- **E。** 原始四组融合在三个窗口依次为 −33°、+36°、−33°。R2 中另一个 −30° 局部峰只低 0.20 dB，低半频段选 +39°，高半频段选 −27°，最后一个时间块也选 −30°。G4 的 +36° 在空场和有人时均为主峰。由此可见，人进场改变了不同方向的相对权重，+36° 分支在空场也有响应。",
        "- **E 的 C1 候选。** 在逐频参考和协方差白化后，分别用前空场和后空场作背景，R2 中 +42° 均为 C1 内分数最高的稳定候选，两个半频段和四个时间块均有邻近峰。在对应空场对照的 ±9° 内均无稳定峰。它距离 C1 理论 +44.40° 为 2.40°。以 R1 和 R3 各为背景时，+42° 得分为 26.17 和 31.34，同场也有 −30° 竞争峰（24.99 和 29.18），而 −30° 与空场峰重叠。+78° 还与后空场的 +87° 边界峰邻近。参考切换用了同一段有人数据，只检验背景选择的敏感性。两条 C1 相对链在后空场的相位残差 P90 为 17.14°、18.52°，均超过预设的 15°。G3 在同一处理下还有 −45° 候选，方向解释仍有分歧。",
        "- **E 的 C2。** 仅用前空场参考时有 +45°，但后空场对照也有 +42°，换后空场为参考后 +45° 未通过相同稳定性筛选。按实际频率和阵元坐标计算，C2 在 +45° 和 −18° 的理想导向矢量相似度中位数为 0.9993，单靠该组很难区分两方向。",
        "",
        "| 点 | R2 原始融合 | 低半频段 | 高半频段 | 四个时间块主峰 | 第二局部峰 | 第二峰比主峰低 |",
        "|---|---:|---:|---:|---|---:|---:|",*segments,
        "",
        "![新 B 和 E 三轮融合谱及分频结果](BE_三轮融合谱与分频段检查.png)",
        "",
        "![B 的 C1 重测与 E 的 C1 候选](B同点复验与E的C1候选.png)",
        "",
        "## 其余站位的证据",
        "",
        "- **A。** G3 的两条相对链均通过后空场参考检查，相位残差第 90 百分位分别为 14.73° 和 13.09°。有人段中，0° 的谱值仅比 −6° 主峰低 0.61 dB。谱峰较宽，这一次的偏转仍需独立进场记录复验。",
        "- **C。** 本批四组原始融合主峰为 +3°，理论 +22.83°。C2 在逐频参考的白化搜索中分别给出 +18°、+24°（以 R1、R3 各作背景），两者包围 C2 理论 +23.55°。同一子阵存在方向歧义，其他子阵也保有不同方向。两次独立采集的 C 都有六组全窗口配对变化。",
        "- **D。** 原始 G2 是 −36° → +54° → −36°，R2 与 G2 理论 +51.26° 相差 2.74°。原始融合 +48° 距阵列中心理论 +52.00° 为 4.00°。G2 的低、高半频段分别为 +66°、−27°。C2 则同时保留 +48° 和 −18°，两方向理想响应相似度中位数为 0.9898。",
        "",
        "## 各三阵元在有人段选到的主峰",
        "",
        "各格先列原始空间 MUSIC 主峰，再列以前空场建立逐频参考的主峰。C1、C2 是开关参考状态，各观测三根物理天线。",
        "",
        "| 点 | G1 | G2 | G3 | G4 | C1 | C2 |","|---|---:|---:|---:|---:|---:|---:|",*subarray,
        "",
        "共享天线 6/8 的成对相位在新 B、E 各轮的周期残差中位数均小于 2°。但常数或频率线性参考检查均为 0/12 链对通过。逐频参考 B 为 1/12，E 为 0/12。六次新采集加上各点历史重复共有 21 个互不重复的 CSI 文件。基于现有对照，可以报告在场相关的信道变化和具体候选方向。要检验十二根天线相对三根的物理增益，仍需独立方向标定、跨组相干验证以及冻结规则后的完整留出进场记录。",
        "",
        "## 复算和数据表",
        "",
        "依次运行本目录的 process.py、aggregate.py、report.py。第一步在引入人体真值前保存新 B/E 盲选输出和哈希。第二步才关联坐标，并用未改动的 A、旧 B/C、新 C/D 结果汇总。原始数据位于每点目录的 raw_runs.csv 所列绝对路径。报告中两背景候选的空场否决把 ±87° 和 ±90° 边界也算作潜在空场干扰，同时保留历史同口径表。",
        "",
        "[五点方向汇总](ABCDE_angles.csv) · [各状态、各窗口、理论方向及偏差](directions_by_state.csv) · [B 的预设 −63° 分支](B_C1_minus63_fixed_branch.csv) · [两参考候选与完整空场否决](two_reference_including_empty_edges.csv) · [频段和分块敏感性](raw_fusion_sensitivity.csv) · [各状态空场距离](paired_comparisons.csv) · [采集时钟与覆盖](capture_window_check.csv) · [原始文件索引](record_index.csv) · [接收阵列歧义](steering_similarity.csv) · [盲选输出哈希](blind_registry.json)。",
    ]


def verify():
    p=reviewer();validate_frozen(p.bc)
    table=read("ABCDE_angles.csv")
    assert list(table.point)==list(PRIMARY) and table.independently_confirmed_human_aoa_deg.isna().all()
    assert len(read("record_index.csv"))==read("record_index.csv").sha256.nunique()==21
    # Cross-check all five primary readouts with the stored numerical spectra.
    for row in table.itertuples():
        site=row.point
        with np.load(FOLDERS[site]/"angular_spectra.npz") as z:
            assert row.G2_raw_R2==z["angles"][z["G2_R2_raw"][0].argmax()]
            assert row.four_G_raw_R2==z["angles"][z["fusion_R2_raw"].argmax()]
        with np.load(FOLDERS[site]/"exploratory_empirical_spectra.npz") as z:
            assert row.G2_R1ref_R2==z["angles"][z["G2_R2"].argmax()]
            fused=np.mean([z[f"G{i}_R2"] for i in range(1,5)],axis=0)
            assert row.four_G_R1ref_R2==z["angles"][fused.argmax()]
        assert np.isclose(row.theory_centre_deg,np.degrees(np.arctan2(-row.Y_m,3.65-row.X_m)))
    assert table.loc[table.point=="E","theory_centre_deg"].iat[0]==45.
    e=read("two_reference_including_empty_edges.csv").query('site=="E" and state=="C1" and mode=="empirical_background"')
    assert e.query("pre_angle==42 and post_angle==42").clears_empty_controls.all()
    assert len(e.query("pre_angle==42 and post_angle==42"))==1
    assert not e.query("pre_angle==78 and post_angle==78").clears_empty_controls.any()
    overview(table); fusion_figure(table); candidate_figure(p.bc)
    report=ROOT/"ABCDE_理论与实测方向.md"
    report.write_text("\n".join(build_report(p.bc,table))+"\n",encoding="utf8")
    links=re.findall(r'\]\(([^)]*)\)',report.read_text(encoding="utf8"))
    assert len(links)>=12 and all((ROOT/link).is_file() for link in links)
    print(f"Matched numerical spectra for five points; {len(links)} report links verified.")


if __name__=="__main__":
    verify()

