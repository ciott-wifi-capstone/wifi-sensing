"""Compare the two C acquisitions and the new D, after blind peak selection."""
import importlib.util
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("previous_paired_report",ROOT.parent/"paired_bc_20261105"/"report.py")
bc=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
bc.ROOT=ROOT
bc.FOLDERS={"C_previous":ROOT.parent/"paired_bc_20261105"/"C","C":ROOT/"C","D":ROOT/"D"}
bc.POINTS={"C_previous":(.8,-1.2),"C":(.8,-1.2),"D":(2.4,-1.6)}
plt=bc.plt
SITES=tuple(bc.FOLDERS)
NAMES={"C_previous":"上次 C","C":"本次 C","D":"本次 D"}
COLORS=["#446d94","#bb5421","#17816d"]


def two_reference_candidates():
    """Pair stable foreground peaks across both references; annotate empty controls before evaluating truth."""
    rows=[]
    for site in SITES:
        gt,_=bc.geometry(site)
        for mode,name,col in [("raw_background","blind_background_candidates","angle_parameter_deg"),
                              ("empirical_background","exploratory_empirical_bg_peaks","angle")]:
            df=bc.read(site,name)
            df=df[df.stable & df[col].abs().lt(87)]
            for state,g in df.groupby("state"):
                def subset(ref,test):
                    return g[(g.reference==ref)&(g.test==test)]
                for _,a in subset(1,2).iterrows():
                    for _,b in subset(3,2).iterrows():
                        u,v=a[col],b[col]
                        if abs(u-v)>9:
                            continue
                        near_pre_control=bool((subset(1,3)[col]-u).abs().le(9).any())
                        near_post_control=bool((subset(3,1)[col]-v).abs().le(9).any())
                        rows.append(dict(site=site,mode=mode,state=state,pre_angle=u,post_angle=v,
                                         pre_ratio=a.ratio,post_ratio=b.ratio,
                                         pre_empty_control_near=near_pre_control,
                                         post_empty_control_near=near_post_control,
                                         passes_both_empty_vetoes=not(near_pre_control or near_post_control),
                                         theoretical_deg=gt[state],pre_difference=abs(u-gt[state]),
                                         post_difference=abs(v-gt[state])))
    df=pd.DataFrame(rows);df.to_csv(ROOT/"two_reference_candidates.csv",index=False,encoding="utf-8-sig")
    return df


def alias_control():
    y=np.array(bc.GEOM["array_y_local_m"])[np.array(bc.GEOM["state_map"]["C2"])-1]
    f=5.23e9+np.arange(-58,59,4)*312500.
    rows=[]
    for u,v in [(18,-48),(24,-36),(48,-18),(54,-12),(60,-9)]:
        sim=np.abs(np.exp(2j*np.pi*f[:,None]*y[None,:]*
                    (np.sin(np.radians(u))-np.sin(np.radians(v)))/299792458.).mean(axis=1))
        rows.append(dict(positive_deg=u,alternative_deg=v,min=sim.min(),median=np.median(sim),max=sim.max()))
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/"C2_geometric_ambiguity.csv",index=False,encoding="utf-8-sig")
    return df


def make_figures(contrast, angles):
    fig,ax=plt.subplots(figsize=(9.8,4.7),layout="constrained")
    for n,(site,color) in enumerate(zip(SITES,COLORS)):
        sub=contrast.query('point==@site and metric=="covariance_distance"').set_index("state").reindex(bc.STATES)
        ax.bar(np.arange(6)+(n-1)*.25,sub.ratio,width=.23,label=NAMES[site],color=color)
    ax.axhline(1,color="#363636",ls="--",lw=1.2,label="两次空场自身波动")
    ax.set_xticks(np.arange(6),bc.STATES)
    ax.set(ylabel="较小的有人对空场差异 / 两次空场差异",ylim=(0,3.2))
    ax.grid(axis="y",alpha=.16)
    ax.legend(ncol=4,fontsize=9,loc="upper left")
    ax.set_title("相同 C 点两次测量与 D 点｜均以各自前后空场作对照")
    fig.savefig(ROOT/"CD_出现离开与C重复性.png",dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(12.3,4.3),sharey=True,layout="constrained")
    for ax,site in zip(axs,SITES):
        row=angles[(angles.point==site)&(angles.state=="G3")].iloc[0]
        with np.load(bc.FOLDERS[site]/"exploratory_empirical_spectra.npz") as sp:
            for r,color,label in zip([1,2,3],COLORS,["前空场","人在场","后空场"]):
                ax.plot(sp["angles"],10*np.log10(np.maximum(sp[f"G3_R{r}"],1e-10)),c=color,lw=1.6,label=label)
        ax.axvline(row.theory_deg,ls=":",c="#523653",lw=2,label="子阵理论角")
        ax.plot(row.per_tone_r2,0,"o",ms=5,c=COLORS[1])
        ax.set(xlim=(-80,80),ylim=(-12,.6),xticks=[-60,-30,0,30,60],xlabel="谱角度参数 / °")
        ax.set_title(f"{NAMES[site]}　理论 {row.theory_deg:+.1f}°　R2 {row.per_tone_r2:+.0f}°",fontsize=11)
        ax.grid(alpha=.16)
    axs[0].set_ylabel("当条谱相对峰值 / dB")
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",ncol=4,fontsize=9,frameon=False)
    fig.suptitle("G3｜各批使用本批 R1 逐频参考，沿用相同选峰规则",fontsize=13)
    fig.savefig(ROOT/"G3_旧C新C与D.png",dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11.9,4.6),layout="constrained")
    site="D"; gt,_=bc.geometry(site)
    with np.load(bc.FOLDERS[site]/"angular_spectra.npz") as raw:
        for r,color,label in zip([1,2,3],COLORS,["R1 前空场","R2 人在 D","R3 后空场"]):
            curve=raw[f"G2_R{r}_raw"][0]
            axs[0].plot(raw["angles"],10*np.log10(np.maximum(curve,1e-10)),c=color,lw=1.7,label=label)
        axs[0].axvline(gt["G2"],c="#513854",ls=":",lw=2,label="D 的 G2 理论角")
    axs[0].set(xlim=(-90,90),xticks=[-90,-45,0,45,90],xlabel="角度参数 / °",ylabel="当条谱相对峰值 / dB",
               title="D · G2 实测三轮原始空间谱")
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    # Compare the two peaks within each reference model; ratios across models have different denominators.
    points=bc.read("D","exploratory_empirical_bg_peaks").query('stable and test==2 and state=="C2"')
    for n,(ref,color) in enumerate([(1,COLORS[1]),(3,COLORS[2])]):
        data=points[points.reference==ref].set_index("angle")
        values=[data.loc[v,"ratio"] for v in [-18,48]]
        bars=axs[1].bar(np.arange(2)+(n-.5)*.32,values,width=.30,color=color,label=f"用 R{ref} 作背景")
        axs[1].bar_label(bars,fmt="%.2f",fontsize=9,padding=3)
    axs[1].set_xticks([0,1],["−18° 候选","+48° 候选"])
    axs[1].set(ylabel="同参考模型内的白化功率比分数",ylim=(0,42),title="D · C2 两个方向均获近等分数")
    axs[1].grid(axis="y",alpha=.16);axs[1].legend(fontsize=8)
    fig.suptitle("D 点接近真值的线索与替代解释",fontsize=13)
    fig.savefig(ROOT/"D点_G2原始谱与C2双峰.png",dpi=180);plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(11.2,6.5),sharex=True,sharey=True,layout="constrained")
    for row,site in zip(axs,["C","D"]):
        with np.load(bc.FOLDERS[site]/"joint_spectra.npz") as sp:
            for ax,r,role in zip(row,[1,2,3],["前空场","人在场","后空场"]):
                curve=sp[f"G2_R{r}"];i,j=np.unravel_index(curve.argmax(),curve.shape)
                im=ax.pcolormesh(sp["tau_ns"],sp["angles"],10*np.log10(np.maximum(curve,1e-10)),
                                 shading="auto",cmap="viridis",vmin=-8,vmax=0)
                ax.plot(sp["tau_ns"][j],sp["angles"][i],"+",color="white",ms=9,mew=1.5)
                ax.set_title(f"{site} · {role}　{sp['angles'][i]:+.0f}°",fontsize=10)
        row[0].set_ylabel("未校准角度参数 / °")
    for ax in axs[-1]:ax.set_xlabel("未校准时延参数 / ns")
    fig.colorbar(im,ax=axs.ravel().tolist(),label="当条谱相对峰值 / dB",shrink=.85)
    fig.suptitle("G2 三阵元基线｜C、D 两批原始二维 MUSIC，白色标记为窗口最高峰",fontsize=13)
    fig.savefig(ROOT/"G2_CD三轮AoA_ToF.png",dpi=180);plt.close(fig)


def report():
    registry=json.loads((ROOT/"blind_registry.json").read_text(encoding="utf8"))
    assert all(bc.digest(ROOT/k)==v for k,v in registry.items())
    contrast=bc.contrast_summary()
    angles=bc.angle_summary()
    matched=two_reference_candidates()
    alias=alias_control()
    make_figures(contrast,angles)
    qc_rows=[];comparison=[];g3=[];allstates=[];cal=[];bg=[]
    for site in SITES:
        data=angles[angles.point==site]
        row=data[data.state=="G3"].iloc[0]
        g3.append(bc.md([NAMES[site],f"{row.theory_deg:+.2f}",f"{row.per_tone_r1:+.0f} / {row.per_tone_r2:+.0f} / {row.per_tone_r3:+.0f}",
                         f"{row.per_tone_abs_difference:.2f}",row.pair_p90_deg,f"{row.supported_pairs:.0f}/2"]))
        runs=bc.read(site,"raw_runs");q=bc.read(site,"state_quality");shared=bc.read(site,"shared_pair_check")
        qc_rows.append(bc.md([NAMES[site],runs.phy.iloc[0],runs.ifindex.iloc[0],
                              " / ".join(str(v) for v in runs.target_frames),f"{q.guarded_packets.min()}–{q.guarded_packets.max()}",
                              f"{shared.median_relative_phase_deg.min():.2f}–{shared.median_relative_phase_deg.max():.2f}°"]))
        if site=="C_previous":continue
        for v in contrast.query('point==@site and metric=="covariance_distance"').itertuples():
            comparison.append(bc.md([site,v.state,f"{v.empty_drift:.3f}",f"{v.appeared:.3f}",f"{v.departed:.3f}",
                                     f"{v.ratio:.3f}",f"{v.passing_quarters}/4"]))
        for v in data.itertuples():
            j="—" if pd.isna(v.joint_r2) else f"{v.joint_r1:+.0f} / {v.joint_r2:+.0f} / {v.joint_r3:+.0f}"
            allstates.append(bc.md([site,v.state,f"{v.theory_deg:+.2f}",f"{v.raw_r1:+.0f} / {v.raw_r2:+.0f} / {v.raw_r3:+.0f}",
                                    f"{v.per_tone_r1:+.0f} / {v.per_tone_r2:+.0f} / {v.per_tone_r3:+.0f}",j]))
        regular=bc.read(site,"ref_run3_validation");emp=bc.read(site,"empirical_repeatability")
        cal.append(bc.md([site,int(regular.pass_pair.sum()),int(emp.pass_pair.sum()),"、".join(emp[emp.pass_pair].apply(lambda v:f"{v.state} 链{v.chain}",axis=1))]))
        counts=bc.read(site,"background_tests")
        bg.append(bc.md([site,*[int(counts[(counts.reference==r)&(counts.test==t)].stable_peaks.sum()) for r,t in [(1,2),(1,3),(3,2),(3,1)]]]))
    prose=[
        "# Real New C 与 Real New D 实测分析",
        "",
        "同一 C 点在第二次采集中仍能区分有人段与前后空场；D 点也有六状态的配对变化。G3 的逐频参考主峰在两点均靠近 0°，无法随人体真实方位移动。D 出现了值得再检验的 G2 +54° 和 C2 +48° 候选，但有频段分歧及稀疏阵列方向歧义。当前能够报告的是配对变化和待复验的谱峰。",
        "",
        "## 采集条件与样本",
        "",
        "操作者确认橙色 C=(0.80,−1.20)、D=(2.40,−1.60) 为本次站位，run1 无人、run2 静止人体、run3 再次无人。图中粉色 ±45° 另有其坐标，本分析只对橙色站位评分。按已确认布置，TX=(0,0)，RX 阵列中心=(3.65,0)，图上 (1.64,0) 是测线与基线的交点。同频点、同包三链相位按保留的链重排字段恢复，每包用总体幅度范数归一化。",
        "",
        "| 批次 | PHY | ifindex | 三轮 10 秒窗口目标包数 | 去除边界后每轮每状态包数 | 共享 6/8 相位残差各轮中位数 |",
        "|---|---|---:|---|---:|---:|",*qc_rows,
        "",
        "两份新清单的 72 个条目核验通过；两批合计 6 个新的 CSI 文件，含 HT40 的 30 个频率组，每个点三轮内的 PHY、ifindex 和 boot 标识相同，完整驻留访问均有有效包。新批原始文件与 A、B、上次 C 的六状态记录已排重。上次 C、本次 C、本次 D 分别独立初始化，参考和复数 CSI 按批处理。一次进场对应一个有人窗口。",
        "",
        "## 人在场时的变化是否超过前后空场自身波动",
        "",
        "以同频三链协方差的相对距离 2‖U−V‖/(‖U‖+‖V‖) 比较。下表的比值取两次有人/空场距离中较小的一次，除以前后空场距离；一块为 2.5 秒。分块和六状态共用同一个进出过程。",
        "",
        "| 点 | 状态 | 前后空场 | 入场差异 | 离场差异 | 比值 | 超过空场的时间块 |",
        "|---|---|---:|---:|---:|---:|---:|",*comparison,
        "",
        "上次 C 的六状态比值为 1.61–2.21，24 个分块比较均通过。本次 C 仍为 6/6 超过 1，范围 1.045–2.858；其中 G1 仅 1/4 个时间块，其余五组全部通过。D 六组比值 1.40–2.35，全部 24 个分块通过。改用 5 ms 切换保护间隔，全窗口判断相同。幅度形状的较小有人/空场差异也在两点的全部六组超过前后空场差异。反射与人体遮挡均可产生同类变化。",
        "",
        "![配对变化](CD_出现离开与C重复性.png)",
        "",
        "## 用上次固定的选峰规则检验 G3",
        "",
        "每批只用 R1 前 5 秒建立逐频点有效参考。扫描完整前半平面，原始峰和候选保存后再与真值对照。R3 同方向留出标准为每对相对链至少 80% 频率达到支持要求，且相位 P90 ≤15°。",
        "",
        "| 点 | G3 理论角/° | 前空 / 有人 / 后空峰/° | 有人峰事后偏差/° | 两链对留出 P90/° | 达标链对 |",
        "|---|---:|---|---:|---|---:|",*g3,
        "",
        "C 重测的 G3 有人峰从上次 −3°变为 −6°，仍偏向理论方向的另一侧。D 的四个时间块都选 −3°。由本次 C 到 D，理论 G3 方向应增加约 29.17°，实测只增加 3°。站定段的时间重复性支持谱峰存在，但未验证其为人体方向。每个新点的 G3 两条链均未通过后空场参考复核。",
        "",
        "![G3 方向](G3_旧C新C与D.png)",
        "",
        "## 三阵元基线、各子阵及四组融合",
        "",
        "| 点 | 状态 | 子阵理论角/° | 原始空间峰 R1/R2/R3 | 逐频参考峰 R1/R2/R3 | 原始二维峰 R1/R2/R3 |",
        "|---|---|---:|---|---|---|",*allstates,
        "",
        "G2 是先前指定的三阵元基线。G fusion 是四个 G 子阵各自归一化后的非相干等权融合。D 的原始 G2 有人峰为 +54°，与其理论 +51.26°相差 2.74°；四组非相干融合的原始有人峰 +48° 与阵列中心 +52.00°相差 4.00°。C 的相应原始峰为 0°、+3°。使用 R1 逐频参考后，两点的 G2 及四组融合的有人峰均在 0°。因此这些读数不能证明完整十二阵元的相干增益。",
        "",
        "| 点 | 常数/线性参考通过链对（共 12） | 逐频参考通过链对（共 12） | 逐频通过项 |",
        "|---|---:|---:|---|",*cal,
        "",
        "D 的 G2 是本次唯一两条链都通过逐频同方向复核的子阵，两条链的相位 P90 为 8.75° 和 14.64°。C2 的两个链对在 C 和 D 都未通过。共享天线 6/8 的局部相位关系较稳定，但完整阵列仍缺各支路的可靠固定响应和独立非零方向验证。",
        "",
        "## 哪些近真值候选值得保留",
        "",
        "1. D 的 G2 原始主峰为 −36°→+54°→−36°。R2 的低、高半频主峰分别为 +66°、−27°，+54° 全频峰未通过既定的两频段匹配。后空场已有 +48° 次峰，仅比当轮最大峰低 0.43 dB。原始二维谱的 D/G2 主峰为 −42°→+57°→+54°；后空场也保有接近真值的峰。",
        "2. 对 C2，采用逐频参考和正则化协方差白化，并同样变换导向矢量。以 R1 和 R3 分别作背景，C 有人的正向候选是 +18°、+24°，与 C2 理论 +23.55°相差 5.55°、0.45°。D 两种背景都选出 +48° 分支，距理论 +52.72°为 4.72°。这几条局部峰均满足两个半频支持及至少三个时间块的规则。对称检查两个无人窗口时，在这些方向 ±9° 内没有同规则的稳定 C2 假峰。两种背景共享同一个 R2，不构成两次独立人体重复。",
        "3. 正角度候选仍有竞争解释。D 的两个背景也都支持 −18°。以 R1 作背景，−18°/+48° 的白化分数约 35.20/35.42；以 R3 作背景约 8.20/8.22。C2 的天线 10/8/6 间距 56 mm；按实际频率计算，这两个方向的理想导向矢量相似度中位数约 0.990。C 的 +18° 同样有 −48° 近别名，+24° 同样有 −36° 近别名。理论相似度是几何量，不代表人体识别概率。",
        "4. 作为同点复现检查，上次 C 在同一逐频参考白化规则下，跨前/后参考均保留的正向 C2 峰为 +42°/+51°。本次的 +18°/+24° 更贴近 C，但上次没有复现这条近真值正向分支。下列候选表列出所有匹配，包括错误方向和空场也能解释的情形。不能用已知站位在正、负两支之间选答案。",
        "",
        "![D 的候选与歧义](D点_G2原始谱与C2双峰.png)",
        "",
        "## 空场对照、AoA–ToF 与结论边界",
        "",
        "| 点 | R1参考→有人稳定峰 | R1参考→后空场稳定峰 | R3参考→有人稳定峰 | R3参考→前空场稳定峰 |",
        "|---|---:|---:|---:|---:|",*bg,
        "",
        "计数来自未经初相校正的白化搜索，合并六状态且保留角度边界。每批只有一个完整的空场负对照，不把峰数换算为误报概率。C 的 C1 原始二维有人主峰由上次 −6° 变为本次 +3°；D 为 −3°。名义双基地超额时延，C 为 2.951 ns、D 为 4.219 ns；实测谱轴尚未获得同记录的有效直达径零点。物理 ToF、二维位置误差和人体 AoA MAE 均保留为待验证。",
        "",
        "![G2 二维谱](G2_CD三轮AoA_ToF.png)",
        "",
        "对论文可报告：C 在两次独立接收机初始化下都观察到与人体在场对应的信道变化，D 也有完整的出现/恢复证据；G3 最大谱峰不能跨点估计人体方向。优先复验 D 的 G2 正角度响应与 C2 的 +48°/−18° 两支，在独立进场轮次和不重叠 G 子阵上按预设规则消歧。3 根与完整 12 根的物理性能比较仍缺验证。",
        "",
        "复算顺序为 process.py、report.py，原始文件只读。方法沿用 New A 和 New B/C，B/C 曾用的阈值与峰规则保持固定。本次新目录与历史输出隔离，验算 SHA256。输入时钟的文件名日期仅作文件标识。",
        "",
        "[原始记录与标签](record_index.csv) · [各状态配对差异](paired_comparisons.csv) · [各子阵盲选峰与事后几何对照](directions_by_state.csv) · [全部局部候选](all_candidates_vs_geometry.csv) · [前后空场同时支持的候选及空场否决项](two_reference_candidates.csv) · [C2 阵列几何歧义](C2_geometric_ambiguity.csv) · [盲选文件哈希](blind_registry.json)。"
    ]
    file=ROOT/"Real_New_C与D_配对实测报告.md"
    file.write_text("\n".join(prose)+"\n",encoding="utf8")
    links=re.findall(r'\]\(([^)]*)\)',file.read_text(encoding="utf8"))
    assert all((ROOT/link).is_file() for link in links)
    summary={"new_raw_files":6,"confirmed_points":["C","D"],"input_geometry":"orange C/D",
             "previous_C_only_as_repeat_comparison":True,
             "total_formal_target_frames":int(sum(bc.read(s,"raw_runs").target_frames.sum() for s in ["C","D"])),
             "total_guarded_packets":int(sum(bc.read(s,"state_quality").guarded_packets.sum() for s in ["C","D"])),
             "independent_new_occupied_segments":2,"C2_48_minus18_similarity":float(alias.query('positive_deg==48').iloc[0]['median']),
             "candidate_matches_stored":len(matched),"human_AoA_MAE":None,"physical_ToF_error":None,"position_error":None,
             "coherent_12_gain":None,"blind_registry_entries_verified":len(registry)}
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(summary)


if __name__=="__main__":
    report()
