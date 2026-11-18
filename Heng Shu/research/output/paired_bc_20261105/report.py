"""Score full-range, frozen outputs against the corrected geometry, and draw figures."""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
FOLDERS={"A":ROOT.parent/"new_a_paired_20261105","B":ROOT/"B","C":ROOT/"C"}
GEOM=json.loads((ROOT.parent/"static_human_aoa_validation_20261101"/"geometry.json").read_text(encoding="utf8"))
POINTS={"A":(.80,+.60),"B":(2.80,+1.20),"C":(.80,-1.20)}
STATES=list(GEOM["state_map"])
COLORS=["#466f99","#bd5420","#178372"]
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})


def read(point, stem):
    return pd.read_csv(FOLDERS[point]/f"{stem}.csv")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md(fields):
    return "| "+" | ".join(map(str,fields))+" |"


def geometry(point):
    x,y=POINTS[point]
    dx=GEOM["global_rx"][0]-x
    array_y=np.array(GEOM["array_y_local_m"])
    gt={state:np.degrees(np.arctan2(-y-np.mean(array_y[np.array(ants)-1]),dx))
        for state,ants in GEOM["state_map"].items()}
    gt["G fusion"]=np.degrees(np.arctan2(-y,dx))
    delay=(np.hypot(x,y)+np.hypot(dx,y)-3.65)/.299792458
    return gt,delay


def contrast_summary():
    rows=[]
    for point in FOLDERS:
        full=read(point,"paired_changes")
        split=read(point,"paired_quarters")
        for metric in ["covariance_distance","amplitude_shape_distance","phase_median_deg"]:
            c=full.query('guard=="primary"').pivot(index="state",columns="pair",values=metric)
            deep=full.query('guard=="guard5ms"').pivot(index="state",columns="pair",values=metric)
            q=split.pivot(index=["state","quarter"],columns="pair",values=metric)
            n=(q[[12,23]].min(axis=1)>q[13]).groupby("state").sum()
            for s in STATES:
                ratio=min(c.loc[s,12],c.loc[s,23])/c.loc[s,13]
                rows.append(dict(point=point,state=s,metric=metric,empty_drift=c.loc[s,13],
                                 appeared=c.loc[s,12],departed=c.loc[s,23],ratio=ratio,
                                 ratio_guard5ms=min(deep.loc[s,12],deep.loc[s,23])/deep.loc[s,13],
                                 passing_quarters=int(n[s]),all_four_quarters_pass=int(n[s])==4))
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/"paired_comparisons.csv",index=False,encoding="utf-8-sig")
    return out


def angle_summary():
    rows=[]
    candidates=[]
    bg_tables=[]
    ident=[]
    for point in FOLDERS:
        gt,delay=geometry(point)
        emp=read(point,"exploratory_empirical_angular_peaks")
        raw=read(point,"blind_angular_maxima").query('mode=="raw"')
        joint=read(point,"blind_joint_maxima")
        pairs=read(point,"empirical_repeatability")
        for state in [*STATES,"G fusion"]:
            e=emp[emp.state==state].set_index("run")
            rr=raw[raw.state==state].set_index("run")
            j=joint[joint.state==state].set_index("run")
            hold=pairs[pairs.state==state]
            rows.append(dict(point=point,state=state,theory_deg=gt[state],
                             per_tone_r1=e.loc[1,"dominant_deg"],per_tone_r2=e.loc[2,"dominant_deg"],
                             per_tone_r3=e.loc[3,"dominant_deg"],per_tone_abs_difference=abs(e.loc[2,"dominant_deg"]-gt[state]),
                             supported_pairs=int(hold.pass_pair.sum()) if len(hold) else None,
                             pair_p90_deg="/".join(f"{x:.2f}" for x in hold.p90_deg),
                             per_tone_r2_low=e.loc[2,"low_band_dominant_deg"],per_tone_r2_high=e.loc[2,"high_band_dominant_deg"],
                             per_tone_r2_quarters=e.loc[2,"quarter_dominant_deg"],
                             raw_r1=rr.loc[1,"peak_deg"],raw_r2=rr.loc[2,"peak_deg"],raw_r3=rr.loc[3,"peak_deg"],
                             joint_r1=j.loc[1,"angle_parameter_deg"] if len(j) else None,
                             joint_r2=j.loc[2,"angle_parameter_deg"] if len(j) else None,
                             joint_r3=j.loc[3,"angle_parameter_deg"] if len(j) else None,
                             diagnostic_joint_delay_ns=j.loc[2,"uncalibrated_delay_ns"] if len(j) else None,
                             theoretical_excess_ns=delay,physical_human_path_validated=False))
        modes=[("local","blind_local_peaks","peak_deg"),("whitened","blind_background_candidates","angle_parameter_deg"),
               ("empirical_whitened","exploratory_empirical_bg_peaks","angle")]
        for mode,name,col in modes:
            df=read(point,name)
            df=df.assign(point=point,analysis=mode,theory_deg=df.state.map(gt),diagnostic_angle=df[col])
            df["retrospective_difference_deg"]=(df.diagnostic_angle-df.theory_deg).abs()
            candidates.append(df)
        for (ref,test),sub in read(point,"background_tests").groupby(["reference","test"]):
            bg_tables.append(dict(point=point,reference=ref,test=test,stable_peaks=int(sub.stable_peaks.sum())))
        rr=read(point,"raw_runs")
        for row in rr.itertuples():
            ident.append(dict(point=point,run=row.run,condition=["empty_pre","occupied","empty_post"][row.run-1],
                              phy=row.phy,ifindex=row.ifindex,sha256=row.sha256,packets=row.target_frames,
                              pps=row.pps,continuity_gate=row.continuity_gate))
    pd.concat(candidates,ignore_index=True).to_csv(ROOT/"all_candidates_vs_geometry.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(bg_tables).to_csv(ROOT/"background_controls.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(ident).to_csv(ROOT/"record_index.csv",index=False,encoding="utf-8-sig")
    results=pd.DataFrame(rows)
    results.to_csv(ROOT/"directions_by_state.csv",index=False,encoding="utf-8-sig")
    return results


def figures(contrast, angle):
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.45),sharey=True,layout="constrained")
    for ax,tag in zip(axes,["B","C"]):
        ss=contrast[(contrast.point==tag)&(contrast.metric=="covariance_distance")].set_index("state").reindex(STATES)
        for k,(col,name,color) in enumerate([("empty_drift","前后空场","#91a0ac"),
                                            ("appeared","前空场与有人",COLORS[1]),("departed","有人与后空场",COLORS[2])]):
            ax.bar(np.arange(6)+(k-1)*.25,ss[col],width=.23,color=color,label=name)
        ax.set_xticks(np.arange(6),STATES)
        ax.set_title(f"{tag} 点");ax.grid(axis="y",alpha=.17)
    axes[0].set_ylabel("空间协方差相对距离")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False,fontsize=9)
    fig.suptitle("空场 → 有人 → 空场｜分别使用每个点自己的三轮数据",fontsize=13)
    fig.savefig(ROOT/"BC_人体出现与离开.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.25),sharey=True,layout="constrained")
    for ax,tag in zip(axes,["A","B","C"]):
        g=angle[(angle.point==tag)&(angle.state=="G3")].iloc[0]
        with np.load(FOLDERS[tag]/"exploratory_empirical_spectra.npz") as ss:
            for run,color,label in zip([1,2,3],COLORS,["前空场","人在场","后空场"]):
                curve=ss[f"G3_R{run}"]
                ax.plot(ss["angles"],10*np.log10(np.maximum(curve,1e-8)),label=label,color=color,lw=1.55)
        ax.axvline(g.theory_deg,ls=":",color="#492952",lw=2,label="该点 G3 理论角")
        ax.plot(g.per_tone_r2,0,marker="o",color=COLORS[1],ms=5)
        ax.set(xlim=(-80,80),xticks=[-60,-30,0,30,60],ylim=(-16,.7),xlabel="G3 谱角度参数 / °")
        ax.set_title(f"{tag}　理论 {g.theory_deg:+.1f}°　有人峰 {g.per_tone_r2:+.0f}°",fontsize=11)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("各曲线自身峰值归一化 / dB")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False,fontsize=9)
    fig.suptitle("G3 逐频点参考检验｜A 为已有发现，B、C 沿用同一条选峰规则",fontsize=13)
    fig.savefig(ROOT/"G3_跨站位方向检验.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11.2,6.7),sharey=True,sharex=True,layout="constrained")
    for row,tag in zip(axes,["B","C"]):
        with np.load(FOLDERS[tag]/"joint_spectra.npz") as z:
            for r,ax,stage in zip([1,2,3],row,["前空场","人在场","后空场"]):
                m=z[f"C1_R{r}"];i,j=np.unravel_index(m.argmax(),m.shape)
                im=ax.pcolormesh(z["tau_ns"],z["angles"],10*np.log10(np.maximum(m,1e-10)),
                                 shading="auto",cmap="viridis",vmin=-8,vmax=0)
                ax.plot(z["tau_ns"][j],z["angles"][i],"+",ms=9,mew=1.5,color="white")
                ax.set_title(f"{tag} · {stage}　{z['angles'][i]:+.0f}°",fontsize=10)
        row[0].set_ylabel("未校准角度参数 / °")
    for ax in axes[-1]:
        ax.set_xlabel("未校准时延参数 / ns")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="各窗口相对谱峰 / dB",shrink=.88)
    fig.suptitle("C1 原始二维 MUSIC｜三阶模型，白色加号为当轮最高峰",fontsize=13)
    fig.savefig(ROOT/"C1_BC二维谱.png",dpi=180)
    plt.close(fig)


def report():
    registry=json.loads((ROOT/"blind_registry.json").read_text(encoding="utf8"))
    assert all(digest(ROOT/k)==v for k,v in registry.items())
    cs=contrast_summary()
    angles=angle_summary()
    figures(cs,angles)
    pos_rows=[]; change_rows=[]; full_rows=[]; qc_rows=[]; cal_rows=[]; bg_rows=[]
    for point in ["A","B","C"]:
        rr=angles[(angles.point==point)&(angles.state=="G3")].iloc[0]
        pos_rows.append(md([point,f"{rr.theory_deg:+.2f}",f"{rr.per_tone_r1:+.0f} / {rr.per_tone_r2:+.0f} / {rr.per_tone_r3:+.0f}",
                            f"{rr.per_tone_abs_difference:.2f}",f"{rr.pair_p90_deg}",f"{rr.supported_pairs:.0f}/2"]))
        if point=="A":
            continue
        q=read(point,"state_quality"); runs=read(point,"raw_runs"); sh=read(point,"shared_pair_check")
        qc_rows.append(md([point,f"{runs.phy.iloc[0]} / {runs.ifindex.iloc[0]}",
                           " / ".join(str(x) for x in runs.target_frames),f"{q.guarded_packets.min()}–{q.guarded_packets.max()}",
                           f"{sh.median_relative_phase_deg.min():.2f}–{sh.median_relative_phase_deg.max():.2f}°"]))
        const=read(point,"ref_run3_validation");emp=read(point,"empirical_repeatability")
        cal_rows.append(md([point,int(const.pass_pair.sum()),int(emp.pass_pair.sum()),"、".join(emp[emp.pass_pair].apply(lambda r:f"{r.state} 链{r.chain}",axis=1))]))
        for row in cs.query('point==@point and metric=="covariance_distance"').itertuples():
            change_rows.append(md([point,row.state,f"{row.empty_drift:.3f}",f"{row.appeared:.3f}",f"{row.departed:.3f}",
                                   f"{row.ratio:.3f}",f"{row.passing_quarters}/4"]))
        for row in angles[angles.point==point].itertuples():
            joint="—" if np.isnan(row.joint_r2) else f"{row.joint_r1:+.0f} / {row.joint_r2:+.0f} / {row.joint_r3:+.0f}"
            full_rows.append(md([point,row.state,f"{row.theory_deg:+.2f}",
                                 f"{row.raw_r1:+.0f} / {row.raw_r2:+.0f} / {row.raw_r3:+.0f}",
                                 f"{row.per_tone_r1:+.0f} / {row.per_tone_r2:+.0f} / {row.per_tone_r3:+.0f}",joint]))
        bg=read(point,"background_tests")
        bg_rows.append(md([point,*[int(bg[(bg.reference==r)&(bg.test==t)].stable_peaks.sum()) for r,t in [(1,2),(1,3),(3,2),(3,1)]]]))
    prose=[
        "# B、C 两点配对复验",
        "",
        "B、C 各有一批完整的空场→有人→空场记录。以 A 已保存的算法和阈值独立处理两批。C 的六个状态都支持与人在场有关的变化；B 的结果受较大的前后空场差异限制，G4 最清楚，C1 也有较弱的支持。A 点提出的 G3 方向线索在两个新点上均未通过验证。",
        "",
        "## 原始数据和坐标",
        "",
        "左侧 TX=(0,0)，右侧 RX 阵列中心=(3.65,0)。图上 (1.64,0) 是地面测量线交点。采用 B=(2.80,+1.20)、C=(0.80,−1.20)，人体标签取自操作者。文件日期只作来源标识。安装几何采用已确认的 28 mm 线阵和六状态映射。",
        "",
        "| 点 | 本批 RX 标识 | 三个 10 秒窗口目标包数 | 每轮每状态保护后包数 | 共享天线 6/8 周期相位残差，各轮中位数 |",
        "|---|---|---|---:|---:|",*qc_rows,
        "",
        "各批 36 个原始清单条目校验通过，全部 6 个新 CSI 文件不同，未与 A 或先前输入重复。HT40，控制频率 5240 MHz，中心频率 5230 MHz。三轮的 boot、PHY、ifindex 在各自批内一致，continuity_gate 均通过。所有完整驻留周期均有有效包。每点单独以本批 R1 建立参考，B、C 的固定相位不可作为共同参考池。每点只有一个独立有人窗口。",
        "",
        "## 有人时的变化是否超过前后空场自身波动",
        "",
        "使用逐包归一化、同频率组的三链空间协方差，计算相对距离 2‖U−V‖/(‖U‖+‖V‖)。比值取 R1↔R2 和 R2↔R3 中较小者，除以 R1↔R3。比值大于 1 只表示本次对照超过背景波动。每轮四个对应的 2.5 秒时间块用于检查持续性。",
        "",
        "| 点 | 状态 | 空场 R1↔R3 | 入场 R1↔R2 | 离场 R2↔R3 | 比值 | 通过时间块 |",
        "|---|---|---:|---:|---:|---:|",*change_rows,
        "",
        "B 的 G4 比值约 2.06，C1 约 1.20，两者在四块中均超过空场差异。G2 约 1.004，仅 2/4 块超过；其余三组全窗口比值小于 1。C 的比值为 1.61–2.21，六状态 24 个时间块比较全部通过。改用 5 ms 切换保护间隔，两个点的全窗口判断均不变。幅度和相位分项另存于 paired_comparisons.csv；这不是人体反射强度倍数或检测准确率。",
        "",
        "![空间变化](BC_人体出现与离开.png)",
        "",
        "## G3 方向的跨点复验",
        "",
        "主问题来自 A 的观察。每批均以 R1 前 5 秒拟合逐频点有效参考，按同频空间 MUSIC 在整个前半平面盲选最大峰。R3 专门检查同方向参考能否复现。每个状态两条相对链路分别要求可靠频率比例 ≥80%、P90 链间相位残差 ≤15°。选峰和输入划分沿用 A，B/C 无逐点偏置或真值挑峰。",
        "",
        "| 点 | G3 理论角/° | 前空场 / 有人 / 后空场峰/° | 有人峰与几何的偏差/° | 留出两链 P90 相位残差/° | 通过链对 |",
        "|---|---:|---|---:|---|---:|",*pos_rows,
        "",
        "B 的有人峰在两个半频段分别为 +9° / +3°，四个时间块为 +9° / +6° / +6° / +6°。C 的两半频为 −3° / 0°，四块为 0° / −3° / −3° / −3°，后空场主峰也在 −3°。按照子阵真实中心，B→C 理论方向改变 +77.27°，实测峰改变 −9°。两个新点的 G3 均有 0/2 条链对通过参考复核，不能把 A 的 5.08°事后偏差推广成总体 AoA 精度。",
        "",
        "![G3 跨点方向](G3_跨站位方向检验.png)",
        "",
        "## 各子阵与三根、十二根的比较",
        "",
        "| 点 | 子阵 | 理论角/° | 原始空间峰 R1/R2/R3 | 逐频参考峰 R1/R2/R3 | 原始二维峰 R1/R2/R3 |",
        "|---|---|---:|---|---|---|",*full_rows,
        "",
        "G fusion 是四个物理 G 子阵各自求谱、各谱归一化后等权非相干相加。固定三根主基线 G2 的逐频参考有人峰为 B −3°、C 0°；同窗四组融合为 B +3°、C 0°，仍集中在直达方向附近。共享 6/8 的局部相位残差约 1°，支持局部切换一致性。其余支路的有效固定响应、跨状态联合对齐和未拟合主动源方向检查尚未通过。物理人体 MAE、位置误差及完整十二阵元相干增益均缺少验证。",
        "",
        "| 点 | 常数或频率线性模型通过链对数（共 12） | 逐频参考通过链对数（共 12） | 通过的逐频相对链 |",
        "|---|---:|---:|---|",*cal_rows,
        "",
        "## 哪些近真值候选值得保留",
        "",
        "- B 的 C1 逐频参考主峰是 0°→−63°→0°。有人时两个半频段、四个时间块的主峰都是 −63°，与 C1 理论 −55.61°相差 7.39°。但其两条相对链的后空场相位残差 P90 为 45.43°、79.92°，而原始二维谱的主峰序列是 +42°→+39°→+45°。可保留 −63° 作为新一轮 B 的预设复验分支，但这次没有独立的方向标定。",
        "- B 的 G2 白化后可见 −54° 的稳定峰（距理论 1.61°），只在用 R3 作背景时出现。按原先指定的 R1 作背景时，有人段的稳定峰在 +15° 和 +90°。实验不能根据哪个参考更接近真值来挑选背景。",
        "- C 的 C2 用 R1 作背景时产生 +30° 候选，距理论 6.45°，换 R3 作背景只剩 −6°。C1 原始空间谱的 +15° 局部峰则三轮都在，空场时也稳定。都缺乏一致的人体特异证据。",
        "- A 的 C1 二维有人主峰是 −6°。C 的理论角已转正到 +22.11°，三阶模型的有人主峰仍是 −6°，且将阶数改为 2 或 4 都选 −69°。B 的三种阶数均选 +39°，与 B 理论方向在不同一侧。二维谱的时延轴尚未识别同记录直达零点，以下仅作谱形诊断。",
        "",
        "![二维谱](C1_BC二维谱.png)",
        "",
        "| 点 | R1参考→有人稳定峰数 | R1参考→后空场稳定峰数 | R3参考→有人稳定峰数 | R3参考→前空场稳定峰数 |",
        "|---|---:|---:|---:|---:|",*bg_rows,
        "",
        "计数含搜索边界，是一次试验上完整六状态的候选数量。每批仅一个留出空场，不能换算为误报概率。已有 A 用于形成待检假设；不能在观察 B、C 的结果后重选模型或子阵，再用同两点声称独立验证。",
        "",
        "## 下一步与复算",
        "",
        "若继续验证方向，建议在一次 RX 初始化中交错采 空→B→空→C→空→C→空→B→空，每次重新站位；记录门和附近人员活动。这样点位不会与初始化绑定，并可检查 B 的 C1 −63° 能否重复及参考能否过关。本轮记录无法区分 B 的空场漂移具体由门、其他运动或硬件状态造成。",
        "",
        "运行 process_batches.py 得到盲选表，再运行 report.py 加入几何。A 的数据沿用原输出，只读。[逐轮数据索引](record_index.csv) · [各状态配对距离](paired_comparisons.csv) · [各状态方向与实测几何对照](directions_by_state.csv) · [全部局部候选及事后对照](all_candidates_vs_geometry.csv) · [空场候选计数](background_controls.csv) · [盲选文件哈希](blind_registry.json) · [B 逐状态质检](B/state_quality.csv) · [C 逐状态质检](C/state_quality.csv)。"
    ]
    path=ROOT/"B与C_静态人体配对分析.md"
    path.write_text("\n".join(prose)+"\n",encoding="utf8")
    links=re.findall(r'\]\(([^)]*)\)',path.read_text(encoding="utf8"))
    assert all((ROOT/link).is_file() for link in links)
    stats={"heldout_points":["B","C"],"independent_occupied_segments":2,"geometry_join_after_freeze":True,
           "blind_registry_entries_verified":len(registry),"formal_target_frames":int(sum(read(k,"raw_runs").target_frames.sum() for k in ["B","C"])),
           "guarded_packets":int(sum(read(k,"state_quality").guarded_packets.sum() for k in ["B","C"])),
           "empirical_G3_reference_passed_batches":0,"validated_human_AoA_MAE":None,"physical_position_error":None,
           "coherent_12_gain":None}
    (ROOT/"summary.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    print(stats)


if __name__=="__main__":
    report()
