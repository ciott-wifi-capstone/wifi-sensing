"""Report the paired pilot and score frozen peak parameters against survey geometry."""
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
STATES=["G1","G2","G3","G4","C1","C2"]
BLUE,ORANGE,GREEN="#356090","#c25620","#17816f"
COLORS=[BLUE,ORANGE,GREEN]
LABELS=["R1 前空场","R2 人在 A","R3 后空场"]
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})


def table(name):
    return pd.read_csv(ROOT/f"{name}.csv")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mdrow(items):
    return "| "+" | ".join(map(str,items))+" |"


def contrast_plot(change, fractions):
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout="constrained")
    ax=axs[0]
    target=change.query('guard=="primary"').pivot(index="state",columns="pair",values="covariance_distance").reindex(STATES)
    for n,(pair,color,label) in enumerate([(13,"#8997a2","前空场与后空场"),(12,ORANGE,"前空场与有人"),(23,GREEN,"有人与后空场")]):
        ax.bar(np.arange(6)+(n-1)*.25,target[pair],width=.23,color=color,label=label)
    ax.set_xticks(np.arange(6),STATES)
    ax.set_ylabel("归一化空间协方差距离")
    ax.set_title("六个状态的出现 / 离开对照")
    ax.legend(fontsize=8)
    ax.grid(axis="y",alpha=.16)
    ax=axs[1]
    z=fractions.query("quarter==0").sort_values("run")
    for band,color,label in [("low",BLUE,"低半频段"),("all",ORANGE,"全频段"),("high",GREEN,"高半频段")]:
        power=z[f"fraction_{band}"].to_numpy()*100
        ax.plot(np.arange(3),power,marker="o",color=color,lw=2,label=label)
        if band=="all":
            for i,v in enumerate(power):
                ax.annotate(f"{v:.1f}%",(i,v),xytext=(0,8),textcoords="offset points",ha="center",color=color,fontsize=9)
    ax.set_xticks(np.arange(3),LABELS)
    ax.set_ylabel("占本频段总空间功率的比例 / %")
    ax.set_title("C1 在固定 −21° 的波束投影")
    ax.set(xlim=(-.25,2.25),ylim=(20,47))
    ax.grid(axis="y",alpha=.16)
    ax.legend(fontsize=8,loc="lower left")
    fig.suptitle("New A｜同一次 RX 初始化，空场 → 人在 A → 空场",fontsize=14)
    fig.savefig(ROOT/"空场_有人_空场_对照.png",dpi=175)
    plt.close(fig)


def direction_plot(gt):
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    with np.load(ROOT/"angular_spectra.npz") as raw,np.load(ROOT/"exploratory_empirical_spectra.npz") as empirical:
        for ax,state,source,template,title in [
            (axs[0],"C1",raw,"C1_R{}_raw","C1 原始空间谱"),
            (axs[1],"G3",empirical,"G3_R{}","G3 用前空场建立逐频点参考后")]:
            for r,color,label in zip([1,2,3],COLORS,LABELS):
                curve=source[template.format(r)]
                if curve.ndim==2:
                    curve=curve[0]
                ax.plot(source["angles"],10*np.log10(np.maximum(curve,1e-12)),lw=1.7,color=color,label=label)
            ax.axvline(gt[state],color="#48334e",ls=":",lw=2,label="A 点子阵理论角")
            if state=="C1":
                ax.axvline(-21,color=".45",ls="--",lw=1,label="旧 −21° 参数")
                ax.set(xlim=(-90,90),xticks=[-90,-45,0,45,90])
            else:
                ax.set(xlim=(-30,30),xticks=np.arange(-30,31,10))
            ax.set_xlabel("谱角度参数 / °")
            ax.set_ylabel("各曲线自身峰值归一化 / dB")
            ax.set_title(title);ax.grid(alpha=.18)
            ax.legend(fontsize=8,loc="lower right")
    fig.suptitle("A 点方向复验｜右图为探索性结果，仅放大 ±30° 显示",fontsize=13)
    fig.savefig(ROOT/"C1与G3_方向复验.png",dpi=175)
    plt.close(fig)


def joint_plot():
    fig,axs=plt.subplots(1,3,figsize=(11.6,4.2),sharex=True,sharey=True,layout="constrained")
    with np.load(ROOT/"joint_spectra.npz") as data:
        for r,ax,label in zip([1,2,3],axs,LABELS):
            z=data[f"C1_R{r}"]; i,j=np.unravel_index(z.argmax(),z.shape)
            im=ax.pcolormesh(data["tau_ns"],data["angles"],10*np.log10(np.maximum(z,1e-10)),
                             cmap="viridis",vmin=-8,vmax=0,shading="auto")
            ax.plot(data["tau_ns"][j],data["angles"][i],marker="+",color="white",ms=10,mew=1.4)
            ax.set_title(f"{label}　主峰 {data['angles'][i]:+.0f}°")
            ax.set_xlabel("未校准时延参数 / ns")
        axs[0].set_ylabel("未校准角度参数 / °")
        fig.colorbar(im,ax=axs.ravel().tolist(),label="各窗口相对谱峰 / dB")
    fig.suptitle("C1 二维 MUSIC｜同一子阵、三轮对照，白色加号为各自最大峰",fontsize=13)
    fig.savefig(ROOT/"C1_三轮AoA_ToF.png",dpi=175)
    plt.close(fig)


def report():
    frozen=json.loads((ROOT/"frozen_outputs.json").read_text(encoding="utf8"))
    assert all(sha(ROOT/n)==v for n,v in frozen.items())
    geom=json.loads((ROOT.parent/"static_human_aoa_validation_20261101"/"geometry.json").read_text(encoding="utf8"))
    dx=geom["global_rx"][0]-.8
    y=np.array(geom["array_y_local_m"])
    truth={s:np.degrees(np.arctan2(-.60-y[np.array(geom["state_map"][s])-1].mean(),dx)) for s in STATES}
    truth["G fusion"]=np.degrees(np.arctan2(-.60,dx))
    tof=(np.hypot(dx,.60)+1.-3.65)/.299792458
    raw,candidates=table("blind_angular_maxima"),table("blind_local_peaks")
    bg=table("blind_background_candidates")
    for frame,name,angle in [(candidates,"local_peaks_vs_geometry.csv","peak_deg"),(bg,"background_peaks_vs_geometry.csv","angle_parameter_deg")]:
        frame["theory_deg"]=frame.state.map(truth)
        frame["posthoc_difference_deg"]=(frame[angle]-frame.theory_deg).abs()
        frame["human_path_validated"]=False
        frame.to_csv(ROOT/name,index=False,encoding="utf-8-sig")
    change=table("paired_changes")
    fractions=table("C1_minus21_before_during_after")
    contrast_plot(change,fractions)
    direction_plot(truth)
    joint_plot()
    r=table("raw_runs"); qc=table("state_quality"); shared=table("shared_pair_check")
    held=table("ref_run3_validation"); empirical=table("empirical_repeatability")
    checks=table("manifest_checks")
    c=change.query('guard=="primary"').pivot(index="state",columns="pair",values="covariance_distance").reindex(STATES)
    change_rows=[mdrow([s,*[f"{c.loc[s,k]:.3f}" for k in (13,12,23)],
                        f"{min(c.loc[s,12],c.loc[s,23])/c.loc[s,13]:.2f}"]) for s in STATES]
    emp=table("exploratory_empirical_angular_peaks")
    joint=table("blind_joint_maxima")
    angle_rows=[]
    for s in [*STATES,"G fusion"]:
        seq=raw[(raw.state==s)&(raw["mode"]=="raw")].sort_values("run").peak_deg
        two=joint.query("state==@s and run==2").angle_parameter_deg
        epeak=emp.query("state==@s and run==2").dominant_deg.iloc[0]
        angle_rows.append(mdrow([s,f"{truth[s]:+.2f}"," / ".join(f"{v:+.0f}" for v in seq),
                                 f"{two.iloc[0]:+.0f}" if len(two) else "—",f"{epeak:+.0f}"]))
    fvals=fractions.query("quarter==0").sort_values("run").fraction_all.to_numpy()
    delta=table("C1_minus21_contrasts").query('quarter==0').set_index("band")
    stable=bg[bg.stable]; white_rows=[]
    for s in STATES:
        def peaks(ref,test):
            vv=stable[(stable.state==s)&(stable.reference==ref)&(stable.test==test)].angle_parameter_deg
            return "、".join(f"{v:+.0f}" for v in vv) if len(vv) else "—"
        white_rows.append(mdrow([s,peaks(1,2),peaks(1,3),peaks(3,2)]))
    paired_quarters=table("paired_quarters").pivot(index=["state","quarter"],columns="pair",values="covariance_distance")
    assert (paired_quarters[[12,23]].min(axis=1)>paired_quarters[13]).all()
    assert len(r)==3 and checks.ok.all() and qc.coverage.eq(1).all() and not r.matches_old.any()
    assert len(held)==len(empirical)==12
    g3=emp.query('state=="G3" and run==2').iloc[0]
    g3diff=abs(g3.dominant_deg-truth["G3"])
    g3_pass=empirical[empirical.pass_pair].groupby("state").size()
    checked=table("exploratory_C1_joint_time_order").query('run==2 and model_order==3 and quarter>0')
    text=[
        "# New A｜同一次接收机初始化下的 A 点配对分析",
        "",
        "操作标签为 R1 无人、R2 一人静止站 A、R3 再次无人。本次有一次完整的出现和离开过程。六个状态均观察到 R2 偏离前后空场，R3 比 R2 更接近 R1。固定 C1 −21° 的全频段投影没有在 R2 增强。G3 的探索性结果为本次可优先复验的方向线索。",
        "",
        "## 采集与独立性",
        "",
        "原始 config 的通用名称为 kong_boot01；场景标签取自操作者。本批只在整批准备时卸载/载入驱动一次；三轮使用同一 boot_id、phy5、ifindex 18，轮内前后身份一致，continuity_gate 均为 true。批末有 monitor 清理。各轮 logger 重启、ESP 执行逻辑切换复位，核对 READY 轴并重新对齐。未使用上一次 phy3/phy4 的 CSI 或校准系数。文件日期只作来源标识。",
        "",
        f"原始清单 {len(checks)} 项校验通过，3 个 CSI 哈希各不相同，也未与上批或历史清单重复。HT40−，控制/中心频率 5240/5230 MHz，三接收链、30 个频率组。10 秒窗口内目标包依次为 {'、'.join(map(str,r.target_frames))}，合计 {r.target_frames.sum()}。实测窗口间隔分别为 {(r.start_ns.iloc[1]-r.end_ns.iloc[0])/1e9:.2f}、{(r.start_ns.iloc[2]-r.end_ns.iloc[1])/1e9:.2f} 秒；移动过渡没有作为静止样本。每轮每状态去除切换边界后为 {qc.guarded_packets.min()}–{qc.guarded_packets.max()} 包，合计 {qc.guarded_packets.sum()} 包，全部完整驻留访问有数据。",
        "",
        "## 先比较人进入前后，再看方向",
        "",
        "每包除以整体幅度范数，同频点取三链空间协方差，用 2‖Rᵢ−Rⱼ‖/(‖Rᵢ‖+‖Rⱼ‖) 计算相对距离。比值取两次有人/空场距离中的较小值，除以前后空场距离。它是这一次试验的描述量。",
        "",
        "| 状态 | R1↔R3 空场 | R1↔R2 进入 | R2↔R3 离开 | 比值 |",
        "|---|---:|---:|---:|---:|",*change_rows,
        "",
        "每轮按真实时间分四块，24 个状态/对应时间块比较均保持这一大小关系。将切换保护间隔增至 5 ms，六状态的结论相同。归一化幅度形状和可靠频点上的链间相位差也分别在六状态同向。三个记录只产生一次受试者进出，时间块共享这一次实验，不能据此估计独立检出率或人群定位精度。反射和人体遮挡都会改变静态信道。",
        "",
        "![配对变化和固定方向投影](空场_有人_空场_对照.png)",
        "",
        "## 事先指定的 C1 −21° 分支",
        "",
        f"三个原始 C1 谱在 −21°±9°内都有稳定的 −24° 局部峰。将 −21° 固定下来，在同一尺度计算其波束投影占频段总空间功率的比例，前/人/后分别为 {fvals[0]*100:.1f}%、{fvals[1]*100:.1f}%、{fvals[2]*100:.1f}%。R2 相对 R1 和 R3 分别为 {delta.loc['all','run2_minus_run1_db']:+.2f} 和 {delta.loc['all','run2_minus_run3_db']:+.2f} dB。低半频相对 R1 为 {delta.loc['low','run2_minus_run1_db']:+.2f} dB，高半频为 {delta.loc['high','run2_minus_run1_db']:+.2f} dB。因而“同方向新的人体反射在全带增强”这一具体预测未通过。该方向的已有信号可能受到人体扰动，单凭此处无法判定传播路径。",
        "",
        "投影计算为 Σₖ aₖᴴRₖaₖ / (3Σₖ tr Rₖ)，在相同频段比较不同轮次。按各自最高峰归一化的 MUSIC 曲线只能比较谱形，两者尺度不同。旧 −21° 是不同初始化下的未校准谱参数，这里检验相同名义角度上的变化。",
        "",
        "## 各子阵角度与 0°参考",
        "",
        f"按左 TX=(0,0)、右 RX=(3.65,0)、A=(0.8,+0.6)，等高模型下整阵理论角 {truth['G fusion']:.2f}°，理论超额时延 {tof:.3f} ns。理论角按各子阵中心换算。未校准谱峰、背景候选先冻结，再与站位对照。表中二维主峰来自模型阶数 3，逐频点经验参考只由 R1 前 5 秒产生，后一列属于附加检查。",
        "",
        "| 子阵 | 理论角/° | 原始空间主峰 R1/R2/R3 | 原始二维 R2 主峰/° | 逐频参考后 R2 空间主峰/° |",
        "|---|---:|---|---:|---:|",*angle_rows,
        "",
        f"预设常数/频率线性响应模型用 R1 前半段拟合、后半段选型，R3 整轮留出。可靠频点占比 ≥80%、圆相位残差 P90≤15° 的两项标准合计 {int(held.pass_pair.sum())}/12 条链对通过。较灵活的逐频点经验模板有 {int(empirical.pass_pair.sum())}/12 条通过。两条参考链对均通过的子阵为 {', '.join(g3_pass[g3_pass==2].index)}。这只检查同一方向、同一场景的参考重复性，尚无独立非零角度标定源验证。",
        "",
        f"探索性 G3 经验参考结果为 0° → {g3.dominant_deg:+.0f}° → 0°；有人段低/高半频的最大峰为 {g3.low_band_dominant_deg:+.0f}°/{g3.high_band_dominant_deg:+.0f}°，四个时间块为 {g3.quarter_dominant_deg}°。与 G3 理论角的偏差为 {g3diff:.2f}°。R2 的 0° 谱值仅比其 −6° 峰低 0.61 dB，峰较宽。同组未经校准的 R2 最大峰在 −84°。逐频经验参考可能含环境多径，这个偏差作为一个同轮候选记录，不计入定位 MAE。四 G 的经验参考谱非相干融合在三轮均以 0° 为最大峰。",
        "",
        "![两条候选的方向复验](C1与G3_方向复验.png)",
        "",
        "## 背景抑制与二维谱",
        "",
        "另用已有同频协方差白化方法。在 R1 前 5 秒建背景，将数据协方差和导向矢量一起变换；用后 5 秒的四块确定变化门槛。分别向 R2、R3 盲搜。以 R3 作为参考的反向检验单独标注为敏感性分析。下表的稳定峰需通过变化门槛，低/高频均有邻近峰，并且至少三个时间块匹配（容差 9°）。±90°为边界。",
        "",
        "| 子阵 | R1 参考，看 R2 有人 | R1 参考，看 R3 无人 | R3 参考，看 R2 有人 |",
        "|---|---|---|---|",*white_rows,
        "",
        "以 R1 为参考，R2 的 C1 在 −12° 出现候选，白化后的功率比分数为 76.17；R3 空场也在 −15° 出现稳定候选，分数为 5.31。换成 R3 参考时，R2 的近零分支变为 0°。这一分数突出相对于背景的新增成分，与固定 −21° 的原始投影比例是不同的量。R2 有 8 个稳定峰（含边界），R3 空场有 9 个，其中各有 7 个非边界峰。这是单次空场的假候选计数。不同 G 尚未给出共同的近 A 方向。",
        "",
        f"C1 的原始二维最大峰从 R1 的 −69° 变为 R2 的 −6°，离场后回到 −66°。R2 的 −6° 与 C1 理论角相差 {abs(-6-truth['C1']):.2f}°。但固定在 −6° 看时，三轮都在同一个 204 ns 谱格取得该行最大值，相对当轮全谱最高值分别为 −1.18、0、−1.44 dB。这个原本存在的谱区在有人时升到第一位。针对这个已观察到的 C1 结果做附加检查，在整个 R2 上模型阶数 2、3、4 均选 −6°；按 2.5 秒切块，阶数 3 给出 {' / '.join(f'{x:+.0f}' for x in checked.peak_deg)}°。204 ns 是未校准的时延参数，尚未建立同记录直达径零点。这里的 −6° 和上节的 G3 −6° 来自不同子阵和算法。",
        "",
        "![C1 三轮二维谱](C1_三轮AoA_ToF.png)",
        "",
        f"C1/C2 的共享 6/8 相位对每轮有 {int(shared.matched_cycles.min())} 个完整配对周期，轮内中位残差 {shared.median_relative_phase_deg.min():.2f}–{shared.median_relative_phase_deg.max():.2f}°，P90 最大 {shared.p90_relative_phase_deg.max():.2f}°。这是局部跨状态一致性。全阵列的有效固定相位及未拟合方向上的联合阵列验证仍缺失。完整 12 阵元相干增益和物理位置误差保留为待验证。",
        "",
        "## 下一组采什么",
        "",
        "优先在一次 RX 初始化中采 空→A→空→A→空→A→空，每次重新站位，保存独立完整窗口。把 G3 的负向偏转和 C1 的二维候选先写入规则，在新记录上检验。然后加既定的 C=(0.8,−1.2) 做异侧对照，中心理论方向约 +22.83°。只用新轮次比较方向是否随站位改变、空场是否返回，并报告失败与输出率。要验证 12 根相干增益，仍需已知非零方向源、全组相位/时延对齐及与 G2 主基线的同窗比较。",
        "",
        "## 数据与复算",
        "",
        "在本目录依次运行 analyze.py、sensitivity.py、report.py。处理只读原始文件。背景与角度算法复用上一批的 PicoScenes 解码、阵元重排、事件归属、同频空间模型及二维 MUSIC 实现。",
        "",
        "[逐轮文件、接收机和包数](raw_runs.csv) · [状态质检](state_quality.csv) · [成对差异](paired_changes.csv) · [C1 固定方向三轮结果](C1_minus21_before_during_after.csv) · [G3 经验参考方向](exploratory_empirical_angular_peaks.csv) · [C1 时间/阶数敏感性](exploratory_C1_joint_time_order.csv) · [盲选结果哈希](frozen_outputs.json) · [各个局部峰事后几何对照](local_peaks_vs_geometry.csv) · [背景假候选和几何对照](background_peaks_vs_geometry.csv)。"
    ]
    out=ROOT/"New_A_空场有人空场_实测分析.md"
    out.write_text("\n".join(text)+"\n",encoding="utf8")
    links=re.findall(r'\]\(([^)]*)\)',out.read_text(encoding="utf8"))
    assert all((ROOT/v).is_file() for v in links)
    result=dict(manifest_entries_verified=len(checks),total_formal_target_frames=int(r.target_frames.sum()),
                total_guarded_packets=int(qc.guarded_packets.sum()),receiver_epochs=1,independent_occupied_runs=1,
                covariance_contrast_states=6,covariance_contrast_quarters=24,prior_branch_wideband_increase=False,
                empirical_G3_diagnostic_difference_deg=g3diff,physical_human_MAE=None,
                measured_excess_ToF=None,position_error=None,coherent_12_gain=None,frozen_files=len(frozen),checked_links=len(links))
    (ROOT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(result)


if __name__=="__main__":
    report()
