"""Score the frozen predictions, audit equal packet budgets and summarize array geometry."""
import json
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
import predict_subarrays as p
sys.path.insert(0,str(ROOT.parent/"paired_be_abcde_20261105"))
from aggregate import POINTS

LATEST=list("ABCDE")
CFGS=dict(p.SETS)
FOUR="+".join(p.GROUPS)
METHOD_LABEL={"positive_q1":"正变化一阶","background_null":"投影空场主方向","signed_rank":"正负变化自适应阶数"}


def angle(v,ndigits=0):
    if not np.isfinite(v):return "无输出"
    return "0°" if v==0 else f"{v:+.{ndigits}f}°".replace("-","−")


def save(name,d):
    d.to_csv(ROOT/name,index=False,encoding="utf-8-sig")


def md(fields):
    return "| "+" | ".join(map(str,fields))+" |"


def antennas(states):
    return sorted(set(a for s in states for a in p.base.MAP[s]))


def truth(site,states,centre=0):
    x,y=POINTS[site[:1]]
    yc=0. if centre else p.base.Y[np.array(antennas(states))-1].mean()
    return np.degrees(np.arctan2(-y-yc,3.65-x))


def measured_pairs(states):
    return {tuple(sorted((a,b))) for s in states for i,a in enumerate(p.base.MAP[s]) for b in p.base.MAP[s][i+1:]}


def geometry_audit():
    rows=[]
    for label,states in p.SETS:
        pairs=measured_pairs(states)
        moment=np.mean([np.var(np.asarray(p.base.MAP[s],float))*3 for s in states])
        lags=sorted({j-i for i,j in pairs})
        # Ideal single-source, same SNR and packets, with unknown complex gain in each state.
        # No global phase or other cross-state correlations enter either calculation.
        alternatives=[]
        for s in states:
            a,b=(p.previous.STEER[s][:,:,np.flatnonzero(p.ANG==v)[0]] for v in (48,-18))
            alternatives.append(abs(np.sum(a.conj()*b,axis=1)/3)**2)
        rows.append(dict(config=label,physical_antennas=len(antennas(states)),observable_unique_pairs=len(pairs),
                         lags_in_28mm="/".join(map(str,lags)),max_simultaneous_baseline_mm=max(lags)*28,
                         centred_second_moment_d2=moment,equal_packet_local_curvature_vs_G2=moment/2,
                         ideal_alternative_fraction_median=np.median(np.mean(alternatives,axis=0))))
    g=pd.DataFrame(rows)
    save("ideal_geometry.csv",g)
    r=g.set_index("config").loc["six_states"]
    assert r.observable_unique_pairs==17 and r.max_simultaneous_baseline_mm==196
    assert np.isclose(r.equal_packet_local_curvature_vs_G2,3.5)
    return g


def proposed_switch_geometry():
    # Design-only calculation; these two switches did not occur in the raw CSI.
    trial={"X1":[1,2,12],"X2":[1,5,12]}
    eligible_by_chain=[{mapping[i] for mapping in p.base.MAP.values()} for i in range(3)]
    rows=[]
    alternatives=[]
    for label,mapping in {**{s:p.base.MAP[s] for s in p.GROUPS},**trial}.items():
        assert all(m in eligible_by_chain[i] for i,m in enumerate(mapping))
        y=p.base.Y[np.array(mapping)-1]
        phase=2j*np.pi/p.base.C*p.base.F[:,None]*y[None,:]*(np.sin(np.radians(48))-np.sin(np.radians(-18)))
        alternatives.append(abs(np.exp(phase).mean(axis=1))**2)
        rows.append(dict(state=label,mapping="/".join(map(str,mapping)),measured_state=label in p.GROUPS,
                         max_pair_mm=round(1000*np.ptp(y)),ideal_moment_d2=np.var(mapping)*3,
                         ideal_alternative_fraction=float(np.median(alternatives[-1]))))
    df=pd.DataFrame(rows);save("proposed_switch_geometry.csv",df)
    return df,float(np.median(np.mean(alternatives,axis=0)))


def assess(g,powers):
    v=g.set_index(["ref","test","block","band"]).angle
    def calc(r):
        pk=v.loc[1,r,"full","full"]
        bands=np.array([v.loc[1,r,"full",b] for b in ("low","high")])
        q=np.array([v.loc[1,r,f"Q{i}","full"] for i in range(1,5)])
        return pk,bands,q,bool(np.all(np.abs(bands-pk)<=9)),int(np.sum(np.abs(q-pk)<=9))
    fg,band,q,band_ok,cnt=calc(2)
    bg,empty_b,empty_q,bg_b_ok,bg_cnt=calc(3)
    reverse=v.loc[3,2,"full","full"]
    denominator=powers.loc[1,3]
    ratio=powers.loc[1,2]/denominator if denominator>0 else np.nan
    failures=[]
    if not np.isfinite(fg):failures.append("R2不可用")
    if abs(fg)>=87:failures.append("边界")
    if not band_ok:failures.append("频段不一致")
    if cnt<3:failures.append("时间块不足")
    if not abs(fg-reverse)<=9:failures.append("换参考不一致")
    if not ratio>1:failures.append("强度未过空场")
    if abs(fg-bg)<=9 and bg_b_ok and bg_cnt>=3:failures.append("稳定空场近邻")
    assessed=np.isfinite(bg) and np.isfinite(empty_b).all() and np.isfinite(empty_q).all()
    return dict(peak=fg,r3=bg,reverse=reverse,r2_low=band[0],r2_high=band[1],quarter_support=cnt,
                ratio=ratio,gate=not failures,negative_complete=assessed,strict_gate=not failures and assessed,
                failure="、".join(failures+([] if assessed else ["空场角度未能完整检查"])))


def performance():
    b=pd.read_csv(ROOT/"blind_angles.csv")
    w=pd.read_csv(ROOT/"contrast_power.csv").set_index(["point","budget","config","ref","test"]).sort_index().intensity
    baseline=pd.read_csv(ROOT/"packet_budgets.csv")
    matched=baseline[baseline.budget=="equal"]
    assert matched.groupby(["point","run"]).total_packets.nunique().eq(1).all()
    assert matched.groupby("point").total_packets.nunique().eq(1).all()
    records=[]
    for (site,budget,config,method),g in b[b.method!="profile_radius"].groupby(["point","budget","config","method"],sort=False):
        result=assess(g,w.loc[site,budget,config])
        states=CFGS[config]
        local=truth(site,states);common=truth(site,states,1)
        records.append(dict(point=site,budget=budget,config=config,antennas=len(antennas(states)),method=method,theory_deg=local,
                            err=abs(result["peak"]-local),common_centre_err=abs(result["peak"]-common),**result))
    r=pd.DataFrame(records);save("paired_scores.csv",r)
    latest=r[r.point.isin(LATEST)]
    groups=[]
    for (budget,config,method),g in latest.groupby(["budget","config","method"],sort=False):
        valid=g.err.notna()
        groups.append(dict(budget=budget,config=config,method=method,n=5,usable=int(valid.sum()),mean_abs=g.err.mean(),
                           median_abs=g.err.median(),worst=g.err.max(),within8=int(g.err.le(8).sum()),gate=int(g.gate.sum()),
                           fully_checked=int(g.strict_gate.sum()),accepted_within8=int((g.strict_gate&g.err.le(8)).sum()),
                           accepted_mean=g.loc[g.strict_gate,"err"].mean(),common_centre_mean=g.common_centre_err.mean()))
    metrics=pd.DataFrame(groups);save("architecture_summary.csv",metrics)
    matched=latest[(latest.budget=="equal")]
    comparisons=[]
    for method in p.METHODS:
        m=matched[matched.method==method].set_index(["config","point"])
        for label in [FOUR,"six_states"]:
            for site in LATEST:
                a,h=m.loc["G2",site],m.loc[label,site]
                comparisons.append(dict(point=site,method=method,from_config="G2",to_config=label,G2_error=a.err,
                                        twelve_error=h.err,reduction_deg=a.err-h.err,common_centre_reduction_deg=a.common_centre_err-h.common_centre_err,
                                        G2_strict_gate=a.strict_gate,twelve_strict_gate=h.strict_gate))
    comp=pd.DataFrame(comparisons);save("equal_packet_gains.csv",comp)
    near=b[(b.method=="profile_radius")&(b.ref==1)&(b.test==2)].copy()
    near["truth"]=[truth(site,CFGS[c]) for site,c in zip(near.point,near.config)]
    near["err"]=(near.angle-near.truth).abs()
    save("radius_sensitivity.csv",near)
    # Four disjoint G groups. The omitted-group bearing is obtained from disjoint antennas;
    # use all packets so removing a group does not change anyone else's sampling.
    loo=[]
    for row in latest[(latest.budget=="all")&(latest.method=="background_null")].itertuples():
        states=CFGS[row.config]
        if len(states)!=3:continue
        held=(set(p.GROUPS)-set(states)).pop()
        hp=latest.query('point==@row.point and budget=="all" and method=="background_null" and config==@held').peak.item()
        loo.append(dict(point=row.point,fit=row.config,omitted=held,fit_peak=row.peak,omitted_peak=hp,
                        common_angle_difference=abs(row.peak-hp),within9=abs(row.peak-hp)<=9))
    save("disjoint_group_check.csv",pd.DataFrame(loo))
    return r,metrics,comp,near


def draw(metrics):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(11,5.0),layout="constrained",gridspec_kw={"width_ratios":[1,1.25]})
    g=measured_pairs(p.GROUPS);all_pairs=measured_pairs(p.STATES)
    grid=np.full((12,12),np.nan)
    for i in range(1,13):
        for j in range(i+1,13):
            pair=(i,j)
            grid[i-1,j-1]=1 if pair in g else 2 if pair in all_pairs else 0
    m=ListedColormap(["#e7eaef","#467c9f","#d0793e"]);m.set_bad("white")
    axes[0].imshow(grid,cmap=m,vmin=0,vmax=2,interpolation="none")
    axes[0].set_xticks(range(12),range(1,13),fontsize=8);axes[0].set_yticks(range(12),range(1,13),fontsize=8)
    axes[0].set(xlabel="天线编号",ylabel="天线编号",title="各切换状态中可以同包观测的阵元对")
    axes[0].legend(handles=[Patch(color=c,label=l) for c,l in zip(m.colors,["未观测","G1–G4","C1/C2 补充"])],
                   loc="upper left",bbox_to_anchor=(0,-.13),ncol=3,frameon=False,fontsize=8)
    eq=metrics[metrics.budget=="equal"].set_index(["method","config"])
    x=np.arange(3)
    for method,color,mark in [("positive_q1","#36749d","o"),("background_null","#bc562e","s"),("signed_rank","#277767","D")]:
        y=[eq.loc[(method,s),"mean_abs"] for s in ("G2",FOUR,"six_states")]
        axes[1].plot(x,y,marker=mark,c=color,lw=1.8,label=METHOD_LABEL[method])
        for i,val in enumerate(y):
            offset=(4,6 if method!="signed_rank" else -14)
            if i==1:offset={"positive_q1":(-25,6),"background_null":(7,12),"signed_rank":(6,-16)}[method]
            axes[1].annotate(f"{val:.1f}",(i,val),xytext=offset,
                             textcoords="offset points",fontsize=8,color=color)
    axes[1].set_xticks(x,["G2\n3 根","四个 G\n12 根","四个 G + C1/C2\n仍为 12 根"])
    axes[1].set(xlim=(-.17,2.19),ylim=(0,48),ylabel="五点平均绝对谱峰偏差 / °",title="相同的时间和包数，固定算法再比天线")
    axes[1].legend(loc="upper center",ncol=3,frameon=False,fontsize=8)
    axes[1].grid(axis="y",alpha=.2)
    fig.savefig(ROOT/"阵元对与等包数对比.png",dpi=175)
    plt.close(fig)


def report(results,metrics,comp,near,geo):
    m=metrics[metrics.budget=="equal"].set_index(["method","config"])
    main=results[(results.budget=="equal")&(results.method=="background_null")].set_index(["point","config"])
    rows=[]
    for site in LATEST:
        g,f,s=[main.loc[site,c] for c in ("G2",FOUR,"six_states")]
        rows.append(md([site,f"{g.theory_deg:+.2f}°",angle(g.peak),f"{g.err:.2f}°",angle(f.peak),
                        f"{s.theory_deg:+.2f}°",angle(s.peak),f"{s.err:.2f}°",f"{s.r2_low:g}° / {s.r2_high:g}°"]))
    summary=[]
    for method in p.METHODS:
        for c,name in [("G2","G2（3 根）"),("G3","G3（3 根）"),("C1","C1（稀疏 3 根）"),(FOUR,"四 G（12 根）"),("six_states","四 G＋两 C（12 根）")]:
            v=m.loc[method,c]
            summary.append(md([METHOD_LABEL[method],name,f"{v.mean_abs:.2f}°",f"{v.median_abs:.2f}°",f"{int(v.within8)}/5",
                               f"{int(v.gate)}/5",f"{int(v.fully_checked)}/5",f"{int(v.accepted_within8)}/5"]))
    repeats=[]
    for point in ("B_old","C_old"):
        for config in ("G2","six_states"):
            a=main.loc[point,config];n=main.loc[point[:1],config]
            repeats.append(md([point[:1],config,angle(a.peak),angle(n.peak)]))
    budgets=pd.read_csv(ROOT/"packet_budgets.csv")
    pkts=budgets.query('budget=="equal" and config=="G2" and run==2').set_index("point")
    loo=pd.read_csv(ROOT/"disjoint_group_check.csv").groupby("point").within9.sum()
    nv=near[(near.point.isin(LATEST))&(near.config=="six_states")]
    all_plane=metrics.query('method=="background_null" and config=="six_states" and budget=="all"').mean_abs.item()
    geom=geo.set_index("config")
    proposal,alias_fraction=proposed_switch_geometry()
    control=comp.query('method=="positive_q1" and to_config=="six_states"').set_index("point").loc[LATEST]
    pairedrows=[md([s,f"{v.G2_error:.2f}°",f"{v.twelve_error:.2f}°",f"{v.reduction_deg:.2f}°"])
                for s,v in control.iterrows()]
    out=[
        "# 使用十二根切换天线的 MUSIC 对照",
        "",
        "相同的总包数下，投影空场主方向的 MUSIC 在固定 G2 上得到 7.01° 的五点平均谱峰偏差，换成十二根六状态为 23.73°。正变化一阶对照则从 40.14° 降至 19.55°，五点分别都有改善。天线收益取决于子阵模型和传播条件。G2 的 7.01° 属于未筛选的已知站位谱峰，五个 G2 结果均未通过完整诊断。",
        "",
        "## 两种新处理",
        "",
        "1. **投影空场主方向，再做联合 MUSIC。** 每个状态、频点从 R1 学到主特征向量 b，用 B=I−bbᴴ 同时投影测试协方差和真实几何导向矢量 a。取 BR₂B 的主向量 u，在全角域最小化 Σ aᴴ(B−uuᴴ)a / Σ aᴴBa。各状态仅贡献自己实际测得的局部子空间，不需要同步拼接。R1 主分量需要解释至少 70% 的该频组能量。R2 投影后的主能量必须超过 R1 四个时间块各自投影能量的最大值。同一状态至少半数频组可用，且合计投影导向能量达到原值的 10%。分母过小时拒绝该角度，防止把被投影掉的方向误认为理想匹配。",
        "2. **正负变化子空间 MUSIC。** 对 ΔR 的特征值按绝对值判断变化。阈值仅由 R1 四个时间块相对 R1 全轮的协方差算子范数确定。保留一或二维显著变化子空间。变化维数为 0 或 3 的频点没有可用的信号/噪声分离，故弃用。对不同秩按 3/(3−q) 归一化投影残差。负特征方向可以反映原有路径被削弱或静态场的交叉项，不据此增加人数。对照方法用单个最大正特征方向，三种方法都先联合各状态的投影残差再选峰，没有对伪谱取平均或乘积。",
        "",
        "投影方法的动机是若占用时 H₂=H₀+αa，空场由 H₀ 主导，则 BH₂=αBa。真实室内通常有多条路径，投影也可能同时削弱人体分量。正负两方向方案保留的子空间越大，单个三阵元可用于区分角度的噪声维数越少。在单源合成例中，两种算法均在植入角 −28° 产生投影极小值，对无变化输入拒绝输出。此为实现自检。利用子阵局部协方差处理非相干数据的理论可参见 [Suleiman 等](https://arxiv.org/abs/1704.06000)。本文实现的投影准则不同于该文的极大似然法。",
        "",
        "## 怎样公平比较三根和十二根",
        "",
        "同一批 R1 空、R2 人、R3 空，均沿用已经核验的天线映射、Intel 5300 解码、READY 后 3–21 ms 以及相同频率和角度网格。主要基线为事先指定的 G2，另列 G3 和 C1 以避免挑最有利的三根。每批每轮各方案总包数相同，按四个 2.5 s 时间块均匀抽取，再均分给参与状态，不按 CSI 强弱抽包。各站位每方案每轮包数为 "+ "、".join(f"{s} {pkts.loc[s,'total_packets']}" for s in LATEST)+"。各方案总时长相同，切换导致实际包时刻不同。另行提供全部原包、不限制数量的结果。",
        "",
        "| A–E 最新批次 | G2 理论角 | G2 主峰 | G2 偏差 | 四 G 主峰 | 阵列中心理论角 | 六态主峰 | 六态偏差 | 六态低/高半频主峰 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",*rows,
        "",
        "| 等包数方法 | 实际配置 | 五点平均偏差 | 中位偏差 | 在 ±8° 内 | 通过基本检查 | 空场也可完整核对且通过 | 通过完整检查且在 ±8° 内 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",*summary,
        "",
        "基本检查继承先前的双半频、时间块、反向参考、幅度和空场稳定峰否决。若 R3 由于投影不充分而不产生角度，单独列‘基本检查’和‘空场也可完整核对’两种计数。即使波动小也不能确认人体路径。B 在六态背景投影的主峰为 0°，即使基本检查通过也与人体方向相差 54.69°。G2 的均值接近目标，五个结果却全部存在半频段分歧，单个平均数不能代表已验证的误差。所有点保留，失败按完整轮次计数。",
        "",
        "在正变化一阶对照中，五点六态主峰都比预设 G2 的主峰更接近理论方向。改善主要来自 B 和 E。五点中落在 ±8° 内仍均为 2/5，完整诊断通过数从 G2 的 2/5 变为六态的 1/5。这说明谱峰偏差有条件地减小，尚不能证明可输出的人体方向更多。在同样使用全部包时，本轮正变化一阶六态的联合投影残差为 19.75°，上轮六态谱乘积为 17.87°。相对三根天线的收益与更换融合方式的效果需要分别报告。",
        "",
        "| 点 | G2 偏差 | 六态偏差 | 偏差减少量 |","|---|---:|---:|---:|",*pairedrows,
        "",
        f"上述主表照各配置的几何中心评价。再统一按 RX 中心计分，正变化一阶的 G2 / 六态为 {m.loc[('positive_q1','G2'),'common_centre_mean']:.2f}° / {m.loc[('positive_q1','six_states'),'common_centre_mean']:.2f}°，背景投影的 G2 / 六态为 {m.loc[('background_null','G2'),'common_centre_mean']:.2f}° / {m.loc[('background_null','six_states'),'common_centre_mean']:.2f}°。两种比较的方向一致。",
        "",
        "![可测阵元对与同包数偏差](阵元对与等包数对比.png)",
        "",
        "## 天线数量、长基线和可辨识性",
        "",
        "12 根共有 66 个不同的无序阵元对。四个 G 各自同包测到 3 对，总计 12 对。C1/C2 只新增 5 个不同阵元对，六态合计 17 对。实测非零间距只有 1、2、4、5、7 个 28 mm。最长实测间距 196 mm，完整阵列几何跨度 308 mm 的端到端相位差没有被同一包观测。直接补零或推断所有未测阵元对会引入额外假设。",
        "",
        "理想校准、单一远场源、相同信噪比、各状态未知公共相位、总包数相同且在状态间均分时，局部角度相位敏感度正比于各状态的 Σ(yₘ−ȳₛ)² 的平均。G2 和四 G 的该值均为 2d²，C1 为 26d²、C2 为 8d²，六态平均为 7d²。相对于 G2 的理想局部信息比为 3.5，来自接入长基线。单用 C1 的局部信息比更高，但稀疏阵列会产生多个角度分支。此数是几何推算，与实测误差下降倍数不同。",
        "",
        f"C2 理想平面波在 +48° 与 −18° 的归一化导向向量内积模平方中位数为 {geom.loc['C2','ideal_alternative_fraction_median']:.3f}（越接近 1 越难区分）。等权使用六种局部几何时，该替代方向的平均响应分数降至 {geom.loc['six_states','ideal_alternative_fraction_median']:.3f}。这说明不同基线有机会消除歧义，条件是组内相位可信。更多阵元数量本身无法保证该条件。[稀疏子阵及共阵 MUSIC](https://arxiv.org/abs/2409.00033)",
        "",
        "## 有限距离和独立性检查",
        "",
        f"在只用接收阵元几何的球面波敏感性中，同时在 0.75、1.5、3、6 m 和远场上扫描，再保留每个角度最小残差。全包、六态主方案的平均谱峰偏差由 {all_plane:.2f}° 变为 {nv.err.mean():.2f}°，各点的峰从平面波 A–E 的 −30°、0°、−22°、+50°、+45° 变为 "+ "、".join(angle(nv.set_index('point').loc[s,'angle']) for s in LATEST)+"。半径是敏感性参数。该搜索没有测得直达径或物理距离。",
        "",
        "每次留出一个 G，比较另外三个 G 的估计与未参与计算的 G 自己的峰。全包、背景投影分支中，四次留出里符合 ±9° 的次数依次为 "+ "、".join(f"{s} {int(loo.loc[s])}/4" for s in LATEST)+"。各组中心略有不同，指标衡量方向共识，不能代替定位误差。",
        "",
        "| 同点两次采集 | 配置 | 较早批次主峰 | 新批次主峰 |","|---|---|---:|---:|",*repeats,
        "",
        "这些记录之前已被分析，本次先冻结规则和盲选结果，再加入站位。每点最新批次只有一次进场，较早的 B/C 来自另一初始化，未与新 CSI 合并。",
        "",
        "## 建议的下一次独立检验",
        "",
        "保留预设 G2 三阵元基线。让短基线 G 子阵生成多个局部方向候选，再用未参加候选生成的长基线 C1/C2 检查这些方向在各自噪声子空间中的残差。只有长基线能稳定区分两支、且留出一个物理上不重叠的 G 组也支持时才输出单一角度，否则保留双解或拒判。这是待测方案。分离规则和打分差阈值须在标定记录上冻结，在新的独立空→人→空进场上同时报告盲选偏差、误选分支率和单解覆盖率。要使用完整 0.308 m 相干孔径，还需验证状态间相位与频率斜率、非同时切换稳定性及未拟合的已知源方向。[WiCAL 多阶段相位对齐](https://arxiv.org/abs/2505.21408)",
        "",
        f"如果开关板能独立选择三条 RF 链的端口，可在原有六态之外增加两个测试状态 X1=[1,2,12]、X2=[1,5,12]，保留现有 C1/C2 校准连接。编号与各 RF 链现有分工相容，具体开关命令和电路可达性尚未验证。两状态都能在同一包里测到 1–12 的 308 mm 基线，X1 同时有 28 mm 短基线，X2 有不同的中间间距。仅在几何模型中，若等权对比‘四 G＋X1/X2’与‘四 G＋C1/C2’，+48° 与 −18° 的替代响应从 {geom.loc['six_states','ideal_alternative_fraction_median']:.3f} 变为 {alias_fraction:.3f}。X1 或 X2 单独都仍是三阵元，需要把它们单独作为对照，再看使用全部十二根是否有进一步收益。新增状态延长扫描周期，正式比较须维持同等采集时间和总包数。[仅几何的开关设计计算](proposed_switch_geometry.csv)",
        "",
        "[逐点及空场检查](paired_scores.csv) · [固定配置汇总](architecture_summary.csv) · [三根与十二根逐点配对](equal_packet_gains.csv) · [可观测几何](ideal_geometry.csv) · [实际包数](packet_budgets.csv) · [冻结规则](protocol.json) · [盲选哈希](blind_registry.json)。复算顺序为 predict_subarrays.py、score_subarrays.py。",
    ]
    path=ROOT/"十二天线与MUSIC的新方法验证.md"
    path.write_text("\n".join(out)+"\n",encoding="utf8")
    for href in re.findall(r'\]\(([^)]*)\)',path.read_text(encoding="utf8")):
        assert href.startswith("https://") or (ROOT/href).is_file()
    print(m.xs("background_null")[["mean_abs","within8","gate","fully_checked","accepted_within8"]].loc[["G2","G3",FOUR,"six_states"]])
    print(comp.groupby(["method","to_config"]).reduction_deg.agg(["min","mean","max"]))
    print("Leave-out G within 9 degrees:",loo.to_dict())


if __name__=="__main__":
    reg=json.loads((ROOT/"blind_registry.json").read_text(encoding="utf8"))
    assert all(p.base.sha(ROOT/name)==sha for name,sha in reg.items())
    geometry=geometry_audit()
    r,m,c,near=performance()
    draw(m)
    report(r,m,c,near,geometry)

