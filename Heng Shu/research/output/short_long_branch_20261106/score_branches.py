"""Evaluate the two frozen experiments; truth never chooses a candidate or threshold."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import predict_branches as experiment
from predict_subarrays import base
sys.path.insert(0,str(ROOT.parent/"paired_be_abcde_20261105"))
from aggregate import POINTS

CORE="background_null"
LONG="short_long_12"
BASE="G2_splits"
SHORTHW="G_only_12"


def truth(point,centre=0.):
    x,y=POINTS[point]
    return np.degrees(np.arctan2(-y-centre,3.65-x))


def angle(v,precision=0):
    if not np.isfinite(v):return "—"
    if abs(v)<1e-12:return "0°"
    return f"{v:+.{precision}f}°".replace("-","−")


def md(fields):
    return "| "+" | ".join(str(v) for v in fields)+" |"


def save(name,data):
    data.to_csv(ROOT/name,index=False,encoding="utf-8-sig")


def verify_registry(directory):
    j=json.loads((directory/"blind_registry.json").read_text(encoding="utf8"))
    assert all(base.sha(directory/name)==v for name,v in j.items())


def score(directory):
    verify_registry(directory)
    b=pd.read_csv(directory/"blind_decisions.csv")
    b["truth_rx_deg"]=b.point.map(truth)
    local_y=base.Y[np.array(base.MAP["G2"])-1].mean()
    b["truth_G2_deg"]=b.point.map(lambda point:truth(point,local_y))
    for field in ["seed1","seed2","provisional","selected"]:
        b[field+"_abs_error"]=(b[field]-b.truth_rx_deg).abs()
    b["closest_error_diagnostic_only"]=b[["seed1_abs_error","seed2_abs_error"]].min(axis=1)
    assert b.loc[b.decision!="单解","selected"].isna().all()
    return b


def plot(summary,table):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":9.5,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained",gridspec_kw={"width_ratios":[1.18,1]})
    ax=axes[0]
    for i,r in enumerate(summary.itertuples()):
        ax.plot([r.seed1,r.seed2],[i,i],color="#cad1d8",lw=1.4)
        ax.scatter([r.seed1,r.seed2],[i,i],s=54,facecolor="white",edgecolor="#437498",lw=1.7,zorder=3)
        ax.scatter(r.truth_rx_deg,i-.14,marker="D",s=47,color="#2d3542",zorder=5)
        ax.scatter(r.long_preference,i+.13,marker="v",s=53,color="#bb5325",zorder=5)
    ax.set_yticks(range(len(summary)),summary.point)
    ax.set(ylim=(4.5,-.5),xlim=(-98,98),xticks=[-90,-60,-30,0,30,60,90],xlabel="RX 方向角 / °",
           title="共用的 G2 候选与十二根的全频倾向")
    handles=[Line2D([],[],marker=m,ls="none",markerfacecolor=c,markeredgecolor=e,markersize=7,label=l)
             for m,c,e,l in [("D","#2d3542","#2d3542","阵列中心真值"),("o","white","#437498","短基线候选"),
                             ("v","#bb5325","#bb5325","长基线暂定排序")]]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.15),ncol=3,frameon=False,fontsize=8)
    ax.grid(axis="x",alpha=.17)
    ax=axes[1]
    x=np.arange(5)
    for delta,col,key,label in [(-.15,"#437498","L1_margin_to_first_db","C1"),(.15,"#bd6127","L2_margin_to_first_db","C2")]:
        ax.bar(x+delta,table[key],width=.27,color=col,label=label)
    ax.axhline(0,color="#373d45",lw=1);ax.axhline(1,color="#929ba5",ls=":",lw=1)
    ax.axhline(-1,color="#929ba5",ls=":",lw=1)
    ax.set_xticks(x,table.index)
    ax.set(title="两条长基线各偏向哪一支",ylabel="第一候选相对第二候选的优势 / dB",xlabel="站位")
    ax.grid(axis="y",alpha=.15);ax.legend(frameon=False,ncol=2,loc="upper right")
    fig.savefig(ROOT/"五点候选与长基线投票.png",dpi=180);plt.close(fig)


def report():
    primary=score(ROOT)
    secondary=score(ROOT/"two_short_proposal")
    save("all_scored_decisions.csv",primary);save("two_short_scored_decisions.csv",secondary)
    a=pd.read_csv(ROOT/"packet_allocations.csv")
    budgets=a.groupby(["point","run","arm"]).packets.sum()
    assert budgets.groupby(["point","run"]).nunique().eq(1).all()
    assert budgets.groupby("point").nunique().eq(1).all()
    proposal=a[a.role=="P"]
    assert proposal.groupby(["point","run"]).selected_ordinals_sha256.nunique().eq(1).all()
    prior=pd.read_csv(ROOT.parent/"music_subarray_gain_20261105"/"paired_scores.csv")
    baseline=pd.read_csv(ROOT/"G2_all_baseline.csv").query('ref==1 and test==2').set_index(["point","model"])
    checks=pd.read_csv(ROOT/"branch_checks.csv")
    sel=primary[(primary.threshold_db==1)&(primary.model==CORE)]
    twelve=sel[sel.arm==LONG].set_index("point")
    three=sel[sel.arm==BASE].set_index("point")
    short_control=sel[sel.arm==SHORTHW].set_index("point")
    rows=[];extra=[];mainrows=[];margins=[];sensitivityrows=[];other=[]
    sp=primary.query('threshold_db==1 and model=="positive_q1"').set_index(["point","arm"])
    alt=secondary.query('threshold_db==1').set_index(["point","arm"])
    d=pd.read_csv(ROOT/"role_spectra_checks.csv")
    for site in experiment.POINTS:
        v=twelve.loc[site];g=three.loc[site];sc=short_control.loc[site]
        assert np.allclose([v.seed1,v.seed2],[g.seed1,g.seed2],equal_nan=True)
        assert np.allclose([v.seed1,v.seed2],[sc.seed1,sc.seed2],equal_nan=True)
        old=prior.query('point==@site and method==@CORE and budget=="equal" and config=="G2"').iloc[0]
        peak=baseline.loc[site,CORE].global_peak
        assert old.peak==peak
        rec=dict(point=site,packets_per_run=int(budgets.loc[site,1,BASE]),truth_rx_deg=v.truth_rx_deg,truth_G2_deg=v.truth_G2_deg,
                 G2_full_peak=peak,G2_full_rx_error=abs(peak-v.truth_rx_deg),G2_full_local_error=abs(peak-v.truth_G2_deg),
                 G2_full_prior_gate=old.strict_gate,seed1=v.seed1,seed1_error=v.seed1_abs_error,seed2=v.seed2,seed2_error=v.seed2_abs_error,
                 G2_split_preference=g.provisional,G2_split_error=g.provisional_abs_error,G2_split_selected=g.selected,
                 G_only_12_preference=sc.provisional,G_only_12_selected=sc.selected,long_preference=v.provisional,long_preference_error=v.provisional_abs_error,
                 C1_margin_db=v.L1_margin_to_first_db,C2_margin_db=v.L2_margin_to_first_db,
                 long_selected=v.selected,long_selected_error=v.selected_abs_error,long_decision=v.decision,
                 provisional_reduction_vs_equal_G2=g.provisional_abs_error-v.provisional_abs_error,
                 strength_ratio=v.strength_ratio)
        rows.append(rec)
        mainrows.append(md([site,rec["packets_per_run"],angle(v.truth_rx_deg,2),f"{angle(peak)} / {rec['G2_full_rx_error']:.2f}°",
                            f"{angle(v.seed1)} / {v.seed1_abs_error:.2f}°",f"{angle(v.seed2)} / {v.seed2_abs_error:.2f}°",
                            f"{angle(v.provisional)} / {v.provisional_abs_error:.2f}°",v.decision]))
        rv=d.query('point==@site and model==@CORE and arm==@LONG and role in ["L1","L2"] and ref==3 and test==2 and block=="full" and band=="full"').set_index("role")
        margins.append(md([site,f"{v.L1_margin_to_first_db:+.2f}",f"{v.L2_margin_to_first_db:+.2f}",
                           f"{rv.loc['L1','margin_to_first_db']:+.2f}",f"{rv.loc['L2','margin_to_first_db']:+.2f}",
                           f"{v.strength_ratio:.2f}×"]))
        for model in ("C1","C2","G2"):
            y=base.Y[np.array(base.MAP[model])-1]
            diff=np.sin(np.radians(v.seed1))-np.sin(np.radians(v.seed2))
            power=abs(np.exp(2j*np.pi/base.C*base.F[:,None]*y[None,:]*diff).mean(axis=1))**2
            extra.append(dict(point=site,state=model,seed1=v.seed1,seed2=v.seed2,ideal_alias_power_median=np.median(power)))
        tt=alt.loc[site,LONG]
        pair=f"{angle(tt.seed1)} / {angle(tt.seed2)}" if np.isfinite(tt.seed2) else angle(tt.seed1)
        sensitivityrows.append(md([site,pair,f"{angle(tt.provisional)} / {tt.provisional_abs_error:.2f}°",
                                   angle(alt.loc[site,SHORTHW].provisional),tt.decision]))
        ss=sp.loc[site,LONG];sg=sp.loc[site,BASE]
        other.append(md([site,f"{angle(ss.seed1)}、{angle(ss.seed2)}",f"{angle(sg.provisional)} / {sg.provisional_abs_error:.2f}°",
                         angle(sg.selected),f"{angle(ss.provisional)} / {ss.provisional_abs_error:.2f}°",ss.decision]))
    summary=pd.DataFrame(rows);save("five_point_comparison.csv",summary)
    aliases=pd.DataFrame(extra);save("ideal_alias_at_seeds.csv",aliases)
    plot(summary,twelve)
    assert summary.packets_per_run.to_list()==[1056,1008,1056,1056,960]
    assert primary.query('model==@CORE and arm==@LONG').selected.isna().all()
    assert secondary.query('arm==@LONG').selected.isna().all()
    assert checks.query('model==@CORE and arm==@LONG and threshold_db==1').empty_checkable.all()
    assert (summary.provisional_reduction_vs_equal_G2<-1e-8).sum()==2
    metrics=[]
    for name,b in [("shared_G2",primary),("pooled_G2_G3",secondary)]:
        for (model,arm,tol),g in b.groupby(["model","arm","threshold_db"]):
            single=g.selected.notna()
            metrics.append(dict(proposal=name,model=model,arm=arm,threshold_db=tol,total=len(g),singles=int(single.sum()),
                                unresolved_two=(g.decision=="双候选未决").sum(),unconfirmed_one=(g.decision=="单候选未确认").sum(),
                                selected_within8=int((single&g.selected_abs_error.le(8)).sum()),selected_mean_abs=g.selected_abs_error.mean(),
                                first_seed_within8=int(g.seed1_abs_error.le(8).sum()),shortlist_contains_within8=int(g.closest_error_diagnostic_only.le(8).sum()),
                                forced_preference_mean_abs=g.provisional_abs_error.mean()))
    save("coverage_and_sensitivity.csv",pd.DataFrame(metrics))
    prose=[
        "# A–E 短基线提候选、长基线消歧实测",
        "",
        "在现有五次空→人→空记录里，十二根方案没有一个站位把两个短基线候选可靠地缩成单一方向。相同短名单上，A、B、D 的全频暂定方向沿用三根结果。C 和 E 的长基线排序都转到偏差更大的一支。原始候选、暂定排序、规则最终放行的角度分别记录。",
        "",
        "## 五点主结果",
        "",
        "使用最新的 New A、Real New B、Real New C、Real New D 和 New E，各一组 R1/R2/R3。全部峰和判别均用主模型‘投影掉空场主方向后做 MUSIC’。两种硬件共同使用 G2 的同一组包生成候选。为公平比较，表中两臂偏差都相对 RX 阵列中心。G2 自己的子阵理论角和相应误差另列在 CSV 中。候选固定在 1° 网格，长状态只能排序，不可另选新角度。",
        "",
        "| 点 | 每方案每轮包数 | 理论角 | G2 全包主峰 / 偏差 | 共同候选一 / 偏差 | 共同候选二 / 偏差 | 十二根全频暂倾向 / 偏差 | 十二根最终 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",*mainrows,
        "",
        f"G2 全包主峰的平均绝对偏差为 {summary.G2_full_rx_error.mean():.2f}°，按 G2 自己的中心则为 {summary.G2_full_local_error.mean():.2f}°。同一套六角色规则中，三根的全频暂定排序均取第一候选，平均绝对偏差 {summary.G2_split_error.mean():.2f}°。长基线全频暂定排序的对应数为 {summary.long_preference_error.mean():.2f}°。这两列只是‘如果强行给一个答案会怎样’的对照。两臂严格放行均为 0/5，放行后的单解平均误差未定义。共同短名单只有 A、C、E 含一个距真值 ≤8° 的候选，最近候选不当作算法输出。",
        "",
        "![五点候选与长基线分支比较](五点候选与长基线投票.png)",
        "",
        "## 哪一步帮助了，哪一步失败",
        "",
        "| 点 | C1 对候选一的优势 / dB | C2 对候选一的优势 / dB | 换 R3 背景后 C1 / dB | 换背景后 C2 / dB | 有人/后空变化强度 |",
        "|---|---:|---:|---:|---:|---:|",*margins,
        "",
        "正值表示该状态更偏向候选一，负值偏向候选二。这是两个方向邻域内的 MUSIC 投影残差比值，评分较低的残差代表较好的拟合。强度指标按同一尺度计算，用来识别变化，不用于选择靠近真值的角度。",
        "",
        "- **A。** 两条长基线都偏向 −11°，这本来就是短基线的首选。C2 的差只有 0.38 dB，G2 候选没有通过双半频段/分块联合检查。−90° 位于搜索边界，只记录为未排除的端点峰。",
        "- **B。** C1、C2 全频分别以 2.36、1.10 dB 支持 −71°，两状态的频段和分块相对排序均同向。换成后空场作背景，C1 转向另一支，独立留出的 G 组也未支持原分支。与 G2 全包 −72° 相比的 1° 差异来自提议包子集，同一短名单的三根对照也已经选 −71°。+90° 同样是端点峰。",
        "- **C。** 短基线提出了 +20°，距理论 +22.83° 只有 2.83°。C1、C2 全频却偏向 −21°（C1 差值未达 1 dB）。强行服从长基线会把偏差增至 43.83°。两个候选的 C1、C2 理想归一化导向功率相似度都低于 0.02，因此这次相反排序不能仅归因于这两个角的理想几何混叠。",
        "- **D。** +40° 与 −28° 都被保留，但 C1、C2 的全频优势仅 0.29 和 0.60 dB。两个候选本身都未落在理论 +52.00° 的 ±8° 内。此前六态全角域的 +55° 不在这次短名单内，排序阶段无法把它补回来。",
        "- **E。** G2 的 +47° 与 +45° 真值相差 2°，长基线的全频排序反而倾向 −34°，偏差 79°。C2 对 −34° 的优势仅 0.65 dB，换背景后改为偏向 +47°。只看 C1 或只看有人段，会把这里的角度歧义低估。",
        "",
        "## 两项单独列出的对照",
        "",
        "**生成候选时加入 G3。** 这是在共用 G2 主试验之后单独冻结的提议阶段敏感性检查。仍然以 C1 配 G4、C2 配 G1 独立复核，没有改变分离度或放行阈值。短状态硬件消融用同样的 G2/G3 候选，将 C1/C2 观测槽换成等量 G2 短基线观测。两种硬件的这组候选完全相同。",
        "",
        "| 点 | G2+G3 生成的候选 | 长基线全频暂倾向 / 偏差 | 仅短状态硬件暂倾向 | 最终 |",
        "|---|---|---:|---:|---|",*sensitivityrows,
        "",
        "B 出现了更好的 −58° 候选（差 3.31°），但长基线倾向 +5°。D 只生成 +40°，两条稀疏校验和留出复核仍不足，不能计作消歧成功。E 的 +48° 候选在短状态硬件中已经排在前面。这个对照仍然没有新增经验证的单解。",
        "",
        "**改用最强正变化一阶 MUSIC。** 主设计已预先登记此敏感性。保留同一 1 dB 规则，和背景投影主结果分开列出。",
        "",
        "| 点 | 此模型自己的两个候选 | 三根暂倾向 / 偏差 | 三根放行 | 十二根暂倾向 / 偏差 | 十二根放行 |",
        "|---|---|---:|---:|---:|---|",*other,
        "",
        "E 是一个可指出的防错例子。三根同流程放行 −30°（差 75°），十二根保留双候选，把排序移向 +26°（差 19°）并拒绝输出单角。另一方面，D 的三根同流程放行 +45°（差 7.00°），十二根将这一单解也退回双候选。所以这里只有一次避免错误输出，同时损失一次接近真实站位的输出。B 的暂倾向转到 −90° 边界不能作为人体方向。不得事后按点混用两种模型。",
        "",
        "## 采样、判别规则与证据范围",
        "",
        "- 相同采时 10 s，R1、R2、R3 分别计数。每个 2.5 s 时间块从每个用途分配等量包，数量仅由可用包数决定，按时间均匀抽取。一个方案内每个包只用一次。G2 单独包括全包的传统主峰基线，另以六份不重叠的 G2 包模拟相同的提候选/判别/复核流程。十二根的六份分别来自 G2、G3、C1、C2、G4、G1。各站位两臂候选用到的 G2 包序号哈希逐一相同。短状态硬件消融保留四个 G，用 G2 的另外两份包替换长基线包。不同状态非同时观测，未相干拼接。",
        "- 每条 MUSIC 曲线全域搜索。局部峰须处于最佳峰的 6 dB 内，最多保留两个相隔至少 20° 的候选，端点标记后禁止放行。B、C、E 在主试验另有排第三的短基线峰，分别为 +10°、−90°、−90°，未因看到真值而重新纳入。",
        "- C1/C2 在每个候选 ±9° 邻域内比较残差。两个邻域不重叠。主阈值 1 dB，另查 0 和 3 dB。它是本次工程阈值，尚无独立标定来确立其置信水平。要求两个长状态全频、两半频段和至少三个时间块一致，且至少有一对物理不重叠的‘C1 与 G4’或‘C2 与 G1’稳定复核。C1、C2 共用 6/8，不能当作两次独立人体观测。G3 负责辅助反证，换 R3 背景后仍需支持。用 R3 验证附近的稳定峰，同时检查 R3 对照可评估性与 R2/R3 强度比。",
        "- 十二根主模型在 0、1、3 dB 三档均为 0/5 单解。各点只有一个独立进场，窗口内部和角色拆分都不增加独立样本数。这次确定的是该实验规则在这五点的行为，人体 AoA 准确率仍需独立测量。保留双候选也不表示已经发现两个人体反射。",
        "",
        "[五点核心表](five_point_comparison.csv) · [每个候选的拒绝原因](branch_checks.csv) · [分频段及时间块投票](role_spectra_checks.csv) · [三种配置和阈值的全部结果](all_scored_decisions.csv) · [双短子阵提议对照](two_short_scored_decisions.csv) · [覆盖率与误差口径](coverage_and_sensitivity.csv) · [包数和序号哈希](packet_allocations.csv) · [主试验冻结规则](protocol.json) · [候选角理想相似度](ideal_alias_at_seeds.csv)。",
    ]
    path=ROOT/"五点短基线提候选与长基线消歧.md"
    path.write_text("\n".join(prose)+"\n",encoding="utf8")
    for link in re.findall(r'\]\(([^)]*)\)',path.read_text(encoding="utf8")):
        assert (ROOT/link).is_file()
    print(summary[['point','G2_full_peak','seed1','seed2','long_preference','long_preference_error','long_decision']].to_string(index=False))
    print("Five matched station budgets, all frozen output hashes and report links checked.")


if __name__=="__main__":report()

