"""Compare saved peak locations with survey geometry; keep MUSIC and background scores distinct."""
import sys
from pathlib import Path
from contextlib import ExitStack
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True
from aggregate import FOLDERS, POINTS, PRIMARY, reviewer, validate_frozen

METHODS = {"per_tone":"逐频参考空间 MUSIC","raw":"原始空间谱",
           "r1_template":"常数或线性参考空间谱","joint":"原始二维 MUSIC"}
SITES = {s:Path(pd.read_csv(d/"raw_runs.csv").source.iloc[0]).parents[2].name for s,d in FOLDERS.items()}


def save(file,frame):
    frame.to_csv(HERE/file,index=False,encoding="utf-8-sig")


def curve(z,state,method,run):
    if method=="joint":
        return z["joint"][f"{state}_R{run}"].max(axis=1)
    if method=="per_tone":
        if state=="G fusion":
            return np.mean([z["emp"][f"G{i}_R{run}"] for i in range(1,5)],axis=0)
        return z["emp"][f"{state}_R{run}"]
    if state=="G fusion":
        return z["space"][f"fusion_R{run}_{method}"]
    return z["space"][f"{state}_R{run}_{method}"][0]


def music_catalog(bc):
    rows=[]; traces={}
    for site,p in FOLDERS.items():
        theory,_=bc.geometry(site)
        raw=pd.read_csv(p/"blind_angular_maxima.csv")
        per=pd.read_csv(p/"exploratory_empirical_angular_peaks.csv")
        joint=pd.read_csv(p/"blind_joint_maxima.csv")
        phase=pd.read_csv(p/"empirical_repeatability.csv")
        model_phase=pd.read_csv(p/"ref_run3_validation.csv")
        with ExitStack() as stack:
            spectra={k:stack.enter_context(np.load(p/(stem+".npz"))) for k,stem in
                     [("space","angular_spectra"),("joint","joint_spectra"),("emp","exploratory_empirical_spectra")]}
            angles=spectra["space"]["angles"]
            assert np.array_equal(angles,np.arange(-90.,91.,3.))
            assert all(np.array_equal(z["angles"],angles) for z in spectra.values())
            inputs=[("per_tone",per,"dominant_deg"),("raw",raw.query('mode=="raw"'),"peak_deg"),
                    ("r1_template",raw.query('mode=="r1_template"'),"peak_deg"),("joint",joint,"angle_parameter_deg")]
            for mode,df,column in inputs:
                for state,group in df.groupby("state"):
                    group=group.set_index("run")
                    assert set(group.index)=={1,2,3}
                    curves=np.stack([curve(spectra,state,mode,r) for r in (1,2,3)])
                    indices=curves.argmax(axis=1)
                    peaks=angles[indices]
                    assert np.array_equal(group.loc[[1,2,3],column].to_numpy(),peaks)
                    if mode=="joint":
                        for run in (1,2,3):
                            surface=spectra["joint"][f"{state}_R{run}"]
                            i,j=np.unravel_index(surface.argmax(),surface.shape)
                            assert i==indices[run-1] and spectra["joint"]["tau_ns"][j]==group.loc[run,"uncalibrated_delay_ns"]
                    profile=10*np.log10(np.maximum(curves/curves.max(axis=1,keepdims=True),1e-12))
                    traces[site,state,mode]=angles.copy(),profile
                    low=high=np.nan
                    if state!="G fusion" and mode=="per_tone":
                        low,high=group.loc[2,["low_band_dominant_deg","high_band_dominant_deg"]]
                    if state!="G fusion" and mode in ("raw","r1_template"):
                        low,high=angles[spectra["space"][f"{state}_R2_{mode}"][1:].argmax(axis=1)]
                    checks=phase if mode=="per_tone" else model_phase
                    passed=int(checks.loc[checks.state==state,"pass_pair"].sum()) if mode in ("per_tone","r1_template") and state!="G fusion" else np.nan
                    rows.append(dict(point=site,source=SITES[site],method=mode,state=state,theory_deg=theory[state],
                                r1_empty_peak_deg=peaks[0],r2_occupied_peak_deg=peaks[1],r3_empty_peak_deg=peaks[2],
                                abs_difference_deg=abs(peaks[1]-theory[state]),r2_low_half_deg=low,r2_high_half_deg=high,
                                ref_pairs_passed_of_two=passed,
                                r1_db_at_r2_angle=profile[0,indices[1]],r3_db_at_r2_angle=profile[2,indices[1]],
                                r2_boundary=abs(peaks[1])>=87))
    result=pd.DataFrame(rows)
    # 24 single-triplet estimates and 3 noncoherent fusion estimates for each visit.
    assert (result.groupby("point").size()==27).all()
    assert (result[result.state!="G fusion"].groupby("point").size()==24).all()
    save("music_main_peaks.csv",result)
    return result,traces


def background_catalog(bc):
    rows=[]
    for site,p in FOLDERS.items():
        theory,_=bc.geometry(site)
        for method,stem,col in [("raw_BG","blind_background_candidates","angle_parameter_deg"),
                                 ("per_tone_BG","exploratory_empirical_bg_peaks","angle")]:
            data=pd.read_csv(p/(stem+".csv")).rename(columns={col:"peak_deg"})
            data["rank_all_retained"]=data.groupby(["state","reference","test"]).ratio.rank(method="min",ascending=False)
            # With a common scalar gate, a nonempty retained list includes the full-grid maximum.
            # Verify the stored raw-background curves, including endpoints and unstable peaks.
            if method=="raw_BG":
                with np.load(p/"background_scores.npz") as z:
                    for (state,ref,test),group in data.groupby(["state","reference","test"]):
                        scores=z[f"{state}_ref{ref}_R{test}"][0]
                        assert np.isclose(group.ratio.max(),scores.max())
                        indices=np.searchsorted(z["angles"],group.peak_deg)
                        assert np.allclose(group.ratio,scores[indices])
            for row in data[data.test==2].itertuples():
                control=data[(data.reference==row.reference)&(data.test==4-row.reference)&(data.state==row.state)]
                nearby=control[(control.peak_deg-row.peak_deg).abs()<=9]
                rows.append(dict(point=site,source=SITES[site],method=method,state=row.state,reference=row.reference,
                            r2_peak_deg=row.peak_deg,theory_deg=theory[row.state],abs_difference_deg=abs(row.peak_deg-theory[row.state]),
                            rank_in_r2=row.rank_all_retained,score=row.ratio,passes_both_halves=row.both_bands,
                            supported_quarters=row.quarter_support,stable=row.stable,
                            nearby_empty_peaks_deg="/".join(str(int(a)) for a in nearby.peak_deg),
                            nearby_stable_empty_peaks_deg="/".join(str(int(a)) for a in nearby.loc[nearby.stable,"peak_deg"])))
    out=pd.DataFrame(rows); save("background_peaks_ranked.csv",out)
    return out


def closest(data,sites):
    # This is a retrospective survey comparison, never used as a direction estimator.
    ordered=data.sort_values("abs_difference_deg",kind="stable")
    return ordered.drop_duplicates("point").set_index("point").loc[list(sites)]


def deg(x,places=0):
    if np.isclose(x,0): return "0°"
    return f"{x:+.{places}f}°".replace("-","−")


def seq(row):
    return " → ".join(deg(a) for a in [row.r1_empty_peak_deg,row.r2_occupied_peak_deg,row.r3_empty_peak_deg])


def line(fields):
    return "| "+" | ".join(map(str,fields))+" |"


def draw(best,traces,outfile):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":9.5,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(12,6.65),layout="constrained")
    for ax,(site,row) in zip(axes.flat,best.iterrows()):
        x,curves=traces[site,row.state,row.method]
        for values,name,colour,style in zip(curves,["前空场","人在场","后空场"],["#44769b","#bf5628","#197d70"],["--","-",":"]):
            ax.plot(x,values,color=colour,ls=style,lw=1.8,label=name)
        ax.axvline(row.theory_deg,color="#372a45",ls=":",lw=1.4,label="子阵理论方向")
        ax.scatter(row.r2_occupied_peak_deg,0,s=27,color="#bf5628",zorder=4)
        mode={"per_tone":"逐频参考","raw":"原始空间谱","joint":"二维 MUSIC"}[row.method]
        ax.set_title(f"{site} · {row.state} {mode}　实测 {deg(row.r2_occupied_peak_deg)}",fontsize=10.8)
        ax.set(xlim=(-90,90),xticks=[-90,-45,0,45,90],ylim=(-10.5,.5),
               xlabel="RX 角度参数 / °",ylabel="相对本轮最高峰 / dB")
        ax.grid(alpha=.18)
    ax=axes.flat[-1];ax.axis("off")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    ax.legend(handles,labels,loc="center left",bbox_to_anchor=(.08,.65),frameon=False,ncol=2)
    ax.text(.10,.41,"每点从 24 项既有 MUSIC 输出中\n事后展示最接近理论角的一项。\n二维谱沿时延取最大值。",transform=ax.transAxes,va="top",fontsize=10.2,linespacing=1.65)
    fig.suptitle("五点各自的三轮实测谱",fontsize=14)
    fig.savefig(outfile,dpi=175);plt.close(fig)


def write_report(cat,bg,traces):
    single=cat[(cat.state!="G fusion") & cat.point.isin(PRIMARY)]
    empirical=single[single.method=="per_tone"]
    g3=empirical[empirical.state=="G3"].set_index("point")
    same=closest(empirical,PRIMARY)
    best=closest(single,PRIMARY)
    count=single[single.abs_difference_deg.le(8)].groupby("point").size()
    peak_columns=["source","state","method","theory_deg","r1_empty_peak_deg","r2_occupied_peak_deg",
                  "r3_empty_peak_deg","abs_difference_deg"]
    summary=g3.loc[list(PRIMARY),peak_columns].add_prefix("fixed_G3_").join(best[peak_columns].add_prefix("retrospective_"))
    summary["single_triplet_MUSIC_outputs_within_8_deg"]=count.reindex(PRIMARY,fill_value=0)
    summary["single_triplet_MUSIC_outputs_examined"]=24
    save("five_point_summary.csv",summary.reset_index())
    common=[];examples=[]
    for site in PRIMARY:
        g=g3.loc[site]; e=same.loc[site]; b=best.loc[site]
        common.append(line([site,deg(g.theory_deg,2),seq(g),f"{g.abs_difference_deg:.2f}°",f"{int(g.ref_pairs_passed_of_two)}/2",
                            f"{e.state} {deg(e.r2_occupied_peak_deg)}（{e.abs_difference_deg:.2f}°）"]))
        examples.append(line([site,b.source,b.state,METHODS[b.method],deg(b.theory_deg,2),seq(b),f"{b.abs_difference_deg:.2f}°",f"{count.get(site,0)}/24"]))
    # Check the ranks and negative controls used to interpret the shortlist.
    def branch(point,state,method,ref,angle):
        v=bg[(bg.point==point)&(bg.state==state)&(bg.method==method)&(bg.reference==ref)&(bg.r2_peak_deg==angle)]
        assert len(v)==1
        return v.iloc[0]
    a=branch("A","C1","raw_BG",1,-12)
    c1=branch("C","C2","per_tone_BG",1,18); c3=branch("C","C2","per_tone_BG",3,24)
    e1=branch("E","C1","per_tone_BG",1,42); e3=branch("E","C1","per_tone_BG",3,42)
    assert a.rank_in_r2>1 and a.stable and "-15" in a.nearby_stable_empty_peaks_deg
    assert all(v.stable and not v.nearby_stable_empty_peaks_deg for v in [c1,c3,e1,e3])
    assert (c1.rank_in_r2,c3.rank_in_r2,e1.rank_in_r2,e3.rank_in_r2)==(2,2,1,1)
    assert all("42" in v.nearby_empty_peaks_deg for v in (e1,e3))
    old_g=closest(cat[(cat.state!="G fusion") & cat.point.str.contains("earlier") & (cat.method=="per_tone")],["B_earlier","C_earlier"])
    assert old_g.loc["B_earlier","state"]=="C1" and old_g.loc["B_earlier","r2_occupied_peak_deg"]==-63
    small=[]
    for row,note in [
        (a,"后空场有 −15° 稳定峰。"),
        (branch("B_earlier","G2","raw_BG",3,-54),"来自旧 New B。换前空场为参考，主峰在 +15°。"),
        (c3,"前空场作背景为 +18°（差 5.55°），两次都排第二。"),
        (branch("D","C2","raw_BG",1,54),"后空场也有 +57° 稳定峰。"),
        (branch("E","C2","per_tone_BG",1,45),"后空场也有 +42° 稳定峰。换背景后 +45° 仅获 1/4 时间块支持。")
    ]:
        small.append(line([row.point.split("_")[0],row.source,row.state,f"R{row.reference}",deg(row.r2_peak_deg),
                           deg(row.theory_deg,2),f"{row.abs_difference_deg:.2f}°",int(row.rank_in_r2),note]))
    draw(best,traces,HERE/"五点主峰与前后空场.png")
    lines=[
        "# A–E 五点接近理论方向的实测主峰",
        "",
        "在原先固定的 G3 方法下，只有 A 的有人主峰距 G3 理论方向不超过 8°。若把每个点的六个三阵元子阵和四种既有 MUSIC 输出全部展开，A、B、D、E 都有落在 8° 内的全谱主峰，C 的最近值为 9.95°。背景处理下还可以看到更接近的局部峰，两类结果分开记录。",
        "",
        "各点都以最新的一次完整进出为主，R1 无人、R2 人站定、R3 再次无人。TX=(0,0)，RX=(3.65,0)，新 E 在 (1.64,−2.01)。角度从 RX 指向 TX 的方向起算，取实际子阵中心。只读取已有的全带宽谱、原始峰表和空场检查，选峰规则及参数均沿用原分析。",
        "",
        "## 一、完全沿用 A 点 G3 的方法",
        "",
        "| 点 | G3 理论角 | 前空场 → 有人 → 后空场主峰 | R2 偏差 | 后空场通过的相对链 | 同样逐频参考下，六子阵中最接近的 R2 主峰（偏差） |",
        "|---|---:|---|---:|---:|---|",*common,
        "",
        "最后一列保留相同处理，只在看过真值后比较六个子阵。D 的 +87° 已落入预设的边界区域。原先截图的 B、C 来自较早的 New B、New C，其固定 G3 偏差分别为 59.72°、26.55°。",
        "",
        "## 二、五点各自有哪些全角域 MUSIC 主峰最接近站位",
        "",
        "每批逐一复核 24 项输出（G1–G4、C1、C2 × 原始空间谱、逐频参考空间谱、常数或频率线性参考空间谱、三阶联合 AoA–ToF MUSIC）。联合谱在整个角度–时延网格取最大值。只有本来就是全谱最大值的峰可以进入本表。先核对保存的主峰与谱数组，再按对应物理子阵的理论角做事后排序，同误差时优先保留逐频参考口径。四组非相干融合和两次较早的 B、C 保留在明细中，另行标记。",
        "",
        "| 点 | 批次 | 子阵 | 方法 | 子阵理论角 | 前空场 → 有人 → 后空场主峰 | R2 偏差 | 24 项中偏差 ≤8° 的项数 |",
        "|---|---|---|---|---:|---|---:|---:|",*examples,
        "",
        "各点展示的子阵和方法不同，24 项还共享同一个有人窗口。表格用于找出可复验的分支，不能用选出的五个偏差计算定位算法的 MAE 或命中率。",
        "",
        "![五点代表性实测谱](五点主峰与前后空场.png)",
        "",
        "## 三、结合空场和峰排名判断每个点",
        "",
        "- **A。** G3 的 0° → −6° → 0° 来自同一子阵和逐频参考的全谱主峰。两条相对链都通过后空场检查，但有人谱的 0° 仅比 −6° 低 0.61 dB。同批 C1 的 −12° 与理论 −12.69° 只差 0.69°，却是前空场作背景时的第 5 个候选（按全频分数降序）。后空场也有 −15° 的稳定峰。",
        "- **B。** 新 B 的 G3 原始二维谱在有人时选 −60°，理论 −53.72°，差 6.28°。在两次空场里，−60° 处的角谱值也只比各自最高峰低 0.16 和 0.51 dB。旧 B 的 C1 更接近 A 的例子，用相同的逐频参考得到 0° → −63° → 0°，理论 −55.61°，差 7.39°。这条分支在新 B 仍是 R2 的次峰，离主峰仅 0.07 dB，但新 B 后空场的 −63° 也仅低其主峰 0.12 dB。",
        "- **C。** 本批 G4 原始空间谱为 +3° → +15° → +3°。R2 低、高半频段主峰分别为 +15°、+12°。这是稳定而真实的峰移动，全带宽峰与 +24.95° 的理论方向仍相差 9.95°。C2 逐频参考并白化后，分别以前、后空场作背景，R2 的正角候选为 +18°、+24°（偏差 5.55°、0.45°），两者各排第 2，实际最高的是 −48°、−36°。两正角候选通过时间块和半频段检查，未在对应空场发现邻近稳定峰。切换背景复用了同一段人体数据，且稀疏 C2 有方向歧义。G1 的 +27° 则是前空场作背景时的最高得分（偏差 6.35°），但后空场有 +33° 稳定峰，换背景后 R2 主峰为 +9° 且未达稳定门槛。",
        "- **D。** G2 从 −36° 移到 +54° 又回到 −36°，人体窗口与理论 +51.26° 的差为 2.74°。低、高半频段改选 +66°、−27°，后空场仍有 +48° 次峰。C2 逐频参考并白化后，两种背景下均以 +48° 为最高得分（理论 +52.72°，差 4.72°），近等分的 −18° 分支也在。两方向的理想阵列响应相似度约为 0.990。",
        "- **E。** G4 的原始二维主峰为 +45° → +42° → +45°，R2 距 G4 理论 +46.74° 为 4.74°，空场已经存在近似方向。另有 C1 的 +42°，分别以前、后空场作背景都是有人段的最高白化得分，距 C1 理论 +44.40° 为 2.40°。两次比较的有人峰均通过两半频段和 4/4 时间块检查。两组对应空场也都有 +42° 局部峰，但未通过两半频段检查。C1 的两条相对链 R3 相位残差 P90 为 17.14°、18.52°，均超过预设的 15°。",
        "",
        "## 四、背景处理中数值更接近的候选",
        "",
        "同一子阵的排名按有人窗口全频得分从高到低计算，计入端点和未通过稳定性检查的峰。R1、R3 分别是前、后空场。各行注明来源，其中 B 来自较早的重复采集。表内数值均为已保存候选与真值的事后对照。",
        "",
        "| 点 | 批次 | 子阵 | 背景参考 | 有人候选 | 子阵理论 | 偏差 | 谱内排名 | 交叉检查 |",
        "|---|---|---|---|---:|---:|---:|---:|---|",*small,
        "",
        "这批数据最值得继续核验的是 D 的 G2 峰移动和 E 的 C1 得分峰。C 应同时保留正负分支，B 应保留旧、新批次间不一致的记录。总体人体方向精度、相干十二阵元增益及物理位置误差均未由本次回顾确定。",
        "",
        "[五点主峰汇总](five_point_summary.csv) · [所有主峰、空场谱值和频段明细](music_main_peaks.csv) · [所有白化候选、排名和空场检查](background_peaks_ranked.csv) · [原始批次索引](../record_index.csv) · [方法与初次汇总](../ABCDE_理论与实测方向.md)。"
    ]
    report=HERE/"五点实测主峰与候选.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf8")
    links=re.findall(r'\]\(([^)]*)\)',report.read_text(encoding="utf8"))
    assert all((HERE/name).is_file() for name in links)
    print(best[["state","method","theory_deg","r1_empty_peak_deg","r2_occupied_peak_deg","r3_empty_peak_deg","abs_difference_deg","r1_db_at_r2_angle","r3_db_at_r2_angle"]].round(3).to_string())
    print("Verified",len(cat),"MUSIC rows and",len(bg),"foreground background-score candidates; report links passed.")


if __name__=="__main__":
    bc=reviewer().bc
    validate_frozen(bc)
    cat,traces=music_catalog(bc)
    bg=background_catalog(bc)
    write_report(cat,bg,traces)

