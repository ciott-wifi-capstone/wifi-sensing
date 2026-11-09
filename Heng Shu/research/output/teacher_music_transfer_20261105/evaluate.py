"""Join survey truth only after verifying the frozen peak registry."""
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
sys.path.insert(0,str(OUT/"paired_be_abcde_20261105"))
from aggregate import reviewer
bc=reviewer().bc
PRIMARY=list("ABCDE")
SITES=[*PRIMARY,"B_old","C_old"]
SOURCE=Path.home()/"xwechat_files"/"wxid_pmqwggzooxbz22_9e82"/"msg"/"file"/"2026-11"/"WiCAL_ACDE_Teacher_Report(2).pdf"
PEER={
    "A":(+11.89,+14,2.11,+31,19.11,16.38,12.01,17.07,31855,20780),
    "C":(-22.83,+15,37.83,-24,1.17,5.88,4.22,36.94,31610,20603),
    "D":(-52.00,-67,15.,-54,2.00,4.97,4.76,36.77,31788,20726),
    "E":(-44.86,-21,23.86,-38,6.86,4.83,1.63,56.07,30772,20138),
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name):
    return pd.read_csv(ROOT/(name+".csv"))


def save(name,df):
    df.to_csv(ROOT/(name+".csv"),index=False,encoding="utf-8-sig")


def geometry(site,group):
    angles,_=bc.geometry(site[:1])
    return angles[group] if group in bc.STATES else angles["G fusion"]


def format_angle(v):
    return "0°" if v==0 else f"{v:+.0f}°".replace("-","−")


def row(fields):
    return "| "+" | ".join(map(str,fields))+" |"


def gate(summary,peaks,power):
    records=[]
    series=peaks.query('clock=="aligned" and model=="delta" and group=="S6_product"')
    for site in SITES:
        p=series[series.point==site].set_index(["reference","test","block","band"]).angle
        def val(ref,r,block="full",band="full"):
            return p.loc[ref,r,block,band]
        target,empty,reverse=val(1,2),val(1,3),val(3,2)
        def checks(r):
            peak=val(1,r)
            halves=[val(1,r,"full",band) for band in ["low","high"]]
            quarters=[val(1,r,f"Q{i}") for i in range(1,5)]
            return all(abs(z-peak)<=9 for z in halves),sum(abs(z-peak)<=9 for z in quarters),halves,quarters
        ht,qt,hp,qseq=checks(2)
        he,qe,eh,eqseq=checks(3)
        near_empty=abs(target-empty)<=9
        ratio=power.query('point==@site and clock=="aligned" and scope=="S6" and reference==1').set_index("test").contrast_peak_power
        strength=ratio.loc[2]/ratio.loc[3]
        failures=[]
        if not ht:failures.append("R2两半频段分歧")
        if qt<3:failures.append("R2时间分块不足")
        if abs(reverse-target)>9:failures.append("换背景移峰")
        if he and qe>=3 and near_empty:failures.append("邻近空场峰稳定")
        if strength<=1:failures.append("变化强度未超过空场")
        if abs(target)>=87:failures.append("端点")
        match=summary[summary.point==site].set_index(["model","group"])
        initial=match.loc["raw","G4_mean"].angle
        serial=peaks.query('point==@site and clock=="serial" and reference==1 and test==2 and block=="full" and band=="full" and model=="delta" and group=="S6_product"').angle.item()
        expected=geometry(site,"S6_product")
        records.append(dict(point=site,centre_truth_deg=expected,raw_four_mean_deg=initial,
                            raw_four_error=abs(initial-expected),primary_deg=target,primary_error=abs(target-expected),
                            r3_empty_peak_deg=empty,reverse_reference_deg=reverse,R2_low=hp[0],R2_high=hp[1],
                            R2_quarter_peaks="/".join(str(int(a)) for a in qseq),R2_quarter_count=qt,
                            R3_low=eh[0],R3_high=eh[1],R3_quarter_peaks="/".join(str(int(a)) for a in eqseq),
                            R3_near_R2=near_empty,R3_stable=he and qe>=3,
                            strength_R2_over_R3=strength,serial_clock_peak_deg=serial,passes_gate=not failures,
                            failure="、".join(failures) or "通过本轮诊断"))
    return pd.DataFrame(records)


def figures(results):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":9.5,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(11.9,6.4),layout="constrained")
    with np.load(ROOT/"spectra.npz") as spec:
        for ax,r in zip(axes.flat,results.itertuples()):
            for test,name,color,style in [(2,"有人 R2","#bd5624","-"),(3,"后空场 R3","#197e75","--")]:
                p=spec[f"{r.point}_delta_S6_product_R{test}"]
                ax.plot(spec["angles"],10*np.log10(np.maximum(p/np.max(p),1e-12)),c=color,lw=1.9,ls=style,label=name)
                idx=np.argmax(p)
                ax.scatter(spec["angles"][idx],0,c=color,s=25,zorder=4)
            ax.axvline(r.centre_truth_deg,color="#302244",lw=1.3,ls=":",label="阵列中心理论角")
            ax.set(xlim=(-90,90),xticks=[-90,-45,0,45,90],ylim=(-3.4,.24),xlabel="角度 / °",ylabel="各轮相对谱峰 / dB")
            ax.set_title(f"{r.point}　有人 {format_angle(r.primary_deg)}　后空 {format_angle(r.r3_empty_peak_deg)}")
            ax.grid(alpha=.16)
    ax=axes.flat[-1];ax.axis("off")
    h,l=axes.flat[0].get_legend_handles_labels()
    ax.legend(h,l,loc="upper left",bbox_to_anchor=(.09,.98),frameon=False)
    ax.text(.12,.37,"同一 R1 背景，六态非相干乘积。\n两轮分别归一化，仅比较谱形与峰位。\n人体/空场强度比另用共同尺度计算。",transform=ax.transAxes,va="top",fontsize=9.4,linespacing=1.7)
    fig.suptitle("去背景后，五个点的人体与后空场 MUSIC 谱",fontsize=14)
    fig.savefig(ROOT/"五点人体与后空场角谱.png",dpi=175);plt.close(fig)


def main():
    registry=json.loads((ROOT/"blind_registry.json").read_text(encoding="utf8"))
    assert all(digest(ROOT/k)==v for k,v in registry.items())
    assert digest(SOURCE)=="670d373829201910496088d900b06b752ec8df714a6c481e506accd7cf5f620b"
    peaks=read("blind_peaks")
    summary=peaks.query('clock=="aligned" and reference==1 and test==2 and block=="full" and band=="full"').copy()
    summary["theory_deg"]=[geometry(p,g) for p,g in zip(summary.point,summary.group)]
    summary["error_deg"]=(summary.angle-summary.theory_deg).abs()
    primary=summary[summary.point.isin(PRIMARY)]
    agg=primary.groupby(["model","group"]).error_deg.agg(mean_abs="mean",median_abs="median",worst="max",count="size",
                    within8=lambda x:int((x<=8).sum())).reset_index()
    assert len(summary)==len(SITES)*4*11 and agg["count"].eq(5).all()
    save("ablations",agg);save("scored_full_runs",summary)
    power=read("contrast_strength")
    results=gate(summary,peaks,power)
    save("site_results",results)
    audit=read("packet_audit")
    assert audit.sha256.nunique()==21 and len(audit)==21*2*6
    peer=[]
    for name,vals in PEER.items():
        tr,g,ge,s,se,gr,sr,p95,total,used=vals
        a=audit.query('point==@name and clock=="aligned"')
        local_raw=a.drop_duplicates("run").raw_frames.sum()
        assert local_raw==total
        peer.append(dict(site=name,peer_centre_theory_original=tr,peer_G3_original=g,peer_G3_our_sign=-g,
                         peer_G3_error_as_reported=ge,peer_six_original=s,peer_six_our_sign=-s,peer_six_error_as_reported=se,
                         peer_six_error_against_local_geometry=abs(-s-geometry(name,"S6_product")),
                         peer_G3_strength_ratio=gr,peer_six_strength_ratio=sr,peer_G3_P95=p95,
                         peer_raw_frames=total,local_raw_frames=local_raw,peer_used=used,local_used=int(a.selected.sum()),
                         local_phy=a.phy.iat[0],local_ifindex=a.ifindex.iat[0]))
    peer=pd.DataFrame(peer);save("peer_comparison",peer)
    latest=results[results.point.isin(PRIMARY)]
    eigen=read("eigen_diagnostics").query('clock=="aligned" and model=="delta" and reference==1 and test==2')
    diag=eigen[eigen.point.isin(PRIMARY)][["usable_tones","negative_tones","multiple_positive_tones"]].sum()
    arch=pd.read_csv(OUT/"paired_be_abcde_20261105"/"ABCDE_angles.csv")
    detail=[];audit_rows=[]
    for r in latest.itertuples():
        detail.append(row([r.point,f"{r.centre_truth_deg:+.2f}°",format_angle(r.raw_four_mean_deg),format_angle(r.primary_deg),
                           f"{r.primary_error:.2f}°",format_angle(r.r3_empty_peak_deg),f"{r.strength_R2_over_R3:.2f}×",r.failure]))
        audit_rows.append(row([r.point,f"{format_angle(r.R2_low)} / {format_angle(r.R2_high)}",r.R2_quarter_peaks,
                               format_angle(r.reverse_reference_deg),format_angle(r.serial_clock_peak_deg)]))
    arms=[("raw","G4_mean"),("raw","S6_product"),("delta","G4_mean"),("delta","G4_product"),
          ("delta","S6_mean"),("delta","S6_product"),("delta","S6_product_coarse"),("white","S6_product"),("phase_delta","S6_product")]
    names=["原始四 G 算术平均","原始六态乘积","R1 差分四 G 平均","R1 差分四 G 乘积","R1 差分六态平均","R1 差分六态乘积（主分支）",
           "主分支加自定粗筛","R1 白化 MUSIC、六态乘积","加 R1 逐频相位参考、差分六态乘积"]
    a=agg.set_index(["model","group"])
    armrows=[]
    for (m,g),name in zip(arms,names):
        v=a.loc[m,g]
        armrows.append(row([name,f"{v.mean_abs:.2f}°",f"{v.median_abs:.2f}°",f"{v.worst:.2f}°",f"{int(v.within8)}/5"]))
    frows=[]
    for r in peer.itertuples():
        frows.append(row([r.site,format_angle(r.peer_G3_our_sign),format_angle(r.peer_six_our_sign),
                          f"{r.peer_six_error_as_reported:.2f}°",f"{r.peer_raw_frames}",f"{r.peer_used} / {r.local_used}"]))
    g3rows=[]
    g3blind=peaks.query('clock=="aligned" and model=="white" and group=="G3" and reference==1 and block=="full"')
    for site in PRIMARY:
        z=g3blind[g3blind.point==site].set_index(["test","band"]).angle
        actual=geometry(site,"G3")
        g3rows.append(row([site,f"{actual:+.2f}°",format_angle(z[2,"full"]),f"{abs(z[2,'full']-actual):.2f}°",
                           f"{format_angle(z[2,'low'])} / {format_angle(z[2,'high'])}",format_angle(z[3,"full"])]))
    figures(latest)
    out=[
      "# 把朋友报告的空场差分和六态融合用于我们的 MUSIC",
      "",
      "在五个最新站位统一使用 R1 协方差差分、单正特征方向 MUSIC 和六态非相干谱乘积，A、D 的全频峰接近站位。相同时间窗和网格下，五点平均绝对谱峰偏差从原始四 G 平均基线的 28.33° 降至 17.87°，中位偏差却从 19.11° 升至 22.31°。改善主要来自 B 的方向由 +33° 转到 −77°。C、E 均比原始四 G 基线偏得更远。六态平均的平均偏差为 17.72°，乘积本身没有带来额外的总体改善。",
      "",
      "## 朋友报告实际支持什么",
      "",
      "第 1、2 页使用逐子载波的单路径 AoA-only SAGE 估 G3，用固定粗候选域内的六态谱乘积估融合角。报告表里的角以图上 +y 为正，与我们的符号相反。下表已把峰值转换为我们的符号。报告对单组及融合都用阵列中心角，我们对单组另用子阵中心。E 采用 (1.64,−2.00)，我们据现场图用 (1.64,−2.01)。报告的 E 六态偏差 6.86°，换到我们的坐标为 7.00°。",
      "",
      "| 点 | 报告 G3 峰（换符号） | 报告六态峰（换符号） | 报告六态条件偏差 | 原始帧（两边相同） | 报告估计用帧 / 我们采用的帧 |",
      "|---|---:|---:|---:|---:|---:|",*frows,
      "",
      f"报告固定用六态差分时，ACDE 的平均条件偏差约 {peer.peer_six_error_as_reported.mean():.2f}°。A 六态为 19.11°，A 的 2.11° 来自另一条 G3 路线。报告说明 B 是查看结果后排除。它的 840 次检查验证实现的一致性，各点之间没有 840 次独立人体进场。第 3 页的零区间来自候选遮罩。第 4 页用六个未归一化子谱的均值比较强度，真正选方向用的是乘积。E 的六态人体/空场强度仅 1.63 倍，是四点中最弱的。报告给出 G3 空场相位漂移 P95 为 17.07°–56.07°，四点均超过其 10° 参考线。前空场的主相位还包含真实传播与室内多径，不能据此认定已经分离固定硬件相位。",
      "",
      "## 移植到 MUSIC 的固定实现",
      "",
      "保留原来的 Intel 5300 mask-3 AntSEL 解码和物理阵元映射，先核对原始 SHA、状态序号、采集参数及窗口目标包。每包按 3 链 × 30 频率组的整体均方功率归一化。按各点自己的 R1 计算每状态、每频率组 R₁，再分别用 ΔR₂=R₂−R₁ 和 ΔR₃=R₃−R₁。协方差来自同包外积，避免了跨记录直接相减复数 CSI 所需的绝对包相位参考。",
      "",
      f"ΔR 是有符号的 Hermitian 差矩阵。五个最新站位共 {int(diag.usable_tones)} 个状态×频组中，有 {int(diag.negative_tones)} 个含负特征值，{int(diag.multiple_positive_tones)} 个含至少两个正特征值。主分支明确只取最大正特征值的特征向量 u。对实际阵元导向矢量 a 计算 P(θ)=1 / meanₖ[1−|uᴴa|²/‖a‖²]。这里的正方向表示增强的一种空间分量。人体遮挡形成的负分量、静态场和人体场的交叉项，以及没有角度标定的硬件相位，都可能改变结果。记录了全部频组的正负特征值统计，同时保留以 10% 对角加载 R₁ 白化正定协方差、并一致变换导向矢量的 MUSIC 对照。",
      "",
      "六态融合对各子谱先按自身最大值归一化，主分支全角域最大化六个 log P 的等权均值，这等价于取几何均值。六态共享部分物理天线，得分不具有六次独立观测的统计意义。另算了四态/六态、算术平均/乘积、粗筛开关、是否使用保留直达几何的 R1 经验相位。主分支不设粗筛。粗筛对照只用四个 G 谱，在 10° 网格找相距至少 20° 的前两个局部峰，各保留 ±15°，没有输入人体坐标。朋友的 SAGE 代码、粗域构造及替代 AntSEL 实现未获提供，这些分支用于独立移植。",
      "",
      "主时间窗取 10 秒正式窗口内、切换设备对齐后的 READY +3 至 +21 ms。另以串口 host READY 重算。两种时钟、归一化和候选遮罩的具体实施都会影响峰值。ACDE 原始帧数、phy、ifindex 可与报告对应，估计帧数略有差异。PDF 未列原始 CSI 哈希，无法逐文件与对方核对。每个点的 R1 独立建模，R3 只作主结果的负对照，反向用 R3 为背景单列为敏感性检查。先冻结全部角谱与规则，再附加真值。原有 B、C 较早的批次只作重复观察。所有资料已被看过，这次属于回顾性对照。",
      "",
      "## 五点结果及独立空场审计",
      "",
      "| 点 | 中心理论方向 | 同窗原始四 G 平均 | 六态差分乘积 | 六态绝对偏差 | 同一 R1 下 R3 主峰 | 有人/后空变化强度 | 固定诊断 |",
      "|---|---:|---:|---:|---:|---:|---:|---|",*detail,
      "",
      "强度指标另算，不使用归一化 MUSIC 谱高。对每个频率/状态取 max(0,aᴴΔRa/‖a‖²)，等权平均频率和六状态，再在全角域取最大值。R2 与 R3 同尺度比较。这个指标与朋友报告的 SAGE 未归一化强度不同。五批均大于 1，故可报告在场变化超过这一次空场的峰值强度，仍不能把方向自动归因于人体。",
      "",
      "| 点 | R2 低/高半频段主峰 | 四个 R2 时间块主峰（°） | 改用 R3 背景的 R2 峰 | 按串口 host 时钟的主峰 |",
      "|---|---:|---|---:|---:|",*audit_rows,
      "",
      "支持条件在试算前写入 protocol.json。要求 R2/R3 强度比 >1，两半频段及至少三个 2.5 秒时间块的主峰距全频峰 ≤9°，换 R3 为背景后仍在 ±9°，避开 |θ|≥87° 边界，并否决 R3 内同样经分频与分块支持的近邻峰。这里只比较各曲线的全域主峰，空场检查不能消除所有非主峰的背景干扰。D 是五点中唯一通过全部检查的方向，R2 +55°，R3 也在 +56°。不过 R3 的低/高半频段为 −78°、+18°，仅 1/4 时间块接近 +56°。D 的结果是可独立复验的候选，单个完整进场还无法建立可推广的 ≤8° 人体 AoA 误差。",
      "",
      "![五点角谱](五点人体与后空场角谱.png)",
      "",
      "## 哪个改变产生了作用",
      "",
      "| 同一输入上的分支 | 五点平均绝对谱峰偏差 | 中位 | 最大 | ±8° 内的点 |",
      "|---|---:|---:|---:|---:|",*armrows,
      "",
      f"此前 3° 网格、原保护窗的四 G 原始融合为 {arch.four_G_raw_abs_diff_deg.mean():.2f}°。本次为 1° 网格、3–21 ms 窗口的同窗对照。两项同时改变，不能把差值归因于网格一项。固定六态乘积时，仅减 R1 把 41.85° 降至 17.87°。固定差分及六态时，换乘积反而比算术平均多 0.15°。加经验相位后恶化到 59.55°，自定粗筛达 30.35°。B 的 −77° 被自定粗筛改为 +30°。R1 广义白化六态为 18.05°，会改变具体站位，如 E 变为 +38°、C 变为 −22°。不能逐点选择更好的分支后再计算统一精度。",
      "",
      "## 单独看三阵元 G3",
      "",
      "用同一批 R1 作正则化背景白化，并同时变换 MUSIC 导向矢量。下面五点始终采用同一条 G3 规则，偏差均按 G3 中心计算。该分支没有参与六态主分支的拒判统计。",
      "",
      "| 点 | G3 理论方向 | R2 白化 MUSIC 主峰 | 绝对偏差 | R2 低/高半频主峰 | R3 空场主峰 |",
      "|---|---:|---:|---:|---|---:|",*g3rows,
      "",
      "A 从先前逐频参考的 −6° 变为 −13°，单次谱峰偏差从 5.08° 变为 1.92°，但两半频互相矛盾。D 的 +51° 距 G3 理论 +52.72° 为 1.72°。B、C、E 的偏差仍很大。五点 G3 白化平均绝对谱峰偏差为 26.30°，同窗原始 G3 为 45.91°。这些是待复验的谱峰，并非已确认的人体精度。",
      "",
      "较早的同点重复中，B 的同一六态主分支得到 +38°（新 B −77°），C 为 −90° 边界（新 C +53°）。这两组差异限制对角度可复现性的判断。现阶段可保留 R1 协方差去背景，以及独立空场、两半频段和整轮拒判机制。六态算术平均可作为简单基线，乘积可作对照。下一次应固定全部规则，在同一次初始化中用已知非零方向 TX 检验链序/符号，并用独立的空→人→空进场复验，同时保留 B。六态组合属于非相干谱融合，完整十二阵元的相干性能仍需跨状态标定验证。[WiCAL 原论文](https://arxiv.org/abs/2505.21408)",
      "",
      "## 可复查材料",
      "",
      "原报告定位到第 1 页主结果、第 2 页处理分支与帧数、第 3 页粗域图、第 4 页强度图。所用 PDF 的 SHA-256 为 "+digest(SOURCE)+"。本目录先运行 predict.py，再运行 evaluate.py。前者沿用工作目录内的原始 CSI 解码器。数据来源带 SHA，所有失败轮次及对照均保留。",
      "",
      "[逐点结果与拒判原因](site_results.csv) · [固定分支对照](ablations.csv) · [朋友数据和换符号对照](peer_comparison.csv) · [逐轮原始数据审计](packet_audit.csv) · [特征值检查](eigen_diagnostics.csv) · [盲选结果哈希](blind_registry.json)。",
    ]
    report=ROOT/"朋友报告对MUSIC的移植验证.md"
    report.write_text("\n".join(out)+"\n",encoding="utf8")
    links=re.findall(r'\]\(([^)]*)\)',report.read_text(encoding="utf8"))
    assert all(name.startswith("https://") or (ROOT/name).is_file() for name in links)
    print(latest[["point","primary_deg","primary_error","r3_empty_peak_deg","passes_gate","failure"]].to_string(index=False))
    print("Coverage",int(latest.passes_gate.sum()),"/",len(latest),"; all source hashes and report links validated.")


if __name__=="__main__":main()

