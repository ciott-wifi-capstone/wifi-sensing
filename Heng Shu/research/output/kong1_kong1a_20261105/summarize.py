"""Join surveyed geometry after the blind peak tables have been saved and hashed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_captures as a

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/"analysis_rules.json").read_text(encoding="utf8"))
POINT = np.array([.80,.60])
RX = np.array(a.GEOMETRY["global_rx"],float)
TX = np.array(a.GEOMETRY["global_tx"],float)
COLORS = ["#235aae","#c06421","#19766f"]
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})


def table(name):
    return pd.read_csv(HERE/f"{name}.csv")


def angles():
    x = RX[0]-POINT[0]
    y = -POINT[1]
    result = {"G1-G4 noncoherent":np.degrees(np.arctan2(y,x))}
    for state in a.STATES:
        centre = np.mean(a.Y[np.array(a.MAP[state])-1])
        result[state] = np.degrees(np.arctan2(y-centre,x))
    return result


def comparison_figure(data):
    fig, axes = plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    pairs = [("Kong1","空场批内（3 对）"),("Kong1A","A 点批内（3 对）"),("between","跨批（9 对）")]
    for ax,metric,title in zip(axes,["amplitude_shape_distance","covariance_distance"],
                               ["归一化幅度形状","同频空间协方差"]):
        for i,(group,label) in enumerate(pairs):
            grouped = data[data.comparison==group].groupby("state")[metric]
            med = grouped.median().reindex(a.STATES)
            lo = grouped.min().reindex(a.STATES)
            hi = grouped.max().reindex(a.STATES)
            ax.errorbar(np.arange(6)+.18*(i-1),med,yerr=[med-lo,hi-med],fmt="o",
                        ms=4,capsize=3,color=COLORS[i],label=label)
        ax.set_xticks(np.arange(6),a.STATES)
        ax.set_title(title)
        ax.set_ylabel("相对距离（无量纲）")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y",alpha=.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False,fontsize=9)
    fig.suptitle("两批信道的差异｜每个点为轮对中位数，线段为最小至最大值",fontsize=14)
    fig.savefig(HERE/"空场与A_空间变化.png",dpi=175)
    plt.close(fig)


def angular_figure(theory):
    with np.load(HERE/"angular_spectra.npz") as curves:
        fig, axes = plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True,layout="constrained")
        for ax,state in zip(axes.flat,a.STATES):
            ax.plot(a.ANGLE,10*np.log10(curves[f"Kong1_3_{state}_raw"][0]),
                    color="#888888",ls="--",lw=1.45,label="空场 R3")
            for run,color in enumerate(COLORS,1):
                ax.plot(a.ANGLE,10*np.log10(curves[f"Kong1A_{run}_{state}_raw"][0]),
                        color=color,alpha=.9,lw=1.65,label=f"A 点 R{run}")
            ax.axvline(theory[state],color="#4a284b",ls=":",lw=2,label="A 几何方向")
            ax.set_title(f"{state}　天线 {','.join(map(str,a.MAP[state]))}")
            ax.set(xlim=(-90,90),ylim=(-7,.4),xticks=[-90,-45,0,45,90])
            ax.grid(alpha=.17)
        for ax in axes[-1]:
            ax.set_xlabel("谱角度参数 / °")
        for ax in axes[:,0]:
            ax.set_ylabel("相对峰值 / dB")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=5,frameon=False)
        fig.suptitle("实测分状态空间 MUSIC｜各曲线独立归一化\n空场与 A 点属于不同接收机初始化，竖线仅作事后几何对照",fontsize=13)
        fig.savefig(HERE/"A点_分状态角度谱.png",dpi=175)
        plt.close(fig)


def joint_figure():
    with np.load(HERE/"joint_spectra.npz") as curves:
        fig, axes = plt.subplots(2,3,figsize=(11.7,6.9),sharex=True,sharey=True,layout="constrained")
        for ax,state in zip(axes.flat,a.STATES):
            raw = curves[f"Kong1A_3_{state}"]
            im = ax.pcolormesh(curves["tau_ns"],curves["angles"],10*np.log10(np.maximum(raw,1e-5)),
                               shading="auto",vmin=-6,vmax=0,cmap="viridis")
            ax.set_title(state)
        for ax in axes[-1]:
            ax.set_xlabel("未校准时延参数 / ns")
        for ax in axes[:,0]:
            ax.set_ylabel("未校准角度参数 / °")
        fig.colorbar(im,ax=axes.ravel().tolist(),label="本子阵相对谱峰 / dB",shrink=.88)
        fig.suptitle("A 点第三轮｜六个三阵元子阵的二维 MUSIC 诊断谱",fontsize=14)
        fig.savefig(HERE/"A点_R3_AoA_ToF.png",dpi=175)
        plt.close(fig)


def format_row(values):
    return "| "+" | ".join(str(v) for v in values)+" |"


def build_report():
    hashes = a.read(HERE/"blind_output_hashes.json")
    assert all(a.sha(HERE/name)==v for name,v in hashes.items())
    truth = angles()
    dr = np.linalg.norm(POINT-RX)
    dt = np.linalg.norm(POINT-TX)
    excess_ns = (dr+dt-np.linalg.norm(TX-RX))/a.C*1e9
    geo = dict(TX=TX.tolist(),RX=RX.tolist(),A=POINT.tolist(),TX_A_m=dt,RX_A_m=dr,
               theoretical_excess_tof_ns=excess_ns,angles_deg=truth,physical_error=None)
    (HERE/"survey_geometry.json").write_text(json.dumps(geo,indent=2),encoding="utf8")
    runs, qc, comp = table("runs"), table("state_quality"), table("record_comparisons")
    comparisons=[]
    for s in a.STATES:
        ss=comp[comp.state==s]
        for metric in ["covariance_distance","amplitude_shape_distance","phase_median_deg"]:
            rec=dict(state=s,metric=metric)
            for group in ["Kong1","Kong1A","between"]:
                subset=ss[ss.comparison==group][metric]
                for suffix, val in [("min",subset.min()),("max",subset.max()),("median",subset.median())]:
                    rec[f"{group}_{suffix}"]=val
            rec["all_cross_exceed_all_within"] = rec["between_min"]>max(rec["Kong1_max"],rec["Kong1A_max"])
            comparisons.append(rec)
    summary = a.save("comparisons_by_state.csv",comparisons)
    local = table("blind_local_peaks")
    occupied = local[local.batch=="Kong1A"].copy()
    occupied["theory_deg"] = occupied.state.map(truth)
    occupied["retrospective_absolute_difference_deg"] = (occupied.peak_deg-occupied.theory_deg).abs()
    occupied["human_path_verified"] = False
    occupied.to_csv(HERE/"A_peak_geometry_comparison.csv",index=False,encoding="utf-8-sig")
    peaks = table("blind_spatial_maxima")
    occ = peaks[peaks.batch=="Kong1A"]
    peak_rows=[]
    for s in a.STATES+["G1-G4 noncoherent"]:
        seq=occ[occ.state==s].sort_values("run").peak_deg.to_list()
        peak_rows.append(format_row([s,f"{truth[s]:+.2f}",*(f"{x:+.0f}" for x in seq)]))
    glob = comp.groupby("comparison")[["covariance_distance","amplitude_shape_distance","phase_median_deg"]].median()
    comp_rows = [format_row([label,*(f"{x:.3f}" for x in glob.loc[group].values[:2]),
                             f"{glob.loc[group,'phase_median_deg']:.1f}°"])
                 for group,label in [("Kong1","空场内部"),("Kong1A","A 点内部"),("between","空场对 A 点")]]
    comparison_figure(comp)
    angular_figure(truth)
    joint_figure()
    checked = table("input_integrity")
    signs = table("sensitivity_sign_holdout").query('model=="fitted_sign_only" and both_states_unseen')
    empty_cal = table("zero_degree_holdout").query('frequency_set=="all"')
    empirical = table("sensitivity_empty_per_tone").query('empty_target_run==3')
    shared = table("shared_6_8")
    near = occupied[occupied.retrospective_absolute_difference_deg<=9]
    c1 = occupied[(occupied.state=="C1")&(occupied.peak_deg==-21)].sort_values("run")
    empty_c1 = local.query('batch=="Kong1" and mode=="raw" and state=="C1" and peak_deg==-12')
    assert len(c1)==3 and c1.repeat_within_window.all() and len(empty_c1)==3 and empty_c1.repeat_within_window.all()
    gap_s=(runs[runs.batch=="Kong1A"].start_ns.min()-runs[runs.batch=="Kong1"].end_ns.max())/1e9
    raw_threshold_pass = (qc.coverage>=.9) & (qc.guarded_packets>=330) & (qc.first_half_packets>0) & (qc.second_half_packets>0)
    assert len(qc)==36 and len(comp)==90 and checked.ok.all() and raw_threshold_pass.all()
    report=[
        "# Kong1 空场与 Kong1A 静态人体 A 点实测分析",
        "",
        "两批的六个原始 CSI 文件均完成逐包解码。两批的信道特征有可重复的差异，但占用状态和接收机初始化同时改变。A 点的分状态谱中可保留稳定的局部峰，目前人体路径、物理时延、定位精度及 12 根相对 3 根的增益都待独立验证。",
        "",
        "## 数据与接收机",
        "",
        f"用户指定 Kong1 为无人、Kong1A 为随后测得的静止人体 A。两个 config.json 的 label 都是 kong_boot01，场景归属按用户记录。按同一台主机记录的窗口时钟，空场 R3 结束到 A 的 R1 开始间隔 {gap_s:.2f} 秒。文件日期不作实际采集日期证据。",
        "",
        "两批 boot_id 相同。各自的 operations.log 第 5、7、9、11 行均成功执行卸载和加载驱动，第 13 行重新准备网卡，第 71 行删除 monitor。空场是 phy3 / ifindex 12，A 是 phy4 / ifindex 15。两份内核日志均记录固件加载。六个 run.json 的 continuity_gate 均为 true，它检验各自轮内的连续性。详见 [接收机日志摘录](receiver_events.csv)。",
        "",
        f"两份原始清单共 {len(checked)} 项通过文件大小及 SHA256 核验，6 个 CSI 哈希各异，均未命中先前四点分析的清单。每个原始帧均通过现有 Intel 5300 / PicoScenes HT40 专用解码器的格式检查，并按每包天线选择位重排。帧头确认控制 5240 MHz、中心 5230 MHz、40 MHz、1 发 3 收、30 个频率组。两批记录的 802.11 发射地址一致。",
        "",
        "| 批次 | 10 秒窗口内的包数 R1 / R2 / R3 | 边界保护后六状态总包数 | 每轮每状态包数范围 |",
        "|---|---|---:|---:|"]
    for b in a.CAPTURES:
        counts=runs[runs.batch==b].sort_values("run").formal_frames
        guarded=qc[qc.batch==b].guarded_packets
        report.append(format_row([b," / ".join(str(v) for v in counts),int(guarded.sum()),f"{guarded.min()}–{guarded.max()}"]))
    report += [
        "",
        f"全部完整驻留窗口都有有效包，六状态映射和 READY 序号检查通过。只使用采集开始/结束之间的目标包，去除切换两侧约 {runs.guard_ms.min():.2f}–{runs.guard_ms.max():.2f} ms。保护间隔依赖既有时钟漂移假设。增至 5 ms 后，36 个轮/状态的协方差相对变化最大 {table('guard_sensitivity').covariance_distance.max():.3f}。[每轮清单](runs.csv)｜[状态包数](state_quality.csv)｜[保护间隔敏感性](guard_sensitivity.csv)",
        "",
        "## 同一位置是否重复，两批差异多大",
        "",
        "将每包数据用其整体幅度范数归一化，在每个频率组分别计算三链空间协方差。取协方差对角线的平方根衡量幅度形状，取相干度至少 0.5 的链对比较相位。矩阵/幅度距离为 2‖a−b‖/(‖a‖+‖b‖)。它们消去同包公共增益、公共相位与公共定时斜率，协方差仍依赖链间响应。下表是各状态、各轮对的中位数；相位列先在每个轮对/状态内部取中位数。",
        "",
        "| 比较 | 协方差距离 | 幅度形状距离 | 链间相位差 |",
        "|---|---:|---:|---:|",*comp_rows,
        "",
        "每状态有 3 对空场内部、3 对 A 内部、9 对跨批组合，组合复用记录，仅作描述性比较。6/6 个状态的最小跨批协方差距离均大于该状态两批内部的最大值。对完全不依赖相位符号的幅度形状，相同严格比较有 5/6 个状态通过，C1 略有重叠。因此仅改变公共包相位、或仅改变链相位符号，均不足以解释全部观测。人体反射/遮挡、场景变化和重初始化后的链路响应在本设计中无法分别归因。",
        "",
        "![空间特征差异](空场与A_空间变化.png)",
        "",
        "## 空场 0°参考与跨状态检查",
        "",
        "在 Kong1 R1 扣除 TX 到实际阵元的直达传播相位差，拟合每状态的两条相对链响应。以 20/30 个频率组拟合常数或带一个频率斜率的模型，以独立 R2 的另外 10 个频率组选择，常数模型达标时优先选择。R3 完全留出。采用先前操作指南的 80% 可靠频点及 P90≤15°标准。模型仅用于 Kong1，未对 A 施加空场复数校正。",
        "",
        f"留出 R3 上两链对合计 {empty_cal.pass_same_direction.sum()}/{len(empty_cal)} 通过。各链对 P90 为 {empty_cal.p90_phase_deg.min():.1f}–{empty_cal.p90_phase_deg.max():.1f}°。校正后 R3 的 G1–G4 最大谱峰分别在 −3°、0°、+3°、0°，可见谱峰近正前方仍允许很大的逐频点相位残差。[模型选择](zero_degree_model_development.csv)｜[留出预测](zero_degree_holdout.csv)",
        "",
        f"附加检查放宽为逐频点记录 R1 的经验相位。使用相同 R3 检验，{empirical.pass_repeatability_only.sum()}/{len(empirical)} 条链对满足同方向重复性门槛（G1 链3、G2 链2）。这是看过主结果后的描述性检查，反映本场景参考的可重复程度。[逐频点结果](sensitivity_empty_per_tone.csv)",
        "",
        f"每轮可配对约 66 个相邻 C1/C2 周期。共享天线 6 对 8 的相位差，在 C1、C2 之间的绝对残差中位数为 {shared.phase_median_deg.min():.2f}–{shared.phase_median_deg.max():.2f}°，P90 最大 {shared.phase_p90_deg.max():.2f}°。这支持共享天线对的局部一致性。G1–G4 全部组间的延时和相位、跨初始化的固定响应尚需单独验证。[逐轮共享残差](shared_6_8.csv)",
        "",
        "附加符号检验只用两批 R1 的 G1/G2 和 20 个频率组，在四种 RF 链符号组合中选得 (+1,−1,−1)。在两批 R3 的另外 10 个频率组、未参与拟合的 G3/G4/C1/C2 上，中位相位差均有下降，但各状态残差 P90 仍为 "
        f"{signs.phase_p90_deg.min():.1f}–{signs.phase_p90_deg.max():.1f}°。该假设可以描述部分差异，无法作为 A 的硬件校准；没有据此修正人体谱。[留出符号检查](sensitivity_sign_holdout.csv)",
        "",
        "## A 点方向与候选",
        "",
        f"TX=(0,0)、RX=(3.65,0)、A=(0.80,+0.60) m；按照等高躯干中心几何，RX 阵列中心理论角 {truth['G1-G4 noncoherent']:.2f}°，TX–A={dt:.3f} m，A–RX={dr:.3f} m，理论双基地超额时延 {excess_ns:.3f} ns。各子阵按自身物理中心计算理论角。",
        "",
        "表中是同频空间 MUSIC 的未校准角度参数。每个频率组取单信号子空间，角度网格 3°。从整个前半平面盲选，原始最大峰和局部峰在读取人体坐标前已保存并核验哈希。四组融合对各 G 的归一化线性谱等权相加。",
        "",
        "| 子阵 | 理论角 / ° | R1 最大峰 / ° | R2 最大峰 / ° | R3 最大峰 / ° |",
        "|---|---:|---:|---:|---:|",*peak_rows,
        "",
        "C1 的 −21° 分支三轮都在，排名为第 1、1、3；R3 距最强峰仅 0.27 dB。每轮四个时间分块和上下两个频段的局部峰都支持这个分支（匹配容差 9°）。它与 C1 理论角相差 8.31°。但 C1 每轮均有 8 个处于峰值以下 6 dB 以内的局部峰；Kong1 的 C1 也三轮都在 −12° 出现稳定局部峰，几何上更接近同一个 A 标签。C1 的 −21° 可列为待复验分支。",
        "",
        "C2 的 R1 有 −9° 次峰，距本子阵理论角 2.08°；R2 的 −6° 未通过双频段检查，R3 没有相应的近真值分支。G1 的 −6° 只在前两轮出现且频段不一致，G4 的 −18° 出现在 R3。没有独立的多个 G 在实际几何附近三轮一致地支持该方向。事后几何距离均属于单个峰的描述，不当作人体 AoA MAE。",
        "",
        "![各子阵角度谱](A点_分状态角度谱.png)",
        "",
        "## 二维谱和论文能报告的范围",
        "",
        "另复用原工程的分状态二维 MUSIC：15 个连续频率组做频率平滑，模型阶数 3，时延网格间隔 4 ns，去掉 800 ns 周期的重复端点。A 的 G2 三轮二维主峰角度分别为 −69°、−72°、−66°，时延参数均为 176 ns。1D 与 2D 的模型/谱不同。尚未在同一记录中分离可核验的直达径与人体路径，因此不把时延谱轴读数当作人体反射距离，也不计算二维位置误差。",
        "",
        "![A 第三轮二维诊断谱](A点_R3_AoA_ToF.png)",
        "",
        "可报告六状态采集完整性、批内重复程度、空场/有人两批信道差异、原始候选峰和共享天线的局部一致性。人体检出率、人体 AoA MAE、物理超额时延、二维位置误差、12 阵元相干增益均保留为 NA；当前的四 G 融合属于非相干融合。",
        "",
        "## 最小补测",
        "",
        "下一批只调用一次网卡准备和记录流程，同一批内部按 R1 无人 → R2 人在 A 站定 → R3 无人，每轮 10 秒。间隔预留约 15–20 秒给进出与站稳，实际按现场记录。每轮检查 boot_id、phy、ifindex、操作日志和六状态包数；中途无需再启动整套脚本。把两个无人段的差异作为背景波动对照。短测通过后，可在一次初始化中按 空→A→空→A→空→A→空 取得三次独立重新站位。若使论文能够报告绝对方向，还需已知源未拟合方向验证；若比较完整 12 阵元，还需独立跨组相干验证。",
        "",
        "## 复算与文件",
        "",
        "在本目录先运行 analyze_captures.py，再运行 reference_sensitivity.py 和 summarize.py。原始输入保持只读。原工程的 decoder_ht40.py 用于解码/事件对齐，geometry_update.py 用于二维谱。[规则](analysis_rules.json)｜[盲选峰](blind_local_peaks.csv)｜[A 点事后几何对照](A_peak_geometry_comparison.csv)｜[二维主峰](blind_joint_maxima.csv)｜[全部轮对](record_comparisons.csv)｜[盲选输出哈希](blind_output_hashes.json)｜[合成符号/时延检查](numerical_checks.json)。",
        "",
        "WiCAL 将组内固定响应校准与非同时观测的跨组对齐分别处理，这里只检验已有六状态记录支持的部分。[WiCAL 原文 II-B](https://arxiv.org/html/2505.21408v1#S2.SS2)"
    ]
    (HERE/"Kong1与A点_实测分析.md").write_text("\n".join(report)+"\n",encoding="utf8")
    result=dict(verified_raw_files=6,verified_manifest_items=len(checked),total_formal_frames=int(runs.formal_frames.sum()),
                total_guarded_packets=int(qc.guarded_packets.sum()),physical_receiver_initializations=2,
                relative_time_gap_s=gap_s,primary_heldout_phase_pairs_passed=int(empty_cal.pass_same_direction.sum()),
                secondary_empirical_phase_pairs_passed=int(empirical.pass_repeatability_only.sum()),
                retrospective_near_9deg_rows=len(near),physical_AoA_MAE=None,physical_excess_tof_error=None,
                position_error=None,coherent_12_antenna_result=None)
    (HERE/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(result)


if __name__=="__main__":
    build_report()
