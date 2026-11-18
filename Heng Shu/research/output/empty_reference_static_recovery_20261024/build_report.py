"""Build source-linked Chinese interpretation from frozen diagnostic tables."""
import pandas as pd,numpy as np
from common import *
def table(df):return df.to_markdown(index=False,floatfmt='.4f')
def main():
    d=load(R/'results/decision.json');p=pd.read_csv(R/'results/empty_pair_comparisons.csv');a=pd.read_csv(R/'results/state_audit.csv');s=pd.read_csv(R/'results/source_audit.csv');h=pd.read_csv(R/'results/human_stability_summary.csv');pts=pd.read_csv(R/'results/point_status.csv');rr=pd.read_csv(R/'results/reference_checks.csv')
    full=p[p.variant=='full'];stat=full.groupby('comparison')[['spatial_distance','amplitude_distance','covariance_distance','pair_phase_median_deg']].median().reset_index()
    sam=p[p.comparison=='E1_E3'].groupby('variant')[['spatial_distance','covariance_distance']].median().reset_index()
    ref=rr.groupby(['role','variant']).agg(records=('state','size'),passes=('actual_model_applied','sum'),median_future_correlation=('predicted_corr_C','median'),supported_templates=('reference_valid','sum')).reset_index()
    ev=pd.read_csv(R/'results/events.csv');quality=[]
    for idx,z in ev.groupby('index'):
        host=np.diff(z.host_relative_s);dev=np.diff(z.device_relative_s)
        quality.append(dict(index=int(idx),host_ready_interval_median_ms=float(np.median(host)*1000),device_ready_interval_median_ms=float(np.median(dev)*1000),host_minus_device_interval_p95_ms=float(np.quantile(abs(host-dev),.95)*1000),host_nonpositive_event_intervals=int((host<=0).sum()),device_nonpositive_event_intervals=int((dev<=0).sum())))
    pd.DataFrame(quality).to_csv(R/'results/event_quality.csv',index=False)
    relevant=s[s['index'].isin([78,79,80,84,85,86])]
    text=f'''# 连续空场参考与四点静态人体信号可恢复性

2026-10-24。本报告完成的是现有材料的有界可恢复性检验。没有重新采集实验、修改原始数据、更新论文或上传 Overleaf。

## 结论

**当前尚不能利用这组空场建立可信的四点人体背景参考，也不能据此报告四点静态定位成功。四点均处于 A：未能建立可信背景参考。** 这不是“所有 CSI 不可用”的结论：有人数据仍支持有门槛的轮内相对稳定化，并保留可重复的部分谱结构；缺少的是可靠的跨记录参考及人体路径归属。

E1/E3 各自内部比较相对稳定，但它们之间的同频空间协方差、归一化幅度和链间相位存在明显差异。这些量不受各链共享的公共相位或公共定时斜率影响，因而不能仅靠调整首包零点或逐包 CPE/定时修正将差异解释掉。此次不能进一步唯一判定变化属于固定链路响应、实际信道，还是尚未核实的底层采集因素。

## 1. 实际采集关系与来源

- E1=`20261003_130149_student01_6613`，3×10 s，索引 78–80。
- E3=`20261018_151129_student01_118072`，3×15 s，索引 84–86。
- 用户确认二者同场景、连续、同硬件会话，未重启、重接线或移动 TX/RX；文件标题日期错误。
- 用户另确认：**上述空场与原四点有人数据不是同一硬件会话**。不将其作为已标定复数背景。
- E2（索引81–83）及E4（87–89）只核查来源，不擅自加入连续会话。A点有人重复批次不纳入空场。

[empty_manifest.csv](empty_manifest.csv) 保留12个空场文件的原标签、完整路径、轮次和SHA256；[primary_manifest.csv](primary_manifest.csv) 保留39个有人文件。原字段中的 `empty Oct03/Oct18` 仅为历史标签，以 `capture_id` 和 [confirmed_facts.json](provenance/confirmed_facts.json) 为准。原冻结配置中的会话未知字段由该事实文件明确覆盖，数值参数不变。

12个空场原始哈希及39个有人原始哈希均核对；有人结果仅复用。六个连续空场共有 {int(relevant.frames.sum())} 个文件帧；既有保护区及状态选择后共 {int(a.packets.sum())} 个包、36个状态记录，没有缺失整个状态。不同状态包数不均衡，逐项保存在 [state_audit.csv](results/state_audit.csv)。低幅度比例按每包每链中位幅度的0.1倍定义，范围 {a.deep_fade_fraction.min():.4%}–{a.deep_fade_fraction.max():.4%}，不等于已知SNR或丢包率。

每轮独立抽取64帧，用逐位标量解码检查向量解码的幅度、相位及目标端链重排；逐轮包选择与旧缓存一致。包头确认HT40、30频率组、3接收链，状态日志映射符合G1–G4/C1/C2定义。全部标题日期不参与状态归属；对包与事件起点共同平移时间原点，整数相对时间误差为0。该测试只说明共同平移不改变相对时间，不证明绝对时钟正确。

软件日志中 E1 的 ifindex/phy 为9/phy2，E3为81/phy26，而boot_id相同；每轮前后identity_unchanged为真。它们是需要核查来源的软件身份字段，**不据此推翻用户确认或宣称发生重启**。设备事件计数和host时间不能直接用于恢复未知实际采集间隔。轮内事件间隔见 [event_quality.csv](results/event_quality.csv)。日志一致性并非对物理开关瞬间及实际天线接线的独立测量。

解析器审计还发现当前上游PicoScenes源码首链mask与原Intel工具定义不同；本次保留原Intel两位字段与目标端重排语义，不以哪个排列让空场更相似来决定映射。采集端实际软件版本及HT40分段相位去旋转仍未独立确认。详见[协议和源码链接](PROTOCOL.md)。

![实际关系与轮内事件](figures/01_acquisition_and_timeline.png)

## 2. 方法、划分与受控自由度

沿用 Ratnam 条件ML逐包相位/定时校正的三链接收适配、分状态二维MUSIC。不是原论文Algorithm 5的逐字复现，也不是CFO频率Hz估计。原理论与本研究差异详见[既有方法快照](provenance/previous_METHOD.md)，文献DOI：[10.1109/TWC.2024.3376332](https://doi.org/10.1109/TWC.2024.3376332)。

令包归一化观测为 x[p,m,k]，链独立参考为 r[m,k]。估计一个公共时延斜率t和公共相位psi，最大化

`|sum_mk conj(x[p,m,k]) * r[m,k] * exp(-j*2*pi*f[k]*t)|`，

再对同包三链同时乘 `exp(j*2*pi*f[k]*t+j*psi)`。未独立归零每根天线，未拟合固定硬件响应。

同频共轭量 `x_m(k) conj(x_n(k))` 消除这两个公共相位因子。将其逐包平均得到同频空间协方差；跨频率堆叠协方差则仍受定时斜率影响。距离统一定义为

`d(U,V)=||U-V||_F / ((||U||_F+||V||_F)/2)`。

幅度距离采用包归一化后的平均幅度；链间相位使用圆差，只统计双方相干度>=0.5的频点。该支持筛选仅用于描述，不参与校正或调峰。

开发仅E1/run1–2；E1/run3和E3三轮为留出。每状态最多三种初始化：首包、前64包相位不变量代表包、经公共校正且相关通过后取平均的模板。后者是阈值筛选均值，不是已证明具有统计最优性的稳健估计器。开发数据按通过数、有效模板数、未来留出特征相关依次排名，选择 `invariant_medoid`；所有门槛均未放宽。

模型条件为链间独立拟合时延差p90<=5ns、相位差p90<=15°、预测相关中位数>=0.90；这些是工程一致性门槛，不是物理精度置信区间。模板用前64包，有启动等待；随后参考更新仅使用已接受的历史修正包。公共参数预测用链1/2的部分频点拟合，在链3及未参与拟合频点检验；时间前缀之外每8包验证一次。留出记录以前已有诊断，故属于探索性留出而非新采集盲测。

A为基线；B仅公共相位；C仅在模型及包条件均通过时应用公共相位和定时校正。失败包及状态原样保留。当前全部空场C状态拒绝，所以实际C=A；候选拟合参数仍保存但不能冒充已应用校正。

静态多径合成检查中三种参考均保持同频空间相位到数值精度，链特异变化被拒绝，详见[合成结果](results/synthetic_preservation.json)。既有弱动态无硬件误差负对照仍出现1.75%–2.31%信道变化，说明算法可能吸收真实动态；本次没有将该限制取消。

## 3. 连续空场的实际差异

下表为所有对应状态轮对的中位数（within_E1和within_E3各18项，E1_E3为54项）。多项比较共享记录，不能作为18或54个独立实验。

{table(stat)}

E1–E3同频空间距离中位数为 {d['pair_medians']['spatial_distance']:.4f}，支持频点的链间圆相位差中位数为 {d['pair_medians']['pair_phase_median_deg']:.2f}°。这排除了“仅一个公共相位/斜率变化即可解释全部差异”的模型，但不能仅凭此区分环境和链特异硬件变化。归一化幅度也变化，因此额外逐链相位旋转不足以使观测成为相同背景；本次没有尝试依据观测相似度拟合这类旋转。

开发轮次按四个连续时间块建立参考变化尺度：每状态各指标门槛为 `max(0.05,1.5×开发块对最大距离)`，具体值见[development_thresholds.csv](results/development_thresholds.csv)。包数不足16的块对不用于阈值（G2只有7个开发块对，其余各12个）；这是统计支撑限制，不是按结果删数据。所有54个E1–E3状态轮对均未通过空间、幅度和堆叠协方差重复性门槛。

![空场不变量](figures/02_empty_invariant_comparison.png)
![链间相位](figures/03_chain_phase_by_frequency.png)

相位图每个状态显示接收链1–2的三轮曲线，正负180°处跳变是包裹相位显示，不应解读为突然物理相位跳变。其他链对纳入完整CSV指标。该图说明每次内部曲线重叠较多、两次间偏移较大；不能证明某条曲线就是直达径或人体路径。

时长、包数和切换边界敏感性检查结果：

{table(sam)}

统一前10秒及最多512个等间隔选取包，或额外排除边界1/2ms，都未消除空间差异。结果不支持这些因素是当前差异的主要解释，但并未独立验证固件切换延迟和真实物理端口映射。

## 4. A/B/C及参考初始化

{table(ref)}

选定代表包方案的36个状态记录中0个通过；留出24个中，时延条件失败{d['holdout_delay_failures']}个，相位条件失败{d['holdout_phase_failures']}个，前缀预测相关失败{d['holdout_prefix_corr_failures']}个（失败可重叠）。未来留出特征相关中位数为{d['holdout_future_corr_median']:.4f}。开发多包均值方案的12个模板均未满足90%支持率；不把低支持模板包装成成功的稳健参考。

代表包可减小一些首包造成的极端相位差，但并非所有状态改善，也未恢复完整公共模型。因此不能通过事后放宽门槛使全部空场通过。

跨记录使用E1两个开发参考分别预测4个留出记录，共48个状态比较，前缀模型通过0个。未来独立特征候选相关中位数为{d['cross_record_future_corr_median']:.4f}。独立链拟合仅作诊断，未应用。该数量包含E1内部预测与E1–E3预测，逐项记录见[cross_record_prediction.csv](results/cross_record_prediction.csv)。

B改变协方差最大仅{d['invariant_B_max']:.2e}，符合代数不变量；实际C与A协方差差为0，因为C被拒绝。不能将候选残差改善写成应用后的物理校准。

![模型失败原因](figures/04_common_model_failures.png)
![ABC与跨记录预测](figures/05_ABC_and_cross_prediction.png)
![窗口敏感性](figures/07_window_sensitivity.png)

## 5. 无人条件下也出现显著谱差

在背景参考条件未通过时，没有执行白化或复数CSI相减。改为明确标记的空场零假设诊断：对每个状态的归一化MUSIC谱计算 `max(P_test-P_reference,0)`，以E1开发时间块间的最大绝对谱差为阈值，检验留出空场。该阈值只取开发记录，未用人体坐标。

E1–E3的54个状态轮对全部超过阈值；同次E1为1/18，同次E3为2/18。谱归一化和q=3沿用基线。这是“同样无人也能产生明显新增谱结构”的证据，**不是白化检测器的100%虚警率，也不是54次独立假阳性实验**。局部极大值坐标仅是未校准谱参数。

因此，本次不把谱差最大的峰命名为人体，不进入空场白化、背景投影及四点新恢复计算。目标中这些模块是条件分支；停止依据来自实际留出失败，而不是因为会话不同就直接宣布所有方法无效。

![空场谱变化](figures/06_empty_null_changes.png)

## 6. 四点现有结果及逐轮图册

复用全部39轮、234状态的既有A/C谱，核对原始哈希和缓存哈希。每轮一页，每页六状态A/C对照，共39页；未重跑四点原始CSI，未删除低质量状态。索引和来源见[human_reuse_provenance.csv](results/human_reuse_provenance.csv)。

{table(pts[['point','x','y','runs','states','old_common_model_passes','status']])}

- [+1m，12轮图册](figures/human_atlas/oldplus1.pdf)
- [-1m，9轮图册](figures/human_atlas/oldminus1.pdf)
- [+2m，9轮图册](figures/human_atlas/oldplus2.pdf)
- [-2m，9轮图册](figures/human_atlas/oldminus2.pdf)

233/234的旧轮内公共模型通过不等于跨会话背景可用。全部四点当前状态均为A，尚未识别经验证的人体候选路径。图册展示“待解释的谱”，不谎称已经获得人体候选。

下表为同一采集批次不同完整轮次的稳定性；跨批次结果也保留在CSV。这里的极大值变化只是诊断，不能当作人体角度/距离误差。

{table(h[h.same_batch][['point','method','comparisons','median_cov_distance','median_spectrum_rms','median_max_angle_difference_deg','median_max_delay_difference_ns']])}

此次汇总发现旧C在同批次跨轮比较中，四点的谱RMS中位数均高于A；其独立轮次参考可以造成跨轮时延零点不一致。这与旧报告的“轮内时间块改善”不是同一指标，不构成数值矛盾，但说明不能把轮内收益推广成跨轮或定位收益。不同批次仍有较大谱差，不用坐标来选峰或拟合偏置。

![有人既有谱稳定性](figures/08_human_existing_stability.png)

## 7. 几何、可辨识性与下一步

继续采用全局TX=(0,0)、RX=(3.65,0)，RX局部位置=(3.65-X,-Y)；12阵元28mm，r_m=(0,(6.5-m)×0.028)。G1=[1,2,3]、G2=[4,5,6]、G3=[7,8,9]、G4=[10,11,12]、C1=[1,8,6]、C2=[10,8,6]。本次分状态三链结果未相干拼接成12阵元。

无法从这些记录唯一分离的量包括：每状态每链固定响应与静态传播相位、每记录任意时延规范、真正直达径与其他背景径，以及人体反射与遮挡造成的多径变化。缺少共同会话背景进一步限制了有人/无人差异的归因。

只有独立角度/固定响应标定、可复现人体候选和同记录直达径均成立后，才使用 `Delta_tau=tau_candidate-tau_direct` 及 `||p-TX||+||p-RX||=3.65+c*Delta_tau`。当前不输出坐标误差、不将参考残差乘光速当定位误差，也不通过切换球面波模型选取更接近真值的峰。平面/球面位置搜索在人体路径与参考条件满足后才进入；本次不重复已有固定距离几何敏感性图。

最小补充顺序：

1. 在一个持续会话、固定接线和位置下，采集空场—一个固定人体站位—空场，重复3个完整循环，各段至少15秒并记录进入/离开时刻。人体点事先固定，真值只用于冻结后评价，先不扩大至四点。
2. 保存连续状态事件、原始包时间、驱动/固件版本、实际端口接线照片和会话标识；同时核查HT40分段相位处理，不猜测并强制去旋转。
3. 若无人前后仍存在链特异参考差异，先做已知源的链路/开关标定。可选等长分路、已知延迟衰减的有线参考，或具有直达径隔离证据的已知位置OTA源；补偿真实阵元传播相位，不能把所有原始天线相位置零。
4. 首个站位的有人变化超过两侧空场波动、跨重复和子阵相容后，再扩展四点；角度固定响应与同记录直达径成立后，才评价物理定位。

这组补采只能验证新会话的可恢复性，不能自动追溯标定历史四点。需要历史硬件状态桥接证据才能提出旧数据物理恢复的主张。

## 8. 更正、文件与复现

旧文件全部保留。旧图07比较的是E1–E2及E3–E4，不是E1–E3；其原数值不改名移用。撤回从目录日期推导跨日期/跨初始化原因的解释。新报告补充同会话E1–E3实际比较，并明确有人数据属于不同会话。原始A的差异早于C的首包处理，不能由C初始化反向解释。

计算仅包含6轮空场、最多3种初始化、冻结后4轮留出、48项跨记录检查及小范围保护区敏感性；四点39轮是已有结果复用。未运行Widar，未进行真值挑峰，未执行新人体恢复大规模实验。

[README](README.md)列出顺序命令；[冻结配置](config.json)、[开发选择](results/development_selection.json)、[决策记录](results/decision.json)、[参数目录](parameters/)及完整CSV均保留。背景白化和定位是明确未执行的失败条件分支，不将其列为算法成功。
'''
    (R/'REPORT.md').write_text(text,encoding='utf-8')
    print('Report generated',flush=True)
if __name__=='__main__':main()
