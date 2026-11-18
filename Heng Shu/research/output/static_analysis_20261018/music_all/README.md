> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# 全部新旧数据的实测 MUSIC 图

输入为全部 90 轮去重 CSI，原 93 个路径中的 3 个 kong 副本按用户确认归入 A。包含旧失败批次、新 A–D、日志日期为 10 月 3 日和 18 日的空场。未按结果好坏筛选；所有 540 个“轮次×状态”都有 state_results.csv 记录。

G1 的角度轴采用用户确认的三阵元直线、28 mm 间距和 5.23 GHz。按用户最后纠正：从 RX 后方看向 TX，链 1→2→3 从左到右。0 度为朝向 TX 的阵列法线，面向 TX 时左侧为正角、右侧为负角；使用 a_m=exp(-j2π m d sin(theta)/lambda)。此方向定义不消除链间硬件相位偏差。假设链间固定相位偏置为零，因而是未校准角度搜索坐标，不能当作已验证的人体 AoA。

核对用户粘贴的海报对话：2082–2087 行给出 G1=1,2,3；G2=4,5,6；G3=7,8,9；G4=10,11,12；C1=1,8,6；C2=10,8,6。3831 行用户说明之前直接采用静态房间 R1 各天线相位参考；该方法不是独立硬件标定。2230 行旧指南间距是文献/指南参数，本次 G1 采用最新用户确认的 28 mm。最初粘贴对话没有明确链序左右关系；随后用户确认了从 RX 后方朝 TX 看从左到右。PPTX 实拍 image3.jpeg 确认十二根天线为直线结构，编号方向以用户最新说明为准。没有把 R1 参考自动迁移到本次数据。

G1–G4 分状态计算，未跨状态拼接阵列。G2–G4 图保留链索引空间相位轴，不假定它们与 G1 相同的物理方向或间距。它们的线性链索引导向矢量是诊断模型，不能解释为各组的真实物理 AoA。C1/C2 连接非连续阵元，因此仅提供跨频率的一维时延谱，没有强行套用 28 mm 三阵元 AoA 模型。

采用前一轮同一 Intel5300 解码器和时间守护窗。每个非空守护窗中的包全部进入对应状态；状态总计不足 30 包时不伪造谱，标记 insufficient。少于 200 包或少于 10 个窗口标记 sparse，仍计算并保留。无数据状态、丢失时段无法补成测量。保留旧截断轮次的实际观测部分。

2×15 空频子阵前向平滑、逐包单位范数归一化、固定信号子空间维数 3，30 个 HT40 索引 -58:4:58 ×312.5 kHz。G1 角度网格 1 度，未校准时延扫描 -400:2:400 ns。网格间隔不是分辨率。无空场复数相减、无测试坐标引导挑峰、无逐包线性相位去除。组图为各轮归一化谱的逐像素中位数再次归一化，色条不表示绝对功率；所有 sparse 谱亦纳入，详细样本量见逐轮表。

文件：all_groups_G1_angle.png/pdf 为 10 组 G1 局部角度–时延搜索图；all_groups_G1_G4.png/pdf 为全部组别四观测状态；all_runs_G1_G4.pdf 为 90 页逐轮图；reference_states_delay.png 为 C1/C2 时延图。三个 npz 保存全部已算谱与轴，input_runs.csv 和 state_results.csv 保存文件来源、哈希、数量和状态。数组键编号与 state_results.csv 的 index 对应。

这些是由实测 CSI 计算的 MUSIC 诊断谱，但尚不能把谱峰确定为人体路径、绝对 ToF 或定位精度。原始时延偏移没有校正，不应直接与 ground truth 的 0.9–5.9 ns 超额时延对比。用户提供的理论点位未参与谱估计。

复现：python -X utf8 music_all.py。未修改原始数据。

方向修订复现：完成 music_all.py 后运行 orient_music_all.py。新增 old_four_points、new_ABCD、empty_groups 三套 PNG/PDF；orientation.json 和 angle_convention.png 给出可核查的角度约定。角度正负号沿用标准负时延相位模型，尚需已知方向实测验证硬件符号。海报原 R1 场景相位参考未在本轮套用。
