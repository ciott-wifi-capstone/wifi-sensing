> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# 实测二维 MUSIC 诊断谱，非已验证的人体 AoA–ToF

只采用旧四点中 acquisition PASS、完整窗口、逐轮相位波动小于 10 度的轮次；新 A–D/空场及旧失败批次均未进入。使用 G1 的三接收链，避免未经 CPA 的跨状态拼接。选择依据与 ground truth 数值无关。

原始 CSI 保留频率线性相位，不进行会改变延时的逐包去斜率或真值拟合。逐包单位范数归一化；2 链 × 15 子载波组的重叠子阵平滑协方差；HT40 的 30 个子载波索引为 -58:4:58，间隔 312.5 kHz。主图固定信号子空间维数 3，附表给出 1/2/3/4 的敏感性，不将所设维数当作已知路径数。每轮谱先归一化，再取组内逐像素中位数并重归一化。色条不可用于比较绝对接收功率。

纵轴只是 exp(-j2πmu) 中的空间相位频率，m 为解码后链索引；并非实际角度。只有确认等间距线阵、方向及相位校准后，才能使用 u=d sin(theta)/lambda（相应符号约定）转换。三链模型符合程度也未验证。

横轴为未校准时延参数，含硬件和包检测等未知偏置，800 ns 的采样周期内扫描 -400 至 400 ns；不是传播绝对 ToF，也不是已对齐直达路径的超额时延。2 ns 网格不代表 2 ns 分辨率。负值不代表负传播时间。

未做空场复数相减；峰可能来自直达波、墙体、多径或硬件残差，不能标为人体，更不能用来报告米级定位误差。图只能用于检查谱结构与重复性，不能作为“已完成静态人体定位”的证据。

selected_runs.csv 记录筛选和来源，model_order_sensitivity.csv 记录逐轮各维数的最强峰，spectra.npz 保存完整主图输入。合成单路径测试仅验证实现符号/轴与已知模型一致，不验证现场物理标定。

复现：python -X utf8 music_diagnostic.py。
格式和子载波索引来源：https://github.com/wifisensing/RXS-Parsing-Core/blob/master/CSISegment.cxx
