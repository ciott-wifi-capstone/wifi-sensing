> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# 统一非相干 MUSIC：方法和复现说明

本轮方案为本研究的实验性改造，不冒称已有论文的完整算法。预测、候选检查与事后几何评分分开运行。预测代码不读取人体站位坐标；数据已经用于多轮开发，因而不称为新独立盲测。

## 数据预算与参考

每点每轮 G2 与十二天线方案各使用 864 个不同包。十二天线为 24 个有效完整切换周期 × 6 态 × 6 包；G2 为同一 10 秒记录中的 24 组 × 36 包。每态先计算三链协方差，不制造跨状态同步快照。R1 的前 8 组/周期（每方案共 288 包）建立参考和权重，其余 R1 留作检查；R2 前 12 组/周期预测，后 12 组/周期验证；R3 保留为空场对照。具体时间边界在 `temporal_split_audit.csv` 中，两个方案不是逐包同时观测。

## 同频空间协方差

对同一个包的三链、30 频点 CSI 作总能量归一化：

\[
\bar X_p=\sqrt{90}X_p/\|X_p\|_F,\qquad
R_{s,k}=N_s^{-1}\sum_p\bar x_{s,p,k}\bar x_{s,p,k}^{H}.
\]

同包公共相位与公共频率线性相位在每个频点的外积中消除。它们不是已被估计的 CFO 或硬件误差；固定的链间复数响应仍保留在协方差中。

导向矢量使用实际阵元局部坐标及频率：

\[
a_{s,k,m}(\theta)=\exp\{j2\pi f_k y_m\sin\theta/c\}.
\]

角度网格统一为 −90°:1°:+90°，12 根天线相邻 28 mm，G1–G4 与 C1/C2 按确认的物理映射处理。C1/C2 使用非等距坐标。计算的是分状态平面波诊断方向，尚未证明静态人体路径或物理绝对角度。

## 白化背景投影（WBG）

\[
W_{s,k}=\big[0.9R^{(1)}_{s,k}+0.1\,\mathrm{tr}(R^{(1)}_{s,k})I/3\big]^{-1/2}.
\]

同一 W 作用于空场、目标协方差和导向矢量，不能只变换数据：

\[
\widetilde R=WRW^H,\quad\widetilde a=Wa,\quad
B=I-hh^H,\quad u=\mathrm{eigvec}_{\max}(B\widetilde R B).
\]

h 为白化后的前空场主特征向量。逐态代价为

\[
J_s(\theta)=
\frac{\sum_{k\in\mathcal K_s}\widetilde a^H(B-uu^H)\widetilde a}
{\sum_{k\in\mathcal K_s}\widetilde a^HB\widetilde a}.
\]

普通 BG 令 W=I。有效频点必须满足原空场主特征值占比 ≥0.70、目标投影特征值超过前空场训练块波动、可辨的第一特征向量。每态每频段至少一半频点有效，投影后导向能量至少为原变换后能量的 10%；全部六态均须参与。缺失状态不能偷偷缩成更少天线的结果。

另外两种预先声明内核为：正协方差变化的一阶 MUSIC（POS）；对正负协方差变化均建模、以空场训练波动定阶的 SIGNED MUSIC。SIGNED 只允许 1–2 个变化特征模态，噪声残差按其剩余维数归一化。各方案全部处理五点。

## 仅前空场确定权重

每链对和频点的圆相干度：

\[
\gamma_{s,ab,k}=\left|N^{-1}\sum_p
\frac{x_{s,p,a,k}x^*_{s,p,b,k}}{|x_{s,p,a,k}x^*_{s,p,b,k}|}\right|.
\]

零幅度样本记作零贡献。将 90 个相干度平方裁剪到 [0.05,1] 后取几何均值得 q_s。d_s 为 R1 四个两周期小块相对平均协方差的最大 Frobenius 变化，再跨频点取中位数。

\[
w_s=\operatorname{clip}\{q_s/(1+d_s),0.05,1\},\qquad
J_{\mathrm{WBG-R1}}(\theta)=\frac{\sum_s w_sJ_s(\theta)}{\sum_s w_s}.
\]

R1 权重反映观测重复性，不保证人体信息可靠性。基线融合使用各态分子、分母分别汇总；加权方案使用逐态归一化代价。因此加权方案相对基线同时改变权重和状态平衡，不能把差异全部解释成降权收益。没有逐点经验权重，也没有按真值屏蔽状态。

## 选峰、拒绝及评分

全网格统一取最小代价。保留最多三个相隔 ≥15° 的局部极小点，竞争候选在 3 dB 内则标为未消歧。端点只有确实为单侧极小点才成为候选。后半段及半频段的峰须在前半段峰 ±9° 内，至少三个时间块支持，非边界峰，背景投影能量须超过留出 R1 和 R3，且后空场不能稳定重现同方向；不满足条件时保留候选记录但不发布人体方向。

这些阈值是本轮冻结的内部检查，不是统计校准过的普适检出门槛。R3 假候选计数基于五个空场记录，不能当作人群或环境总体假警率。数据与后空场的变化也可能来自遮挡、散射或支路参考变化。

预测和谱文件哈希冻结后才加入 RX 中心几何角：

\[
\theta_i^*=\operatorname{atan2}(-Y_i,3.65-X_i),\quad
e_i=|\hat\theta_i-\theta_i^*|.
\]

五点都保留。这里 e 是诊断候选与人体真值的角度偏离；只有可靠识别人体路径并校准后才能解释为人体 AoA 精度。G2 子阵中心的几何修正也另存于评分表，未用来挑峰或设权。

## 复现

在工作目录 `<private-workspace>` 运行：

```powershell
python -B -X utf8 output/noncoherent_uniform_20261106/predict_uniform.py
python -B -X utf8 output/noncoherent_uniform_20261106/score_uniform.py
```

依赖前一轮已冻结的 `coherent_array_validation_20261106` 包缓存、解码/几何模块及其数据清单。预测前重新核对 15 个原始 CSI 文件与事件日志哈希和包清单；G2 BG 谱与此前缓存逐值一致。软件检查包括合成场景的导向零空间、W=I 退化一致性、公共相位/斜率不变性，以及候选端点规则。

`before_endpoint_fix/` 保留初次结果；`endpoint_fix_audit.json` 记录单侧极值修复。修复前后所有谱数组、权重和主峰相同，完整候选验收结果也相同。它不是阈值放宽或依据真值进行的调参。

