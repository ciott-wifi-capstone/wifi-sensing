from analyze import *
from threadpoolctl import threadpool_limits
DEST=OUT/'music_diagnostic';DEST.mkdir(exist_ok=True)
# Intel5300 HT40 indices verified against official CSISegment.cxx.
K=np.arange(-58,59,4);F=K*312500.
U=np.linspace(-.5,.5,181);TAU=np.arange(-400,401,2.)
M=15
steer=np.exp(-2j*np.pi*(np.repeat(np.arange(2),M)[:,None,None]*U[None,:,None]+np.tile(F[:M],2)[:,None,None]*TAU[None,None,:]*1e-9)).reshape(2*M,-1)

def covariance(x):
    x=x/(np.linalg.norm(x,axis=(1,2),keepdims=True)+1e-12)
    r=np.zeros((2*M,2*M),complex)
    for a in range(2):
        for k in range(31-M):
            z=x[:,a:a+2,k:k+M].reshape(len(x),-1)
            r+=z.T@z.conj()/len(z)
    return r/(2*(31-M))

def spectrum(r,q):
    ev,e=np.linalg.eigh(r);en=e[:,:-q]
    den=np.sum(np.abs(en.conj().T@steer)**2,axis=0)
    s=(1/np.maximum(den,1e-12)).reshape(len(U),len(TAU));return s/s.max(),ev[::-1]

def main():
    # Synthetic check is only a numerical implementation check, NOT field calibration.
    rng=np.random.default_rng(1);u=.15;t=40e-9
    a=np.exp(-2j*np.pi*(np.arange(3)[:,None]*u+F[None,:]*t))
    x=np.exp(1j*rng.uniform(-np.pi,np.pi,200))[:,None,None]*a+.01*(rng.normal(size=(200,3,30))+1j*rng.normal(size=(200,3,30)))
    s,_=spectrum(covariance(x),1);ui,ti=np.unravel_index(s.argmax(),s.shape)
    assert abs(U[ui]-u)<.006 and abs(TAU[ti]-40)<2.1
    df=pd.read_csv(OUT/'curated_runs.csv');b=pd.read_csv(OUT/'batches.csv');good=set(b[b.status=='PASS'].batch)
    keep=df.label.str.startswith('old')&df.batch.isin(good)&df.complete_window&(df.phase_sd_deg<10)
    selected=df[keep];selected.to_csv(DEST/'selected_runs.csv',index=False,encoding='utf-8-sig')
    specs={};peaks=[];saved={}
    for _,r in selected.iterrows():
        p=Path(r.path);raw=read(p.parent/'run.json');c,host,_=decode(p);t=(host-raw['start_ns'])/1e9
        starts,states,guard=align(p.parent/'events.log',raw['start_ns']/1e9);indices=[]
        for start,st in zip(starts[:-1],states[:-1]):
            if st!=0 or start+.025>raw['requested_seconds']:continue
            i,j=np.searchsorted(t,[start+guard,start+.025-guard])
            if j-i>=5:indices.extend(range(i,j))
        x=c[indices];rxx=covariance(x)
        key=f'{r.label}_{r.batch}_{r.run}'
        for q in [1,2,3,4]:
            s,e=spectrum(rxx,q);ui,ti=np.unravel_index(s.argmax(),s.shape)
            peaks.append(dict(label=r.label,batch=r.batch,run=r.run,assumed_paths=q,spatial_cycles_per_chain=U[ui],raw_delay_ns=TAU[ti],packets=len(x),path=str(p),sha256=r.sha256))
            if q==3:specs.setdefault(r.label,[]).append(s);saved[key]=s
        print(key,len(x),flush=True)
    pd.DataFrame(peaks).to_csv(DEST/'model_order_sensitivity.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(DEST/'spectra.npz',spatial_axis=U,raw_delay_ns=TAU,**saved)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(label,ss) in zip(axs.flat,specs.items()):
        z=np.median(ss,axis=0);z=10*np.log10(np.maximum(z/z.max(),1e-6))
        im=ax.pcolormesh(TAU,U,z,shading='auto',cmap='turbo',vmin=-15,vmax=0)
        ax.set(title=f'{label}: {len(ss)} runs, G1 only',xlabel='Uncalibrated delay parameter (ns)',ylabel='Spatial phase frequency (cycles / chain index)')
    fig.colorbar(im,ax=axs,label='Median relative MUSIC spectrum (dB)')
    fig.suptitle('Measured 2D MUSIC diagnostic — NOT calibrated human AoA–ToF\nNo angle conversion; no empty-room subtraction; assumed model order = 3',fontsize=13)
    fig.savefig(DEST/'music_four_points.png',dpi=170);fig.savefig(DEST/'music_four_points.pdf');plt.close(fig)
    pd.DataFrame(peaks).groupby(['label','assumed_paths'])[['spatial_cycles_per_chain','raw_delay_ns']].agg(['median','min','max']).to_csv(DEST/'peak_ranges.csv')
    (DEST/'README.md').write_text('''# 实测二维 MUSIC 诊断谱，非已验证的人体 AoA–ToF

只采用旧四点中 acquisition PASS、完整窗口、逐轮相位波动小于 10 度的轮次；新 A–D/空场及旧失败批次均未进入。使用 G1 的三接收链，避免未经 CPA 的跨状态拼接。选择依据与 ground truth 数值无关。

原始 CSI 保留频率线性相位，不进行会改变延时的逐包去斜率或真值拟合。逐包单位范数归一化；2 链 × 15 子载波组的重叠子阵平滑协方差；HT40 的 30 个子载波索引为 -58:4:58，间隔 312.5 kHz。主图固定信号子空间维数 3，附表给出 1/2/3/4 的敏感性，不将所设维数当作已知路径数。每轮谱先归一化，再取组内逐像素中位数并重归一化。色条不可用于比较绝对接收功率。

纵轴只是 exp(-j2πmu) 中的空间相位频率，m 为解码后链索引；并非实际角度。只有确认等间距线阵、方向及相位校准后，才能使用 u=d sin(theta)/lambda（相应符号约定）转换。三链模型符合程度也未验证。

横轴为未校准时延参数，含硬件和包检测等未知偏置，800 ns 的采样周期内扫描 -400 至 400 ns；不是传播绝对 ToF，也不是已对齐直达路径的超额时延。2 ns 网格不代表 2 ns 分辨率。负值不代表负传播时间。

未做空场复数相减；峰可能来自直达波、墙体、多径或硬件残差，不能标为人体，更不能用来报告米级定位误差。图只能用于检查谱结构与重复性，不能作为“已完成静态人体定位”的证据。

selected_runs.csv 记录筛选和来源，model_order_sensitivity.csv 记录逐轮各维数的最强峰，spectra.npz 保存完整主图输入。合成单路径测试仅验证实现符号/轴与已知模型一致，不验证现场物理标定。

复现：python -X utf8 music_diagnostic.py。
格式和子载波索引来源：https://github.com/wifisensing/RXS-Parsing-Core/blob/master/CSISegment.cxx
''',encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
