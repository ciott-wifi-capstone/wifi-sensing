from music_diagnostic import covariance, spectrum, U, TAU, F, M, threadpool_limits
from analyze import *
from matplotlib.backends.backend_pdf import PdfPages
DEST=OUT/'music_all';DEST.mkdir(exist_ok=True)
ANGLE=np.arange(-90,91,1.)
RATIO=.028*5.23e9/299792458.
AST=np.exp(-2j*np.pi*(np.repeat(np.arange(2),M)[:,None,None]*RATIO*np.sin(np.deg2rad(ANGLE))[None,:,None]+np.tile(F[:M],2)[:,None,None]*TAU[None,None,:]*1e-9)).reshape(2*M,-1)

def angle_spectrum(r,q=3):
    _,e=np.linalg.eigh(r);den=np.sum(np.abs(e[:,:-q].conj().T@AST)**2,axis=0)
    s=(1/np.maximum(den,1e-12)).reshape(len(ANGLE),len(TAU));return (s/s.max()).astype(np.float32)

def delay_spectrum(x):
    x=x/(np.linalg.norm(x,axis=(1,2),keepdims=True)+1e-12);r=np.zeros((M,M),complex)
    for k in range(31-M):
        z=x[:,:,k:k+M].reshape(-1,M);r+=z.T@z.conj()/len(z)
    _,e=np.linalg.eigh(r);st=np.exp(-2j*np.pi*F[:M,None]*TAU[None,:]*1e-9)
    s=1/np.maximum(np.sum(np.abs(e[:,:-3].conj().T@st)**2,axis=0),1e-12)
    return (s/s.max()).astype(np.float32)

def db(s):return 10*np.log10(np.maximum(s/s.max(),1e-6))
def main():
    df=pd.read_csv(OUT/'curated_runs.csv');df=df[df.use_for_comparison].copy()
    # Use every unique capture, irrespective of pass/fail, phase variance or date.
    assert len(df)==90 and df.sha256.nunique()==90
    df.to_csv(DEST/'input_runs.csv',index=False,encoding='utf-8-sig')
    groups=list(df.analysis_group.unique());spectra={};angles={};delays={};rows=[];agg={};aagg={};dagg={};runpages=[]
    with PdfPages(DEST/'all_runs_G1_G4.pdf') as pdf:
        for ri,(_,rr) in enumerate(df.iterrows()):
            p=Path(rr.path);r=read(p.parent/'run.json');c,host,_=decode(p);t=(host-r['start_ns'])/1e9
            al=align(p.parent/'events.log',r['start_ns']/1e9);ix={s:[] for s in STATES};nw={s:0 for s in STATES}
            if al:
                starts,states,guard=al
                for start,st in zip(starts[:-1],states[:-1]):
                    if start+.025>r['requested_seconds'] or .025-2*guard<.005:continue
                    i,j=np.searchsorted(t,[start+guard,start+.025-guard])
                    # Include even sparse interiors; do not silently delete difficult captures.
                    if j>i:ix[STATES[st]].extend(range(i,j));nw[STATES[st]]+=1
            fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained');rs={}
            for st in STATES:
                inds=ix[st];key=f'{ri:03d}_{st}';n=len(inds)
                status='sparse' if n<200 or nw[st]<10 else 'computed'
                if n<30:status='insufficient (<30 packets)'
                row=dict(index=ri,group=rr.analysis_group,batch=rr.batch,run=rr.run,state=st,packets=n,windows=nw[st],status=status,complete_window=rr.complete_window,phase_sd_deg=rr.phase_sd_deg,path=rr.path,sha256=rr.sha256)
                if n>=30:
                    x=c[inds]
                    if st.startswith('G'):
                        rxx=covariance(x);s,_=spectrum(rxx,3);s=s.astype(np.float32);spectra[key]=s;rs[st]=s
                        agg.setdefault((rr.analysis_group,st),[]).append(s)
                        ui,ti=np.unravel_index(s.argmax(),s.shape);row.update(raw_delay_ns=TAU[ti],spatial_frequency=U[ui])
                        if st=='G1':
                            a=angle_spectrum(rxx);angles[key]=a;aagg.setdefault(rr.analysis_group,[]).append(a)
                            ai,ti=np.unravel_index(a.argmax(),a.shape);row.update(local_angle_search_deg=ANGLE[ai],angle_model_delay_ns=TAU[ti])
                    else:
                        s=delay_spectrum(x);delays[key]=s;dagg.setdefault((rr.analysis_group,st),[]).append(s)
                        row['raw_delay_ns']=TAU[s.argmax()]
                rows.append(row)
                if st.startswith('G'):
                    ax=axs.flat[int(st[1])-1]
                    if st in rs:ax.pcolormesh(TAU,U,db(rs[st]),vmin=-15,vmax=0,cmap='turbo',shading='auto',rasterized=True)
                    else:ax.text(.5,.5,'Insufficient observed packets\nNo fabricated spectrum',ha='center',transform=ax.transAxes)
                    ax.set(title=f'{st}: {n} packets / {nw[st]} windows; {status}',xlabel='Uncalibrated delay (ns)',ylabel='Spatial cycles / chain index')
            fig.suptitle(f'{rr.analysis_group} | {rr.batch} | {rr.run}\nMeasured diagnostic, q=3; NOT calibrated human parameters',fontsize=11)
            pdf.savefig(fig);plt.close(fig)
            print(ri+1,rr.analysis_group,rr.run,{st:len(ix[st]) for st in STATES},flush=True)
    pd.DataFrame(rows).to_csv(DEST/'state_results.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(DEST/'all_spectra.npz',spatial_axis=U,delay_ns=TAU,**spectra)
    np.savez_compressed(DEST/'G1_local_angle_spectra.npz',angle_deg=ANGLE,delay_ns=TAU,**angles)
    np.savez_compressed(DEST/'reference_state_delay_spectra.npz',delay_ns=TAU,**delays)
    fig,axs=plt.subplots(5,2,figsize=(13,17),layout='constrained')
    for ax,g in zip(axs.flat,groups):
        ss=aagg.get(g,[]);n=int((df.analysis_group==g).sum())
        if ss:im=ax.pcolormesh(TAU,ANGLE,db(np.median(ss,axis=0)),shading='auto',rasterized=True,vmin=-15,vmax=0,cmap='turbo')
        ax.set(title=f'{g}: G1, {len(ss)}/{n} runs computed',xlabel='Uncalibrated delay (ns)',ylabel='Local angle search (deg)')
    fig.colorbar(im,ax=axs,label='Median normalized MUSIC (dB)',shrink=.7)
    fig.suptitle('All datasets: measured 2D MUSIC, G1 / 28 mm ULA model\nAngle assumes zero chain phase offsets; sign NOT mapped to room; no human-path identification',fontsize=12)
    fig.savefig(DEST/'all_groups_G1_angle.png',dpi=155);fig.savefig(DEST/'all_groups_G1_angle.pdf');plt.close(fig)
    fig,axs=plt.subplots(10,4,figsize=(18,28),layout='constrained')
    for i,g in enumerate(groups):
        for j,st in enumerate(STATES[:4]):
            ax=axs[i,j];ss=agg.get((g,st),[])
            if ss:im=ax.pcolormesh(TAU,U,db(np.median(ss,axis=0)),vmin=-15,vmax=0,cmap='turbo',shading='auto',rasterized=True)
            else:ax.text(.5,.5,'Insufficient data',ha='center',transform=ax.transAxes)
            ax.set(title=f'{g} / {st}: {len(ss)} runs',xlabel='Raw delay ns',ylabel='Spatial cycles / chain')
    fig.suptitle('All groups / G1-G4: chain-index model diagnostics, NOT calibrated AoA',fontsize=15)
    fig.savefig(DEST/'all_groups_G1_G4.png',dpi=120);fig.savefig(DEST/'all_groups_G1_G4.pdf');plt.close(fig)
    fig,axs=plt.subplots(5,2,figsize=(12,14),layout='constrained')
    for ax,g in zip(axs.flat,groups):
        for st in ['C1','C2']:
            ss=dagg.get((g,st),[])
            if ss:ax.plot(TAU,db(np.median(ss,axis=0)),label=f'{st}, {len(ss)} runs')
        ax.set(title=g,xlabel='Uncalibrated delay (ns)',ylabel='Relative MUSIC (dB)',ylim=(-20,1));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('C1/C2 reference-state delay diagnostics; no ULA angle model applied')
    fig.savefig(DEST/'reference_states_delay.png',dpi=130);plt.close(fig)
    (DEST/'README.md').write_text('''# 全部新旧数据的实测 MUSIC 图

输入为全部 90 轮去重 CSI，原 93 个路径中的 3 个 kong 副本按用户确认归入 A。包含旧失败批次、新 A–D、日志日期为 10 月 3 日和 18 日的空场。未按结果好坏筛选；所有 540 个“轮次×状态”都有 state_results.csv 记录。

G1 的角度轴采用用户确认的三阵元直线、28 mm 间距和 5.23 GHz。0 度是模型的阵列法线，正负号由 a_m=exp(-j2π m d sin(theta)/lambda) 定义；尚未映射到房间正负 y。假设链间固定相位偏置为零，因而是未校准角度搜索坐标，不能当作已验证的人体 AoA。

核对用户粘贴的海报对话：2082–2087 行给出 G1=1,2,3；G2=4,5,6；G3=7,8,9；G4=10,11,12；C1=1,8,6；C2=10,8,6。3831 行用户说明之前直接采用静态房间 R1 各天线相位参考；该方法不是独立硬件标定。2230 行旧指南间距是文献/指南参数，本次 G1 采用最新用户确认的 28 mm。未找到实际链序相对房间左右方向的明确文字，不猜测。没有把 R1 参考自动迁移到本次数据。

G1–G4 分状态计算，未跨状态拼接阵列。G2–G4 图保留链索引空间相位轴，不假定它们与 G1 相同的物理方向或间距。它们的线性链索引导向矢量是诊断模型，不能解释为各组的真实物理 AoA。C1/C2 连接非连续阵元，因此仅提供跨频率的一维时延谱，没有强行套用 28 mm 三阵元 AoA 模型。

采用前一轮同一 Intel5300 解码器和时间守护窗。每个非空守护窗中的包全部进入对应状态；状态总计不足 30 包时不伪造谱，标记 insufficient。少于 200 包或少于 10 个窗口标记 sparse，仍计算并保留。无数据状态、丢失时段无法补成测量。保留旧截断轮次的实际观测部分。

2×15 空频子阵前向平滑、逐包单位范数归一化、固定信号子空间维数 3，30 个 HT40 索引 -58:4:58 ×312.5 kHz。G1 角度网格 1 度，未校准时延扫描 -400:2:400 ns。网格间隔不是分辨率。无空场复数相减、无测试坐标引导挑峰、无逐包线性相位去除。组图为各轮归一化谱的逐像素中位数再次归一化，色条不表示绝对功率；所有 sparse 谱亦纳入，详细样本量见逐轮表。

文件：all_groups_G1_angle.png/pdf 为 10 组 G1 局部角度–时延搜索图；all_groups_G1_G4.png/pdf 为全部组别四观测状态；all_runs_G1_G4.pdf 为 90 页逐轮图；reference_states_delay.png 为 C1/C2 时延图。三个 npz 保存全部已算谱与轴，input_runs.csv 和 state_results.csv 保存文件来源、哈希、数量和状态。数组键编号与 state_results.csv 的 index 对应。

这些是由实测 CSI 计算的 MUSIC 诊断谱，但尚不能把谱峰确定为人体路径、绝对 ToF 或定位精度。原始时延偏移没有校正，不应直接与 ground truth 的 0.9–5.9 ns 超额时延对比。用户提供的理论点位未参与谱估计。

复现：python -X utf8 music_all.py。未修改原始数据。
''',encoding='utf-8')
    print(pd.DataFrame(rows).groupby(['state','status']).size().to_string(),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
