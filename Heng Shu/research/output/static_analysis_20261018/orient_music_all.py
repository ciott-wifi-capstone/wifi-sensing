from music_all import *

def main():
    data=np.load(DEST/'G1_local_angle_spectra.npz');df=pd.read_csv(DEST/'input_runs.csv').reset_index(drop=True)
    states=pd.read_csv(DEST/'state_results.csv');angles=data['angle_deg'];tau=data['delay_ns']
    # Latest user correction: viewed from behind RX toward TX, 1 -> 2 -> 3 left -> right.
    # In a left-positive lateral coordinate y_m=-m*d. Under exp(-j*k*path length),
    # the far-field manifold is exp(-j*2*pi*m*d*sin(beta)/lambda), matching saved axes.
    assert np.allclose(angles,ANGLE)
    groups=list(df.analysis_group.unique())
    agg={g:[data[f'{i:03d}_G1'] for i,r in df.iterrows() if r.analysis_group==g and f'{i:03d}_G1' in data] for g in groups}
    def plot(gs,name,nrows):
        fig,axs=plt.subplots(nrows,2,figsize=(12,3.5*nrows+1),layout='constrained',squeeze=False)
        for ax,g in zip(axs.flat,gs):
            ss=agg[g];total=int((df.analysis_group==g).sum());sel=states[(states.group==g)&(states.state=='G1')];sparse=int((sel.status=='sparse').sum())
            im=ax.pcolormesh(tau,angles,db(np.median(ss,axis=0)),vmin=-15,vmax=0,cmap='turbo',shading='auto',rasterized=True)
            ax.set(title=f'{g}: {len(ss)}/{total} runs; {sparse} sparse',xlabel='Uncalibrated delay parameter (ns)',ylabel='AoA search (deg; +left / -right)')
        fig.colorbar(im,ax=axs,label='Median normalized MUSIC spectrum (dB)',shrink=.8)
        fig.suptitle('Measured 2D MUSIC | G1, 28 mm | UNCALIBRATED\nViewed behind RX toward TX: antennas 1-2-3 left-to-right; 0 deg = array broadside',fontsize=12)
        fig.canvas.draw(); fig.savefig(DEST/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(DEST/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    plot(groups,'all_groups_G1_angle',5)
    plot(groups[:4],'old_four_points',2);plot(['A','B','C','D'],'new_ABCD',2);plot(['empty Oct03','empty Oct18'],'empty_groups',1)
    # Short geometry illustration to make the sign convention auditable.
    fig,ax=plt.subplots(figsize=(7,5));d=.028
    ax.scatter([0,0,0],[d,0,-d],s=60,c='tab:blue')
    for i,y in enumerate([d,0,-d],1):ax.text(-.008,y,f'{i}',ha='right')
    ax.annotate('TX / 0 deg',xy=(.1,0),xytext=(.025,0),arrowprops=dict(arrowstyle='->'),va='center')
    ax.annotate('+ angle: viewer left',xy=(.08,.045),xytext=(.008,.075),arrowprops=dict(arrowstyle='->'))
    ax.annotate('- angle: viewer right',xy=(.08,-.045),xytext=(.008,-.075),arrowprops=dict(arrowstyle='->'))
    ax.set(xlim=(-.025,.13),ylim=(-.09,.09),xlabel='Forward from RX toward TX (schematic)',ylabel='Lateral: positive to viewer left',title='Latest user-confirmed chain order; 28 mm adjacent spacing');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(DEST/'angle_convention.png',dpi=160);plt.close(fig)
    metadata=dict(spacing_m=.028,frequency_hz=5.23e9,order='Viewed from behind RX toward TX: chain 1 to 2 to 3 left to right',angle_zero='array normal toward TX',angle_positive='viewer left',manifold='exp(-j*2*pi*m*d*sin(beta)/lambda)',calibration='not performed; chain phase offsets assumed zero',source='latest user correction plus poster image3.jpeg confirming linear 12-element hardware')
    (DEST/'orientation.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    p=DEST/'README.md';s=p.read_text(encoding='utf8')
    s=s.replace('0 度是模型的阵列法线，正负号由 a_m=exp(-j2π m d sin(theta)/lambda) 定义；尚未映射到房间正负 y。','按用户最后纠正：从 RX 后方看向 TX，链 1→2→3 从左到右。0 度为朝向 TX 的阵列法线，面向 TX 时左侧为正角、右侧为负角；使用 a_m=exp(-j2π m d sin(theta)/lambda)。此方向定义不消除链间硬件相位偏差。')
    s=s.replace('未找到实际链序相对房间左右方向的明确文字，不猜测。','最初粘贴对话没有明确链序左右关系；随后用户确认了从 RX 后方朝 TX 看从左到右。PPTX 实拍 image3.jpeg 确认十二根天线为直线结构，编号方向以用户最新说明为准。')
    s+='\n方向修订复现：完成 music_all.py 后运行 orient_music_all.py。新增 old_four_points、new_ABCD、empty_groups 三套 PNG/PDF；orientation.json 和 angle_convention.png 给出可核查的角度约定。角度正负号沿用标准负时延相位模型，尚需已知方向实测验证硬件符号。海报原 R1 场景相位参考未在本轮套用。\n'
    p.write_text(s,encoding='utf8')
    assert len(states)==540
    assert states[['index','state']].drop_duplicates().shape[0]==540
    assert len(data.files)-2==89
    print('90 unique runs; 540 state attempts; 89 G1 angle spectra; latest orientation applied.')

if __name__=='__main__':main()
