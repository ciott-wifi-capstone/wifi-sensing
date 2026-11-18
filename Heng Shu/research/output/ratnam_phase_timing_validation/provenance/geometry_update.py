from analyze import *
from threadpoolctl import threadpool_limits
from matplotlib.backends.backend_pdf import PdfPages
DEST=OUT/'geometry_update_20261019';DEST.mkdir(exist_ok=True)
C=299792458.;FC=5.23e9;Y=(6.5-np.arange(1,13))*.028
MAP={'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]}
ANG=np.arange(-90.,91.,3);TAU=np.arange(-400.,401.,4);M=15;F=np.arange(-58,59,4)*312500.

def steering(ys,radius=np.inf):
    b=np.deg2rad(ANG)
    # relative RX distance wrt array centre, preserving physical spatial phase.
    delta=-ys[:,None]*np.sin(b)[None,:] if np.isinf(radius) else np.sqrt((radius*np.cos(b)[None,:])**2+(radius*np.sin(b)[None,:]-ys[:,None])**2)-radius
    a=np.exp(-2j*np.pi*(FC/C*np.repeat(delta,M,axis=0)[:,:,None]+np.tile(F[:M],3)[:,None,None]*TAU[None,None,:]*1e-9))
    return a.reshape(3*M,-1)

def covariance(x):
    x=x/(np.linalg.norm(x,axis=(1,2),keepdims=True)+1e-12)
    r=np.zeros((3*M,3*M),complex)
    # Frequency smoothing only. No illegal spatial shifts of nonuniform C arrays.
    for k in range(31-M):
        z=x[:,:,k:k+M].reshape(len(x),-1);r+=z.T@z.conj()/len(z)
    return r/(31-M)

def music(r,a,q=3):
    _,e=np.linalg.eigh(r);den=np.sum(np.abs(e[:,:-q].conj().T@a)**2,axis=0)
    s=(1/np.maximum(den,1e-12)).reshape(len(ANG),len(TAU));return (s/s.max()).astype(np.float32)
def db(s):return 10*np.log10(np.maximum(s/s.max(),1e-6))
def circular_sd(z):return np.sqrt(-2*np.log(np.clip(np.abs(z),1e-12,1)))*180/np.pi

def geometry_check():
    xx=np.linspace(.25,3.65,120);yy=np.linspace(-2.5,2.5,140);X,Q=np.meshgrid(xx,yy);R=np.hypot(X,Q)
    delta=np.sqrt(X[...,None]**2+(Q[...,None]-Y)**2)-R[...,None]
    err=delta+Y*Q[...,None]/R[...,None]
    # Remove arbitrary common phase; peak-to-peak residual is invariant to reference.
    span=np.ptp(err,axis=-1)*FC/C*360
    fig,ax=plt.subplots(figsize=(8,6));im=ax.pcolormesh(X,Q,span,cmap='magma',shading='auto');fig.colorbar(im,ax=ax,label='Full-aperture phase residual span (deg)')
    geo=[]
    for k,xy in COORD.items():
        if xy:
            x,y=xy;d=np.hypot(x,y-Y);r=np.hypot(x,y);e=d-r+Y*y/r
            row=dict(label=k,x=x,y=y,range_rx_m=r,full_aperture_phase_span_deg=np.ptp(e)*FC/C*360)
            for st,ids in MAP.items():
                ys=Y[np.array(ids)-1];centre=ys.mean();rs=np.hypot(x,y-centre)
                # local far-field fit at subarray centre, unlike full-array-centred angle.
                es=np.hypot(x,y-ys)-rs+(ys-centre)*(y-centre)/rs
                row[st+'_local_curvature_deg']=np.ptp(es)*FC/C*360
            geo.append(row);ax.scatter(x,y,c='cyan',s=20);ax.annotate(k,(x,y),xytext=(4,4),textcoords='offset points',color='cyan')
    ax.set(xlabel='x from array centre toward TX (m)',ylabel='y, viewer left positive (m)',title='Geometry-only spherical vs plane-wave discrepancy\nNot measured localization error');fig.tight_layout();fig.savefig(DEST/'near_field_model_error.png',dpi=170);plt.close(fig)
    pd.DataFrame(geo).to_csv(DEST/'near_field_geometry.csv',index=False,encoding='utf-8-sig')
    return geo

def main():
    ast={s:steering(Y[np.array(ids)-1]) for s,ids in MAP.items()}
    # Verify angle sign and delay with synthetic exact same-model channels (no field validation).
    rng=np.random.default_rng(4)
    for st in ['G1','C1','C2']:
        ys=Y[np.array(MAP[st])-1];theta=24.;tau=80.
        a=np.exp(-2j*np.pi*(-FC/C*ys[:,None]*np.sin(np.deg2rad(theta))+F[None,:]*tau*1e-9))
        x=np.exp(1j*rng.uniform(0,6.28,100))[:,None,None]*a+.001*(rng.normal(size=(100,3,30))+1j*rng.normal(size=(100,3,30)))
        s=music(covariance(x),ast[st],1);i,j=np.unravel_index(s.argmax(),s.shape);assert abs(ANG[i]-theta)<3.1 and abs(TAU[j]-tau)<4.1
    df=pd.read_csv(OUT/'curated_runs.csv');df=df[df.use_for_comparison].reset_index(drop=True);df.to_csv(DEST/'inputs.csv',index=False,encoding='utf-8-sig')
    rows=[];closure=[];timing=[];spectra={};agg={};covs={};sphere={}
    for ri,rr in df.iterrows():
        p=Path(rr.path);r=read(p.parent/'run.json');c,host,nic=decode(p);t=(host-r['start_ns'])/1e9
        al=align(p.parent/'events.log',r['start_ns']/1e9);ix={s:[] for s in STATES};bursts={};nw={s:0 for s in STATES}
        if al:
            starts,states,guard=al
            timing.append(dict(index=ri,group=rr.analysis_group,guard_ms=guard*1e3,state_period_ms=float(np.median(np.diff(starts))*1e3),cycle_period_ms=float(np.median(starts[6:]-starts[:-6])*1e3),host_nonmonotonic=int(np.sum(np.diff(host)<0))))
            for ei,(start,st) in enumerate(zip(starts[:-1],states[:-1])):
                if start+.025>r['requested_seconds'] or .025-2*guard<.005:continue
                i,j=np.searchsorted(t,[start+guard,start+.025-guard]);state=STATES[st]
                if j>i:ix[state].extend(range(i,j));nw[state]+=1
                if state in ['C1','C2'] and j-i>=5:
                    prod=c[i:j,1]*c[i:j,2].conj();u=prod/(np.abs(prod)+1e-12)
                    mean=u.mean(0);bursts[(ei//6,state)]=(mean,start,j-i)
        # Antennas 8 and 6 appear on the same receiver chains in C1 and C2.
        for cycle in sorted(set(k[0] for k in bursts)):
            if (cycle,'C1') not in bursts or (cycle,'C2') not in bursts:continue
            u,t1,n1=bursts[(cycle,'C1')];v,t2,n2=bursts[(cycle,'C2')]
            z=v*np.conj(u);unit=z/(np.abs(z)+1e-12)
            closure.append(dict(index=ri,group=rr.analysis_group,batch=rr.batch,run=rr.run,cycle=cycle,gap_ms=(t2-t1)*1000,packets_C1=n1,packets_C2=n2,median_abs_residual_deg=float(np.median(np.abs(np.angle(unit,deg=True)))),signed_mean_deg=float(np.angle(unit.mean(),deg=True)),across_tone_coherence=float(np.abs(unit.mean())),within_burst_coherence=float(np.median(np.minimum(np.abs(u),np.abs(v))))))
        for st in STATES:
            n=len(ix[st]);key=f'{ri:03d}_{st}';row=dict(index=ri,group=rr.analysis_group,state=st,packets=n,windows=nw[st],status='insufficient' if n<30 else 'sparse' if n<200 or nw[st]<10 else 'computed',path=rr.path,sha256=rr.sha256)
            if n>=30:
                rxx=covariance(c[ix[st]]);covs[key]=rxx;s=music(rxx,ast[st]);spectra[key]=s;agg.setdefault((rr.analysis_group,st),[]).append(s)
                i,j=np.unravel_index(s.argmax(),s.shape);row.update(angle_search_deg=ANG[i],raw_delay_ns=TAU[j])
            rows.append(row)
        print(ri+1,rr.analysis_group,rr.run,flush=True)
    pd.DataFrame(rows).to_csv(DEST/'state_results.csv',index=False,encoding='utf-8-sig');pd.DataFrame(closure).to_csv(DEST/'C1_C2_closure.csv',index=False,encoding='utf-8-sig');pd.DataFrame(timing).to_csv(DEST/'timing.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(DEST/'covariances.npz',**covs);np.savez_compressed(DEST/'spectra.npz',angle_deg=ANG,delay_ns=TAU,**spectra)
    groups=list(df.analysis_group.unique())
    with PdfPages(DEST/'all_groups_six_states.pdf') as pdf:
        for g in groups:
            fig,axs=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
            for ax,st in zip(axs.flat,STATES):
                ss=agg.get((g,st),[])
                if ss:im=ax.pcolormesh(TAU,ANG,db(np.median(ss,axis=0)),vmin=-15,vmax=0,cmap='turbo',shading='auto',rasterized=True)
                else:ax.text(.5,.5,'Insufficient data',ha='center',transform=ax.transAxes)
                ax.set(title=f'{st} {MAP[st]}: {len(ss)} runs',xlabel='Raw delay (ns)',ylabel='Angle search (deg; +left)')
            fig.colorbar(im,ax=axs,label='Relative MUSIC (dB)',shrink=.7);fig.suptitle(f'{g}: actual antenna coordinates, separate-state 2D MUSIC\nUNCALIBRATED; frequency smoothing only; no coherent 12-element assembly',fontsize=12)
            fig.savefig(DEST/(g.replace(' ','_')+'_six_states.png'),dpi=140);pdf.savefig(fig);plt.close(fig)
    cl=pd.DataFrame(closure);cr=cl.groupby(['index','group']).agg(residual=('median_abs_residual_deg','median'),coherence=('within_burst_coherence','median'),cycles=('cycle','size')).reset_index();cr.to_csv(DEST/'closure_per_run.csv',index=False,encoding='utf-8-sig')
    cg=cr.groupby('group',sort=False).agg(residual_deg=('residual','median'),coherence=('coherence','median'),runs=('index','size'));cg.to_csv(DEST/'closure_groups.csv',encoding='utf-8-sig')
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,col,title in zip(axs,['residual','coherence'],['C1/C2 shared-pair residual (degrees)','Within-burst shared-pair coherence']):
        ax.boxplot([cr[cr.group==g][col] for g in groups],tick_labels=groups);ax.tick_params(axis='x',rotation=60);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Antennas 8 and 6: common-packet phase cancels; cross-state check is necessary, not sufficient')
    fig.savefig(DEST/'reference_closure.png',dpi=170);plt.close(fig)
    geo=geometry_check()
    # Data-based curvature sensitivity at predeclared receive ranges, no GT-fitting.
    # Average per-run unit-trace covariance equally within group, G1 and widest reference C1.
    with PdfPages(DEST/'spherical_sensitivity.pdf') as pdf:
        for g in groups:
            fig,axs=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
            for si,st in enumerate(['G1','C1']):
                keys=[f'{i:03d}_{st}' for i,r in df.iterrows() if r.analysis_group==g and f'{i:03d}_{st}' in covs]
                rxx=np.mean([covs[k]/np.trace(covs[k]).real for k in keys],axis=0)
                for j,rad in enumerate([np.inf,1.,2.]):
                    s=music(rxx,steering(Y[np.array(MAP[st])-1],rad));sphere[f'{g}_{st}_{rad}']=s
                    ax=axs[si,j];im=ax.pcolormesh(TAU,ANG,db(s),vmin=-15,vmax=0,cmap='turbo',shading='auto',rasterized=True);ax.set(title=f'{st}, RX range hypothesis: {rad} m',xlabel='Raw delay (ns)',ylabel='Angle search (deg)')
            fig.colorbar(im,ax=axs,label='Relative MUSIC dB',shrink=.7);fig.suptitle(f'{g}: plane vs spherical receive-wave model sensitivity\nRanges are fixed hypotheses, NOT estimated positions; no hardware calibration')
            pdf.savefig(fig)
            if g in ['B','old+1']:fig.savefig(DEST/(g+'_spherical_sensitivity.png'),dpi=145)
            plt.close(fig)
    np.savez_compressed(DEST/'spherical_sensitivity.npz',angle_deg=ANG,delay_ns=TAU,**sphere)
    (DEST/'geometry.json').write_text(json.dumps(dict(antenna_y_m=Y.tolist(),state_mapping=MAP,tx=[3.65,0],ground_truth_origin='User confirmed RX 6/7 midpoint',geometry_confirmed=True,aperture_m=.308,frequency_hz=FC,far_field_scale_m=2*.308**2/(C/FC),method='Separate-state 2D MUSIC, 3 antennas x 15 frequency groups; frequency-only smoothing; signal subspace dimension 3; narrowband spatial phase at 5.23 GHz; no spatial smoothing or cross-state coherent assembly',spherical_ranges_m=[1.,2.],sphere_note='Fixed range sensitivity; unknown common delay remains free. Not full bistatic position inference.'),indent=2),encoding='utf8')
    print(cg.to_string());print(pd.DataFrame(geo)[['label','full_aperture_phase_span_deg']].to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
