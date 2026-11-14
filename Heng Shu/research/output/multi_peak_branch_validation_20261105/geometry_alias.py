from branch_analysis import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def coherence(state,t1,t2):
    y=np.array(sm.GEO['array_y_local_m'])[np.array(sm.MAP[state])-1]
    a=np.exp(2j*np.pi*sm.F[:,None]*y[None,:]*np.sin(np.deg2rad(t1))/299792458.)
    b=np.exp(2j*np.pi*sm.F[:,None]*y[None,:]*np.sin(np.deg2rad(t2))/299792458.)
    return abs(np.sum(a.conj()*b,axis=1))/len(y)

def main():
    verify_freeze(R/'development_freeze.json');verify_freeze(R/'predictions_frozen.json')
    dev=pd.read_csv(R/'development_branches.csv');rows=[]
    freeze('alias_inputs_frozen.json',[R/'development_branches.csv',R/'geometry_alias.py',R/'ALIAS_DIAGNOSTIC_ADDENDUM.md',O/'static_human_aoa_validation_20261101/geometry.json'])
    for (ref,first,state),q in dev.groupby(['reference','batch_first','state']):
        for a,b in itertools.combinations(q.itertuples(),2):
            rho=coherence(state,a.center_dev_deg,b.center_dev_deg)
            rows.append(dict(reference=ref,batch_first=first,state=state,branch_a=a.branch_id,branch_b=b.branch_id,
                angle_a=a.center_dev_deg,angle_b=b.center_dev_deg,coherence_min=rho.min(),coherence_median=np.median(rho),coherence_mean=rho.mean(),coherence_max=rho.max(),measured_CSI_correlation=False))
    df=write(rows,'branch_steering_similarity.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.5),layout='constrained')
    mats={}
    for ax,state in zip(axes,['G3','C2']):
        mat=np.array([[coherence(state,a,b).mean() for b in sm.ANG] for a in sm.ANG]);mats[state]=mat
        im=ax.imshow(mat,extent=[-91.5,91.5,-91.5,91.5],origin='lower',vmin=0,vmax=1,cmap='viridis',aspect='equal')
        ax.set_title(f'{state}: '+('28 mm spacing' if state=='G3' else '56 mm spacing'))
        ax.set_xlabel('Second angle (deg)');ax.set_ylabel('First angle (deg)');ax.set_xticks([-90,-45,0,45,90]);ax.set_yticks([-90,-45,0,45,90])
        if state=='C2':ax.scatter([24],[-39],facecolors='none',edgecolors='red',s=120,lw=1.7)
    fig.colorbar(im,ax=axes,label='Mean ideal steering-vector similarity',shrink=.85)
    fig.suptitle('Array geometry ambiguity (model diagnostic; no human ground truth)',fontsize=11)
    fig.savefig(R/'array_alias_diagnostic.png',dpi=180);fig.savefig(R/'array_alias_diagnostic.pdf');plt.close(fig)
    np.savez_compressed(R/'array_alias_matrices.npz',angles=sm.ANG,**mats)
    ex=df[(df.reference=='E1')&(df.batch_first==12)&(df.state=='C2')]
    print(ex.to_string(index=False))
    freeze('alias_outputs_frozen.json',[R/'branch_steering_similarity.csv',R/'array_alias_matrices.npz'])
if __name__=='__main__':main()
