from run_check import *
from disambiguate import coh
import py_compile

def main():
    for f in ['model_freeze.json','spectral_predictions_frozen.json','branch_development_freeze.json','decisions_frozen.json']:verify(R/f)
    src=pd.read_csv(R/'source_manifest.csv')
    for p in src.itertuples():assert sha(p.path)==p.sha256
    # Identity must reproduce the immediately preceding frozen same-batch baseline.
    err=0.
    for first,off,s in itertools.product(FIRST,[0,1,2],ST):
        idx=first+off;a=np.load(R/'spectra'/f'identity_{idx}_{s}.npz');b=np.load(A/'occupied_spectra'/f'E1_{idx}_{s}.npz')
        err=max(err,float(abs(a['scores']-b['scores']).max()))
    assert err<1e-10,err
    # A static coherent mixture can look rank one and generate a false hardware phase template.
    freq=sm.F[15];d=np.sqrt(3.65**2+Y**2);a0=np.exp(-2j*np.pi*freq*d/C)
    a1=np.exp(-2j*np.pi*freq*3.65/C)*np.exp(2j*np.pi*freq*Y*np.sin(np.deg2rad(31))/C)
    h=a0+.7*np.exp(.3j)*a1
    apparent=np.angle(h/a0);apparent-=apparent[0]
    false_phase_rms=float(np.degrees(np.sqrt(np.mean(apparent**2))))
    assert false_phase_rms>5
    # Shared-graph synthetic hardware recovery from exact, known phase constraints.
    true=np.r_[0,np.linspace(-.8,1.1,11)];edges=[];obs=[]
    for s in ST:
        ids=np.array(sm.MAP[s])-1
        for i,j in PA:edges.append([ids[i],ids[j]]);obs.append(np.exp(1j*(true[ids[j]]-true[ids[i]])))
    fitted,_=fitgraph(edges,obs,np.ones(len(obs)),12)
    e=float(np.max(abs(np.angle(np.exp(1j*(fitted-true))))));assert e<1e-7
    # Verify phase/steering conventions with an exact synthetic plane wave on the search grid.
    s='G2';yy=Y[np.array(sm.MAP[s])-1]
    a=np.exp(2j*np.pi*yy[:,None]*sm.F[None,:]*np.sin(np.deg2rad(21))/C)
    x=np.tile(a[None],(128,1,1));inv=invariants(x,s)
    assert inv['best_single_plane_angle_parameter_deg']==21 and inv['single_plane_phase_rms_deg']<1e-8
    # Same-frequency pair phases/covariances are invariant to arbitrary common packet phase and timing.
    rng=np.random.default_rng(1105);phase=np.exp(1j*rng.normal(size=(len(x),1,1))-2j*np.pi*(sm.F-sm.F.mean())[None,None,:]*rng.normal(size=(len(x),1,1))*1e-8)
    ce=float(abs(sm.spatial(x)-sm.spatial(x*phase)).max());assert ce<1e-12
    controls=pd.read_csv(R/'coordinate_controls.csv');assert controls.coordinate_change_score_max_error.max()<1e-8
    scores=pd.read_csv(R/'occupied_scores.csv');null=pd.read_csv(R/'empty_scores.csv');assert len(scores)==288 and len(null)==96
    v=pd.read_csv(R/'validated_branches.csv');assert not v[v.batch_first==0].complete_repeat.any()
    gt=pd.read_csv(R/'all_branch_truth_comparison.csv');assert len(gt)==len(v)
    for j in [1,2,3]:assert np.allclose(gt[f'absolute_difference_r{j}'],abs(gt[f'angle_r{j}']-gt.theory_deg),equal_nan=True)
    released=pd.read_csv(R/'release_decisions.csv');assert not released.human_output.any() and not released.position_output.any()
    assert .99<np.median(coh('C2',-39,24))<1
    assert .30<np.median(coh('G1',-39,24))<.35
    for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
    result=dict(status='PASS',source_files=len(src),baseline_reproduction_max_error=err,synthetic_graph_phase_error_rad=e,
                synthetic_plane_wave_direction_deg=inv['best_single_plane_angle_parameter_deg'],common_phase_timing_invariance=ce,
                static_multipath_false_hardware_phase_RMS_deg=false_phase_rms,coordinate_change_invariance_max=controls.coordinate_change_score_max_error.max(),
                occupied_state_model_cases=len(scores),empty_state_model_cases=len(null),partial_run_retained=True,
                human_outputs=0,position_outputs=0,all_prediction_freezes_verified=True)
    (R/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
