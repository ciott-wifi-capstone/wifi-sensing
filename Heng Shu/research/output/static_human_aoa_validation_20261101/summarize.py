"""Summaries and publication figures from frozen diagnostics; evaluation only reads geometry."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,name):
 fig.savefig(R/'figures'/f'{name}.pdf');fig.savefig(R/'figures'/f'{name}.png',dpi=160);plt.close(fig)
def table(suffix):
 files=[m.with_name(m.name.replace('_complete.json',f'_{suffix}.csv')) for m in sorted((R/'results').glob('*_complete.json'))]
 return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
def main():
 states=table('states');pred=table('predictions');params=table('packet_parameters');blocks=table('temporal_blocks')
 for n,x in [('state_summary',states),('prediction_records',pred),('packet_parameters_all',params),('temporal_blocks_all',blocks)]:x.to_csv(R/'results'/f'{n}.csv',index=False)
 summary=[]
 for (stage,band,method),g in states.groupby(['stage','band','method']):
  p=pred[(pred.stage==stage)&(pred.method==method)&(pred['index'].astype(str).isin(g['index'].astype(str)))];good=p[p.applied]
  summary.append(dict(stage=stage,band=band,method=method,states=len(g),passed=int(g.common_model_pass.sum()),packet_acceptance_mean=g.accepted_fraction.mean(),prediction_records=len(p),applied_records=len(good),heldout_correlation_median=good.test_magnitude.median(),raw_abs_delay_median_ns=good.raw_test_delay_ns.abs().median(),corrected_abs_delay_median_ns=good.actual_test_residual_delay_ns.abs().median(),raw_abs_phase_median_deg=np.rad2deg(good.raw_test_phase_rad.abs()).median(),corrected_abs_phase_median_deg=np.rad2deg(good.actual_test_residual_phase_rad.abs()).median()))
 pd.DataFrame(summary).to_csv(R/'results/preprocessing_summary.csv',index=False)
 psummary=pred.groupby(['stage','method','mask']).agg(records=('mask','size'),applied=('applied','sum'),median_test_magnitude=('test_magnitude','median')).reset_index();psummary.to_csv(R/'results/mask_summary.csv',index=False)
 # B has one common phase and zero slope; recover the applied rotation from saved observations.
 bpar=[]
 for f in sorted((R/'results').glob('*_complete.json')):
  meta=json.loads(f.read_text(encoding='utf-8'));key=str(meta['index'])
  for s in STATES:
   z=np.load(R/'packets'/f'{key}_{s}.npz');ph=np.angle(np.einsum('nmk,nmk->n',z['out_B'],z['x'].conj()))
   for i in range(len(ph)):bpar.append(dict(index=key,state=s,packet_index=int(z['packet_id'][i]),method='B',tau_ns=0,psi_rad=ph[i],reference=f'{key}:{s}:causal_running_phase_reference',applied=i>0,source=meta['source'],raw_sha256=meta['raw_sha256'],status='unconditional_phase_only_control'))
 pd.DataFrame(bpar).to_csv(R/'results/B_packet_parameters.csv',index=False)
 # Frozen cross-record graph prediction: primary uses a pooled development template, never a position-specific one.
 edge=pd.concat([pd.read_csv(R/'results'/f'graph_{s}_edges.csv') for s in ['development','evaluation']]);tpl=pd.read_csv(R/'results/frozen_graph_templates.csv');pr=[]
 for e in edge[edge.stage=='evaluation'].itertuples():
  cohort='Office' if str(e.index).startswith('office') else 'primary';r=tpl[(tpl.cohort==cohort)&(tpl.edge==e.edge)].iloc[0];rat=engine('HT20' if cohort=='Office' else 'HT40');dt=e.delay_ns-r.delay_ns;wd=float(rat.wrap_delay(dt));phase=float(rat.wrap_phase(e.phase_rad-r.phase_rad+2*np.pi*rat.F[0]*(dt-wd)*1e-9))
  pr.append(dict(index=e.index,cohort=cohort,cycle=e.cycle,edge=e.edge,antenna=e.antenna,phase_error_deg=abs(phase)*180/np.pi,delay_error_ns=abs(wd),correlation=e.correlation,passed=abs(phase)*180/np.pi<=15 and abs(wd)<=5 and e.correlation>=.9,template='Office_R1' if cohort=='Office' else 'pooled_four_development_runs',interpretation='reference-transfer diagnostic; no fixed-chain calibration'))
 gp=pd.DataFrame(pr);gp.to_csv(R/'results/graph_frozen_predictions.csv',index=False)
 gr=[]
 for (cohort,ed),g in gp.groupby(['cohort','edge']):gr.append(dict(cohort=cohort,edge=ed,n=len(g),phase_p90_deg=g.phase_error_deg.quantile(.9),delay_p90_ns=g.delay_error_ns.quantile(.9),pass_fraction=g.passed.mean()))
 pd.DataFrame(gr).to_csv(R/'results/graph_prediction_summary.csv',index=False)
 # Within-cycle overlap closures and leave-one-edge checks do not make independent hardware validation.
 closures=pd.concat([pd.read_csv(R/'results'/f'graph_{s}_closure.csv') for s in ['development','evaluation']]);cs=[]
 for (ix,loop),g in closures.groupby(['index','loop']):cs.append(dict(index=ix,loop=loop,n=len(g),phase_p90_deg=g.phase_abs_deg.quantile(.9),delay_p90_ns=g.delay_abs_ns.quantile(.9),pass_fraction=g.gate.mean()))
 pd.DataFrame(cs).to_csv(R/'results/closure_summary.csv',index=False)
 # Office covariance transfer is assessed without fitting each evaluation state to its own human-labelled template.
 office=[]
 for s in STATES:
  a=np.load(R/'packets'/f'office_R1_{s}.npz');b=np.load(R/'packets'/f'office_R2_{s}.npz')
  for method in ['A','B','C','D']:office.append(dict(state=s,method=method,R1_R2_covariance_distance=distance(a[f'cov_{method}'],b[f'cov_{method}'])))
 pd.DataFrame(office).to_csv(R/'results/office_covariance_transfer.csv',index=False)
 # No coordinates are consulted before spectral extraction and its outputs are saved.
 assert (R/'results/music_sensitivity.csv').exists();gt=[];curv=[];c=299792458.;fc=5.23e9
 for y in [1,-1,2,-2]:
  p=np.array([2.01,-y]);rr=np.linalg.norm(p);theta=np.rad2deg(np.arctan2(p[1],p[0]));excess=(np.hypot(1.64,y)+rr-3.65)/c*1e9
  gt.append(dict(label=f'old{y:+d}',X_m=1.64,Y_m=y,rxlocal_x_m=p[0],rxlocal_y_m=p[1],theory_angle_deg=theta,theory_excess_delay_ns=excess,rx_range_m=rr,role='evaluation_geometry_only'))
  for s,ids in {**MAP,'Full12':list(range(1,13))}.items():
   ys=np.array([(6.5-m)*.028 for m in ids]);yc=ys.mean();offset=ys-yc;pc=p-np.array([0,yc]);rc=np.linalg.norm(pc);exact=np.sqrt(p[0]**2+(p[1]-ys)**2);linear=rc-offset*pc[1]/rc;patherror=exact-linear
   curv.append(dict(label=f'old{y:+d}',state=s,aperture_m=float(np.ptp(ys)),range_to_aperture_centre_m=rc,max_path_error_mm=max(abs(patherror))*1e3,max_phase_error_deg=max(abs(patherror))*360*fc/c,max_full_frequency_phase_error_deg=max(abs(patherror))*360*(fc+18.125e6)/c,farfield_screen_m=2*np.ptp(ys)**2/(c/fc),model='tangent plane at arithmetic aperture centre; geometric sensitivity, not estimated range'))
 pd.DataFrame(gt).to_csv(R/'results/evaluation_geometry.csv',index=False);pd.DataFrame(curv).to_csv(R/'results/corrected_curvature.csv',index=False)
 m=pd.read_csv(R/'input_manifest.csv');old=pd.read_csv(ROOT/'output/static_analysis_20261018/geometry_update_20261019/state_results.csv');outputs=[]
 for r in m.itertuples():
  oldr=old[old['index']==r.index];computed=oldr[oldr.status=='computed'];missing=sorted(set(STATES)-set(computed.state))
  outputs.append(dict(index=r.index,label=r.label,batch=r.batch,run=r.run,role=r.role_current,bounded_stage=r.bounded_stage,source=r.path,raw_sha256=r.sha256,requested_seconds=r.duration_s,complete_window=r.complete_window,prior_computed_states=len(computed),missing_states=';'.join(missing),current_bounded_states=6 if r.bounded_stage!='not_scheduled' else 0,human_output_valid=False,angle_estimate_deg=np.nan,human_AoA_error_deg=np.nan,position_error_m=np.nan,reason='fixed_response_unvalidated;human_path_unassigned;physical_delay_unreferenced'+(';short_capture' if not r.complete_window else ''),statistical_interpretation='pre-inference eligibility blocked; not an empirical detector miss'))
 out=pd.DataFrame(outputs);out.to_csv(R/'results/run_outputs_and_rejections.csv',index=False)
 metrics=[]
 for role,g in out.groupby('role'):
  metrics.append(dict(role=role,inventory_runs=len(g),bounded_runs=int((g.bounded_stage!='not_scheduled').sum()),validated_human_outputs=0,release_coverage=0.0,conditional_MAE_deg=None,point_macro_MAE_deg=None,median_error_deg=None,p90_error_deg=None,max_error_deg=None,valid_and_within8_fraction=0.0,points_with_valid_output=0,physically_evaluated_detector_trials=0,target_achieved=False))
 dump(R/'results/physical_evaluation.json',metrics)
 # Plots keep eligibility and estimation separate: there is no fabricated error CDF.
 fig,ax=plt.subplots(figsize=(7.5,4.6));ax.plot([0,3.65],[0,0],color='.55');ax.scatter([0,3.65],[0,0],marker='D',s=65,color='#153958');ax.text(0,.18,'TX (0,0)');ax.text(3.65,.18,'RX (3.65,0)',ha='right')
 for g in gt:ax.scatter(g['X_m'],g['Y_m'],c='#156fb5');ax.annotate(f"({g['X_m']:.2f}, {g['Y_m']:+.0f})\nlocal angle {g['theory_angle_deg']:+.2f}°",(g['X_m'],g['Y_m']),xytext=(12,0),textcoords='offset points',va='center')
 ax.annotate('',xy=(2.8,0),xytext=(3.65,0),arrowprops={'arrowstyle':'->'});ax.text(3.12,-.23,'local +x',ha='center',fontsize=8);ax.annotate('',xy=(3.65,-.7),xytext=(3.65,0),arrowprops={'arrowstyle':'->'});ax.text(3.78,-.55,'local +y',fontsize=8);ax.set(xlim=(-.25,4.8),ylim=(-2.55,2.55),xlabel='Global X (m)',ylabel='Global Y (m)',title='Measured positions and theoretical evaluation angles');ax.set_aspect('equal');ax.grid(alpha=.2);fig.tight_layout();save(fig,'corrected_geometry')
 fig,axs=plt.subplots(1,2,figsize=(9.5,3.5));ps=pd.DataFrame(summary)
 for ax,var,lab in zip(axs,['delay','phase'],['Absolute residual delay parameter (ns)','Absolute residual phase (deg)']):
  ev=ps[ps.stage=='evaluation'];xx=np.arange(len(ev));raw=ev[f'raw_abs_{var}_median_'+('ns' if var=='delay' else 'deg')];cor=ev[f'corrected_abs_{var}_median_'+('ns' if var=='delay' else 'deg')];ax.bar(xx-.17,raw,.34,label='Before');ax.bar(xx+.17,cor,.34,label='After');ax.set_xticks(xx,[f'{r.band} {r.method}' for r in ev.itertuples()]);ax.set_ylabel(lab);ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.legend()
 fig.suptitle('Evaluation: held-out entries, accepted corrections only (medians)');fig.tight_layout();save(fig,'reference_prediction')
 fig,ax=plt.subplots(figsize=(7.5,3.6));office=pd.DataFrame(office)
 for method in ['A','B','C','D']:ax.plot(STATES,office[office.method==method].R1_R2_covariance_distance,marker='o',label=method)
 ax.set(ylabel='Symmetric relative covariance distance',title='Office R1–R2: per-state reference transfer');ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();save(fig,'office_transfer')
 fig,axs=plt.subplots(1,2,figsize=(10,3.6));gs=pd.DataFrame(gr)
 for cohort,color in [('primary','#156fb5'),('Office','#d17b1c')]:
  g=gs[gs.cohort==cohort];axs[0].plot(g.edge,g.phase_p90_deg,'o-',label=cohort,color=color);axs[1].plot(g.edge,g.delay_p90_ns,'o-',label=cohort,color=color)
 for ax,limit,yl in zip(axs,[15,5],['P90 phase prediction error (deg)','P90 delay prediction error (ns)']):ax.axhline(limit,color='k',ls='--',lw=.8,label='engineering gate');ax.set(xlabel='Overlap edge ID',ylabel=yl);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Frozen development graph templates → bounded evaluation');fig.tight_layout();save(fig,'graph_prediction')
 fig,ax=plt.subplots(figsize=(8.5,3));colors=['#c17a24' if r.bounded_stage!='not_scheduled' else '#b9bec4' for r in out.itertuples()];ax.scatter(out['index'],np.zeros(len(out)),c=colors,marker='x',s=45)
 for i in [2,5]:ax.annotate('short',xy=(i,0),xytext=(i,.35),ha='center',arrowprops={'arrowstyle':'-','color':'.5'})
 ax.set(xlabel='Primary record index (all 39 retained)',yticks=[0],yticklabels=['No validated\nhuman output'],ylim=(-.6,.8),title='Physical-output eligibility audit; no angle-error distribution available');ax.text(.5,.9,'Orange: bounded recomputation; grey: inventory-only this stage',transform=ax.transAxes,ha='center',fontsize=8);fig.tight_layout();save(fig,'output_eligibility')
 fig,ax=plt.subplots(figsize=(7.5,3.4));cv=pd.DataFrame(curv)
 for label,g in cv.groupby('label'):ax.plot(g.state,g.max_phase_error_deg,'o-',label=label.replace('old',''))
 ax.set(ylabel='Max geometric phase approximation error (deg)',title='Plane-wave versus spherical receive path at known positions');ax.legend(ncol=4,title='Position Y (m)');ax.grid(alpha=.2);fig.tight_layout();save(fig,'curvature_corrected')
 totals=dict(primary_inventory=39,development_inventory=12,evaluation_inventory=27,bounded_primary_runs=8,bounded_Office_runs=2,unique_state_records=60,method_state_checks=len(states),common_model_pass=int(states.common_model_pass.sum()),prediction_records=len(pred),frozen_config_sha256=sha(R/'config.json'),independent_hardware_calibration=False,human_path_identified=False,physical_AoA_metrics_available=False,coherent_full12_executed=False,position_estimation_executed=False)
 dump(R/'results/summary.json',totals);print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(gr).to_string(index=False));print(pd.DataFrame(office).groupby('method').R1_R2_covariance_distance.median());print(totals)
if __name__=='__main__':main()
