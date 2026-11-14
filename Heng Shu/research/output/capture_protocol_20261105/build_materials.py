from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent;O=R.parent;C=299792458.;D=3.65
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tab(df):
    v=df.fillna('NA').astype(str)
    return '| '+' | '.join(v.columns)+' |\n| '+' | '.join(['---']*len(v.columns))+' |\n'+'\n'.join('| '+' | '.join(a)+' |' for a in v.values)
def main():
    source=O/'four_point_blind_peak_truth_comparison_20261104'
    for f,h in json.loads((source/'blind_predictions_frozen.json').read_text()).items():assert sha(source/f)==h
    allc=pd.read_csv(source/'all_candidates_truth_annotations.csv')
    near=allc[allc.absolute_difference_deg<=9].copy()
    near['interpretation']='retrospective proximity only; not a human-path identification'
    near.to_csv(R/'all_near_geometry_candidates_within9deg.csv',index=False)
    mf=pd.read_csv(O/'static_human_aoa_validation_20261101/input_manifest.csv')
    ix=[0,1,2,12,13,14,21,22,23,30,31,32]
    mf[mf['index'].isin(ix)][['index','label','batch','run','complete_window','path','sha256']].to_csv(R/'current_12_record_sources.csv',index=False)
    patterns=[(0,'C1',[-33,-36,-36]),(0,'G3',[-33,-30,-30]),(12,'C1',[18,18,18]),(12,'C2',[24,24,24]),(12,'G3',[36,36,36]),(21,'C2',[-36,-39,-36]),(30,'G1',[51,48,42]),(30,'G3',[48,51,45])]
    labels={0:'+1 m',12:'-1 m',21:'+2 m',30:'-2 m'}
    rows=[]
    for first,s,angles in patterns:
        q=allc[(allc.reference=='E1')&(allc.batch_first==first)&(allc.state==s)]
        # G1 strongest peak +51 on record30 exists in frozen candidate set; all selected rows audited.
        selected=[]
        for k,angle in enumerate(angles):
            h=q[(q['index']==first+k)&(q.angle_parameter_deg==angle)]
            assert len(h)==1,(first,s,k,angle)
            selected.append(h.iloc[0])
        errors=[a.absolute_difference_deg for a in selected]
        rows.append(dict(point=labels[first],state=s,theory_deg=selected[0].theory_subarray_deg,
            angles_R1_R2_R3='/'.join(f'{a:g}' for a in angles),absolute_differences_R1_R2_R3='/'.join(f'{a:.2f}' for a in errors),
            descriptive_mean_difference_deg=np.mean(errors),stable_R1_R2_R3='/'.join('yes' if a.stable else 'no' for a in selected),
            all_rounds_complete=all(a.complete_window for a in selected),human_path_verified=False))
    match=pd.DataFrame(rows);match.to_csv(R/'current_promising_and_failed_branches.csv',index=False)
    # Proposed HUMAN positions; do not use these as a restricted estimator search grid.
    coords=[('P1',1.64,1.,'core'),('P2',1.64,-1.,'core'),('P3',1.64,2.,'core'),('P4',1.64,-2.,'core'),
            ('P5',2.40,1.,'core_same_bearing_pair'),('P6',1.15,2.,'core_same_bearing_pair'),
            ('P7',2.40,-1.,'optional_mirror_pair'),('P8',1.15,-2.,'optional_mirror_pair')]
    pts=[]
    for pid,x,y,role in coords:
        rr=np.hypot(D-x,y);rt=np.hypot(x,y);ex=rr+rt-D
        pts.append(dict(point_id=pid,x_global_m=x,y_global_m=y,rx_distance_m=rr,
            theory_array_center_deg=np.degrees(np.arctan2(-y,D-x)),theory_excess_path_m=ex,theory_excess_ToF_ns=ex/C*1e9,role=role,
            measured=False,body_heading_reference='baseline midpoint (1.825,0)',body_coordinates_reference='torso-axis projection on floor'))
    points=pd.DataFrame(pts);points.to_csv(R/'proposed_human_positions.csv',index=False)
    p=points.set_index('point_id');assert np.isclose(p.loc['P5','theory_array_center_deg'],p.loc['P6','theory_array_center_deg'])
    assert not np.isclose(p.loc['P5','theory_excess_ToF_ns'],p.loc['P6','theory_excess_ToF_ns'])
    cal=[]
    for theta in [0.,-20.,20.,-40.,40.]:
        radius=4.;x=D-radius*np.cos(np.deg2rad(theta));y=-radius*np.sin(np.deg2rad(theta))
        cal.append(dict(local_angle_deg=theta,source_radius_m=radius,source_x_global_m=x,source_y_global_m=y,
            role='fit_only_at_0deg' if theta==0 else 'unfitted_source_validation',subject_absent=True,
            requires_clear_space=True,source_height_m='match measured RX phase-centre height'))
    pd.DataFrame(cal).to_csv(R/'proposed_active_source_calibration_positions.csv',index=False)
    # Predetermined acquisition order; point labels do not influence estimator peak selection.
    orders=[['P1','P4','P5','P2','P6','P3'],['P6','P2','P3','P5','P1','P4'],['P3','P5','P4','P1','P2','P6']]
    rows=[];seq=0
    stages=[('EMPTY_PRE',0,30),('ENTER_AND_SETTLE',30,45),('HUMAN_STATIC',45,75),('EXIT_AND_SETTLE',75,90),('EMPTY_POST',90,120)]
    for rep,order in enumerate(orders,1):
        for pid in order:
            seq+=1
            for phase,start,end in stages:
                rows.append(dict(sequence=seq,repeat=rep,point_id=pid,phase=phase,planned_start_within_block_s=start,planned_end_within_block_s=end,
                    role='evaluation_after_freeze',x_global_m=p.loc[pid,'x_global_m'],y_global_m=p.loc[pid,'y_global_m'],
                    session_uuid='',rx_epoch_id='',actual_start_monotonic_ns='',actual_end_monotonic_ns='',raw_file='',raw_sha256='',quality_status='',notes=''))
    pd.DataFrame(rows).to_csv(R/'acquisition_log_template.csv',index=False)
    meta=dict(template_only=True,all_config_values_require_confirmation=True,actual_geometry_verified=False,session_uuid='',rx_epoch_id='',local_datetime_with_timezone='',host_clock_verified=False,boot_id='',phy='',ifindex='',wdev='',driver_reload_counter='',
        control_frequency_MHz=5240,center_frequency_MHz=5230,bandwidth='HT40-',target_pps=1000,mcs='',tx_power_setting='',tx_MAC='',rx_MAC='',tx_stream_count='',
        tx_xyz_m=[0,0,None],rx_center_xyz_m=[3.65,0,None],antenna_spacing_m=.028,array_aperture_m=.308,
        local_transform='(3.65-X,-Y)',states={'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]},
        antenna_y_local_m=[(6.5-m)*.028 for m in range(1,13)],switch_settling_us=100,switch_valid_us=25000,
        empirical_alignment_guard_ms=None,raw_baseline_preserved=None,room_sketch_file='',equipment_photos=[],script_hashes={},calibration_files=[],quality_rejections=[])
    (R/'session_manifest_template.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    # A geometrically scaled plan drawing; no wall outline is invented.
    fig,ax=plt.subplots(figsize=(9.2,8.3),layout='constrained')
    ax.plot([0,D],[0,0],color='.45',lw=2);ax.scatter([0,D],[0,0],marker='D',s=110,color='#19304d',zorder=5)
    ax.text(-.1,-.20,'TX (0,0)',ha='center',fontsize=11);ax.text(D,-.20,'RX (3.65,0)',ha='center',fontsize=11)
    ym=(6.5-np.arange(1,13))*.028;ax.scatter(np.full(12,D),-ym,s=11,color='#19304d')
    for row in points.itertuples():
        if row.point_id in ['P1','P2','P3','P4']:marker='o';color='#2463eb';face=color
        elif row.point_id in ['P5','P6']:marker='^';color='#df6500';face=color
        else:marker='s';color='#178669';face='none'
        ax.scatter(row.x_global_m,row.y_global_m,marker=marker,s=100,edgecolors=color,facecolors=face,zorder=4)
        left=row.point_id in ['P1','P2','P6','P8']
        ax.annotate(f'{row.point_id} ({row.x_global_m:.2f}, {row.y_global_m:+.2f})',(row.x_global_m,row.y_global_m),xytext=(-9 if left else 9,8 if row.y_global_m>0 else -18),textcoords='offset points',ha='right' if left else 'left',color=color,fontsize=10)
    ax.plot([D,p.loc['P6','x_global_m']],[0,p.loc['P6','y_global_m']],'--',color='#df6500',alpha=.6)
    ax.plot([D,p.loc['P8','x_global_m']],[0,p.loc['P8','y_global_m']],'--',color='#178669',alpha=.5)
    ax.text(.08,2.40,'P5/P6: same bearing, different bistatic path length',fontsize=10,color='#ac5000')
    ax.scatter([D/2],[0],marker='*',s=110,color='#b90075');ax.text(D/2,.20,'Face midpoint (1.825,0)',ha='center',color='#b90075',fontsize=10)
    ax.axvline(1.64,color='#2463eb',ls=':',lw=.8,alpha=.4)
    ax.set_xlim(-.35,4.35);ax.set_ylim(-2.65,2.7);ax.set_aspect('equal');ax.grid(alpha=.18)
    ax.set_xlabel('Global x (m), TX to RX');ax.set_ylabel('Global y (m)')
    ax.set_title('Proposed static-human positions\nBlue: original four | Orange: range-disambiguation pair | Green: optional mirrors',fontsize=12)
    fig.savefig(R/'proposed_positions.png',dpi=190);fig.savefig(R/'proposed_positions.pdf');plt.close(fig)
    sources=[source/'all_candidates_truth_annotations.csv',source/'blind_predictions_frozen.json',O/'static_human_aoa_validation_20261101/input_manifest.csv',
             O/'g_subarray_precalibration_20261105/all_branch_truth_comparison.csv',O/'e1_e3_mapping_phase_timing_audit_20261104/driver_reload_evidence.csv']
    (R/'source_hashes.json').write_text(json.dumps({str(x.resolve()):sha(x) for x in sources},indent=2),encoding='utf8')
    theory=dict(wavelength_m=C/5.23e9,full_array_farfield_rule_m=2*.308**2/(C/5.23e9),nominal_40MHz_delay_resolution_ns=1/40e6*1e9,
                sampled_frequency_span_MHz=36.25,approx_full_state_cycle_s=6*(100+25000)*1e-6,
                cycles_per_30s=30/(6*(100+25000)*1e-6),theory_not_measured=True)
    (R/'geometry_checks.json').write_text(json.dumps(theory,indent=2),encoding='utf8')
    print(points[['point_id','x_global_m','y_global_m','theory_array_center_deg','rx_distance_m','theory_excess_ToF_ns','role']].round(3).to_string(index=False))
    print(match.to_string(index=False));print(json.dumps(theory,indent=2))
if __name__=='__main__':main()
