from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
D=3.65
C0=299792458.0
FC=5.23e9
STATES={'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]}
YLOCAL={m:(6.5-m)*.028 for m in range(1,13)}

def write_csv(name, rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    rows=[]
    coords={}
    for label,theta in [('T0',0),('Tn20',-20),('Tp20',20),('Tn45',-45),('Tp45',45)]:
        x=D-D*math.cos(math.radians(theta));y=-D*math.sin(math.radians(theta))
        coords[label]=(x,y,theta)
        rows.append(dict(position_id=label,nominal_array_center_deg=theta,planned_X_m=x,planned_Y_m=y,planned_distance_to_RX_m=D,source_height='match measured RX height',role='calibration_and_return_checks' if theta==0 else 'unfitted_angle_validation',actual_X_m='',actual_Y_m='',actual_Z_m='',measured=False))
    write_csv('TX_move_positions.csv',rows)
    alt=[]
    for label,y in [('Tn_compact',2),('Tp_compact',-2)]:
        x=1.64;r=math.hypot(D-x,y);theta=math.degrees(math.atan2(-y,D-x))
        coords[label]=(x,y,theta)
        alt.append(dict(position_id=label,planned_X_m=x,planned_Y_m=y,planned_distance_to_RX_m=r,theory_array_center_deg=theta,role='alternative_to_45deg_if_room_limited_not_extra_required_point',actual_X_m='',actual_Y_m='',actual_Z_m='',measured=False))
    write_csv('compact_alternative_positions.csv',alt)
    gt=[]
    for label,(x,y,theta) in coords.items():
        for state,ants in STATES.items():
            yc=sum(YLOCAL[m] for m in ants)/3
            distances=[math.sqrt((D-x)**2+(-YLOCAL[m]-y)**2) for m in ants]
            gt.append(dict(position_id=label,state=state,physical_antennas='/'.join(map(str,ants)),theory_array_center_deg=theta,theory_subarray_center_deg=math.degrees(math.atan2(-y-yc,D-x)),d2_minus_d1_m=distances[1]-distances[0],d3_minus_d1_m=distances[2]-distances[0],geometric_pair21_phase_deg_at_fc=-360*FC*(distances[1]-distances[0])/C0,geometric_pair31_phase_deg_at_fc=-360*FC*(distances[2]-distances[0])/C0,formula_uses_planned_equal_height_geometry=True))
    write_csv('predicted_subarray_angles_and_propagation.csv',gt)
    schedule=[]
    def add(name,point,seconds,role):
        x,y,t=coords[point]
        schedule.append(dict(order=len(schedule)+1,segment_id=name,TX_position_id=point,planned_X_m=x,planned_Y_m=y,planned_duration_s=seconds,all_six_states=True,role=role,rx_epoch_id='',actual_X_m='',actual_Y_m='',actual_Z_m='',actual_start_monotonic_ns='',actual_end_monotonic_ns='',raw_file='',events_file='',sha256='',quality_pass='',notes=''))
    add('WARMUP','T0',120,'excluded_engineering_warmup_not_proof_of_stability')
    add('QC0','T0',30,'acquisition_quality_only')
    add('CAL0_R1','T0',30,'fit_relative_fixed_response')
    add('CAL0_R2','T0',30,'development_model_selection_then_freeze')
    add('CAL0_R3','T0',30,'heldout_same_angle_time_window')
    for label in ['Tn20','Tp20','Tn45','Tp45']:
        for rep in range(1,4):add(f'VAL_{label}_R{rep}',label,30,'unfitted_direction_validation')
        add(f'RETURN0_after_{label}','T0',30,'frozen_reference_return_check_no_refit')
    for rep in range(1,4):add(f'EMPTY_FINAL_R{rep}','T0',30,'final_layout_background_not_absolute_phase_truth')
    write_csv('capture_schedule_and_log_template.csv',schedule)
    thresholds=dict(proposed_not_measured=True,freeze_before_validation=True,raw_packets_per_state_per_30s_min=1000,state_visit_coverage_min=.90,minimum_retained_frequency_fraction=.80,phase_prediction_P90_deg_max=15,return_reference_circular_mean_shift_deg_max=10,per_G_AoA_MAE_deg_max=5,per_G_AoA_P90_deg_max=8,per_direction_per_G_MAE_deg_max=8,per_G_output_fraction_min=.90,dual_band_difference_deg_max=6,dual_band_consistency_fraction_min=.90,temporal_5s_windows_per_30s=6,primary_AoA_outputs_per_G=12,angle_score_rule='one frozen blind estimate per complete 30s validation window; no nearest-truth selection')
    config=dict(template_only=True,actual_geometry_requires_measurement=True,date='2026-11-05',method_name='Known-source differential propagation phase calibration',RX_global_m=[3.65,0,None],RX_local_transform='(3.65-X,-Y,Z-h_RX)',original_TX_global_m=[0,0,None],antenna_y_local_m=[YLOCAL[m] for m in range(1,13)],state_chain_order=STATES,nominal_config=dict(control_MHz=5240,center_MHz=5230,width='HT40-',target_pps=1000,settling_us=100,valid_us=25000),nominal_config_requires_header_verification=True,avoid_restart=True,source_points=rows,thresholds=thresholds,not_calibrated_by_this_method=['absolute_packet_phase','absolute_ToF_zero','CFO_estimate','automatic_cross_state_coherence','verified_human_reflection'],fitting_options=['constant_relative_phase','constant_plus_frequency_linear_phase'],selection_data=['CAL0_R1','CAL0_R2'],evaluation_data=['CAL0_R3','VAL_*','RETURN0_*'],frozen_source_peak_rule='G1-G4 one-path spatial MUSIC over full predeclared angle range; strongest peak or frozen rejection; human pipeline remains separately unvalidated',notes=['Three fixed-point windows are temporal repeats in one RX epoch, not independent hardware sessions.','If the compact option is selected, record it before capture and use its measured geometry, never relabel it as exactly 45 degrees.'])
    (ROOT/'calibration_protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    fig,ax=plt.subplots(figsize=(8.2,8.3),layout='constrained')
    angles=np.deg2rad(np.linspace(-52,52,241));ax.plot(D-D*np.cos(angles),-D*np.sin(angles),color='#c9d2db',ls='--',lw=1.2)
    ax.scatter([D]*12,[-YLOCAL[m] for m in range(1,13)],s=20,color='#17334e',zorder=5)
    ax.text(D+.13,0,'RX stays fixed\n(3.650, 0)',va='center',fontsize=10,color='#17334e')
    for row in rows:
        x,y,t=row['planned_X_m'],row['planned_Y_m'],row['nominal_array_center_deg']
        color='#2464d8' if t==0 else '#df6500'
        ax.plot([x,D],[y,0],color=color,lw=1,alpha=.35)
        ax.scatter([x],[y],marker='^',s=95,color=color,zorder=4)
        label=f"TX {t:+.0f} deg\n({x:.3f}, {y:+.3f})" if t else 'TX 0 deg: original position\n(0.000, 0.000)'
        ax.annotate(label,(x,y),xytext=(-8 if t else 9,12 if y>=0 else -35),textcoords='offset points',ha='left',fontsize=10,color=color)
    for row in alt:
        ax.scatter(row['planned_X_m'],row['planned_Y_m'],marker='s',s=65,facecolors='none',edgecolors='#16846e',zorder=4)
    ax.text(1.75,2.02,'Compact alternative\n(1.640, +2.000)',fontsize=9,color='#16846e')
    ax.text(1.75,-2.12,'Compact alternative\n(1.640, -2.000)',fontsize=9,color='#16846e')
    ax.text(2.55,-.20,'radius = 3.650 m',fontsize=9,color='.35',ha='center')
    ax.set_title('TX positions for initial relative-phase calibration\nOnly TX moves; return to the original TX position after each direction',fontsize=12)
    ax.set_xlabel('Global X (m), original TX toward RX');ax.set_ylabel('Global Y (m), same drawing convention')
    ax.set_xlim(-.35,4.7);ax.set_ylim(-3.4,3.2);ax.set_aspect('equal');ax.grid(alpha=.18)
    fig.savefig(ROOT/'TX_calibration_positions.png',dpi=200);fig.savefig(ROOT/'TX_calibration_positions.pdf');plt.close(fig)
    assert len(schedule)==24
    assert sum(r['planned_duration_s'] for r in schedule if r['segment_id']!='WARMUP')==690
    for row in rows:
        assert abs(math.hypot(D-row['planned_X_m'],row['planned_Y_m'])-D)<1e-10
        assert abs(math.degrees(math.atan2(-row['planned_Y_m'],D-row['planned_X_m']))-row['nominal_array_center_deg'])<1e-10
    print(json.dumps({'positions':rows,'G_angles_at_T0':{r['state']:r['theory_subarray_center_deg'] for r in gt if r['position_id']=='T0'},'planned_nonwarmup_s':690,'planned_withwarmup_s':810,'max_center_to_edge_broadside_phase_deg':360*FC*(math.hypot(D,.154)-math.hypot(D,.014))/C0},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
