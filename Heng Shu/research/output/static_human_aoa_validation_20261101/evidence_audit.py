"""Audit numeric claims and maintain current evidence/status records without modifying raw data."""
import pandas as pd
import re,shutil,struct
from common import *
TH=ROOT/'output/thesis_algorithm_draft_20261019';E=TH/'evidence';old=E/'primary_four_point'
def main():
 facts={};s=pd.read_csv(R/'results/preprocessing_summary.csv');m=pd.read_csv(R/'input_manifest.csv')
 assert len(m)==39 and m.sha256.nunique()==39 and m.complete_window.sum()==37
 facts['inventory']={'runs':39,'unique_hashes':39,'complete':37,'incomplete_indices':m.loc[~m.complete_window,'index'].tolist()}
 a=pd.read_csv(old/'model_order_spread.csv');facts['legacy_q1_to4']={'states':len(a),'median_angle_span_deg':a.angle_span_deg.median(),'fraction_over15':(a.angle_span_deg>15).mean()}
 a=pd.read_csv(old/'run_state_peaks.csv');facts['legacy_G1_peaks']=a[a.state=='G1'].groupby('group').median(numeric_only=True).to_dict()
 a=pd.read_csv(old/'background_existing.csv');facts['legacy_background_medians']=a[a.role!='background_fit'].groupby('group').whitened_shape.median().to_dict()
 facts['bounded_preprocessing']=s.to_dict('records');facts['bounded_summary']=json.loads((R/'results/summary.json').read_text(encoding='utf-8'))
 # Independently read the first frame of every primary source; entire bounded files were asserted by the decoder.
 headers=[]
 for r in m.itertuples():
  with Path(r.path).open('rb') as f:size=struct.unpack('<I',f.read(4))[0];f.seek(0);raw=f.read(size+4)
  size,magic,ver,ns=struct.unpack_from('<IIHB',raw,0);assert magic==0x20150315 and ver==1;pos=11;h={}
  for _ in range(ns):
   length,n=struct.unpack_from('<IB',raw,pos);name=raw[pos+5:pos+5+n].rstrip(b'\0').decode();b=pos+7+n
   if name=='RxSBasic':h.update(zip(['csi_centre_mhz','control_mhz','bandwidth_mhz'],struct.unpack_from('<hhh',raw,b+18)))
   if name=='CSI':h.update(tone_groups=struct.unpack_from('<H',raw,b+25)[0],ntx=raw[b+27],nrx=raw[b+28],permutation_byte=raw[b+32])
   pos+=4+length
  assert (h['csi_centre_mhz'],h['bandwidth_mhz'],h['tone_groups'],h['ntx'],h['nrx'])==(5230,40,30,1,3)
  headers.append(dict(index=r.index,raw_sha256=r.sha256,first_frame_verified=True,**h))
 pd.DataFrame(headers).to_csv(R/'results/header_audit.csv',index=False)
 # Current checks, excluding inherited output filenames and review scripts.
 params=pd.read_csv(R/'results/packet_parameters_all.csv',dtype={'index':str});pred=pd.read_csv(R/'results/prediction_records.csv',dtype={'index':str})
 assert len(pred)==19008 and not pred[['source','source_sha256','reference','packet_index','state','model_gate','boundary','applied']].isna().any().any()
 assert not params.loc[~params.model_gate,'applied'].any()
 assert not pred.loc[~pred.model_gate,'applied'].any()
 assert not pred.loc[pred.boundary,'applied'].any()
 assert sha(R/'config.json')==(R/'config.sha256').read_text().strip()
 facts['prediction_traceability']='19008 records complete; failed model/boundary never applied'
 facts['music_phase_only_max_difference']=float(pd.read_csv(R/'results/music_invariants.csv').A_B_spectrum_max_difference.max())
 facts['gate_failures']=pd.read_csv(R/'results/state_summary.csv').query('common_model_pass == False')[['index','state','chain_delay_p90_ns','chain_phase_p90_deg']].to_dict('records')
 dump(R/'results/numeric_claim_audit.json',facts)
 claims=[]
 def claim(name,file,script,scope,status,limit):claims.append(dict(claim=name,result_file='../../static_human_aoa_validation_20261101/'+file,script=script,config_sha256=sha(R/'config.json'),scope=scope,status=status,limitation=limit))
 claim('39 sources, 37 complete; indices 2/5 retained','input_manifest.csv','audit_inputs.py; evidence_audit.py','all 39 primary sources','A','prior diagnostics all six states; only eight runs recomputed')
 claim('HT40 common gate 21/24 dev and 22/24 evaluation','results/preprocessing_summary.csv','preprocess.py; summarize.py','eight primary runs; 768 packets/state','A/B','relative model gate, not calibrated human direction')
 claim('Evaluation held-entry residuals: HT40 8.168 to .654 ns, Office D 8.747 to .197 ns','results/prediction_records.csv','preprocess.py; summarize.py','accepted rotating-mask predictions only','B','dependent masks and packets; not physical ToF error')
 claim('Office median covariance transfer A .004153, C .063058, D .003691','results/office_covariance_transfer.csv','summarize.py','six R1/R2 state pairs','B','same reported human scene; no numeric human geometry')
 claim('Changed weak-component controls fail 5 percent limit','results/weak_component_controls.csv','check_models.py','synthetic two-path configurations and amplitudes','A','component-relative norm; no inference of human detection')
 claim('Complete overlap graph and leaf limitations','results/graph_evaluation_held_edges.csv','state_sync.py; check_models.py','66 cycles/primary run; 199/Office run','A/B','redundant constraints do not calibrate relative chains')
 claim('Frozen primary overlap template fails; Office template transfers','results/graph_prediction_summary.csv','review_development.py; state_sync.py; summarize.py','four primary evaluation runs and Office R2','B','cannot uniquely attribute difference to hardware or channel')
 claim('48 bounded HT40 q2/3/4 spans: median 3 deg, 6 above15','results/music_sensitivity.csv','music_baseline.py','eight runs x six states, treatment C','B','unassigned spectral parameters; not human error')
 claim('Unassigned local maxima and angular fusion','results/unassigned_peaks.csv; results/noncoherent_fusion.csv','music_baseline.py','all bounded runs','B','no aligned 2D-delay summation or coherent12 gain')
 claim('Corrected full aperture maximum geometric phase error 13.72 to27.40 deg','results/corrected_curvature.csv','summarize.py','four known positions, tangent-plane approximation','A','geometry-only; not observed AoA error')
 claim('No validated human output; physical errors NA','results/run_outputs_and_rejections.csv; results/physical_evaluation.json','summarize.py','39-run eligibility ledger; evaluation inventory27/bounded4','C','zero release coverage is not empirical detector miss rate')
 claim('No additional calibration or paired background available','PROTOCOL.md','user confirmation; observation-model audit','current supplied records','C','new-session calibration cannot retrospectively calibrate old data')
 pd.DataFrame(claims).to_csv(E/'claim_evidence_current.csv',index=False)
 legacy=pd.read_csv(E/'claim_evidence.csv');legacy['version']='legacy retained; see current ledger and corrections'
 if not (E/'claim_evidence_legacy.csv').exists():legacy.to_csv(E/'claim_evidence_legacy.csv',index=False)
 pd.DataFrame(claims).assign(version='2026-11-01').to_csv(E/'claim_evidence.csv',index=False)
 components=[('source/roles/geometry','A','hashes, corrected mapping, preserved incomplete records','input_manifest.csv; results/header_audit.csv'),('common reference alignment','A/B','numerical checks and bounded predictive checks; no hardware calibration','results/numerical_controls.csv; results/prediction_records.csv'),('overlap graph','B','complete graph, internal closure and frozen transfer; leaf caveats','results/graph_frozen_predictions.csv; results/graph_evaluation_held_edges.csv'),('MUSIC/local maxima','B','exact-tone smoothing; q sensitivity; unassigned paths','results/music_sensitivity.csv; results/unassigned_peaks.csv'),('noncoherent angular fusion','B','quality weights; delay maximized separately; G states only','results/noncoherent_fusion.csv'),('fixed hardware response','C','no independently validated calibration','PROTOCOL.md'),('empty background','B/C','legacy covariance diagnostics; dates corrected, physical transfer blocked','../thesis_algorithm_draft_20261019/evidence/primary_four_point/background_existing.csv'),('human/direct-path identity','C','not established; no forced estimates','results/run_outputs_and_rejections.csv'),('coherent12/spherical positioning','C','geometry sensitivity only; physical branch not executed','results/corrected_curvature.csv')]
 pd.DataFrame(components,columns=['component','status','scope','evidence']).to_csv(E/'implementation_status.csv',index=False)
 source_by_figure={'algorithm_flow':'review_figures.py / config.json + implementation gates','state_graph':'review_figures.py / confirmed state_map','primary_reference':'build_four_point_assets.py / primary_four_point/reference_per_run.csv','primary_background':'build_four_point_assets.py / primary_four_point/background_existing.csv','primary_music':'build_four_point_assets.py / preserved spectra.npz and primary manifest','primary_curvature':'build_four_point_assets.py / corrected theory and phase-span metric','corrected_geometry':'summarize.py / evaluation_geometry.csv','reference_prediction':'summarize.py / prediction_records.csv','office_transfer':'summarize.py / office_covariance_transfer.csv','graph_prediction':'summarize.py / graph_frozen_predictions.csv','graph_closure_new':'review_figures.py / graph_evaluation_closure.csv','music_3':'music_baseline.py / packets/3_*.npz','fusion_3':'music_baseline.py / noncoherent_fusion.csv','output_eligibility':'summarize.py / run_outputs_and_rejections.csv','curvature_corrected':'summarize.py / corrected_curvature.csv'}
 pd.DataFrame([dict(figure=n+'.pdf',source=src,sha256=sha(TH/'figures'/f'{n}.pdf'),status='B diagnostic' if n not in ['algorithm_flow','state_graph','corrected_geometry','primary_curvature','curvature_corrected'] else 'A geometry/design') for n,src in source_by_figure.items()]).to_csv(E/'figure_sources.csv',index=False)
 refs=[('wang2023','local published PDF text, pp3724-3742; Sections3-6','10.1109/TMC.2022.3141115','passive moving-person tracking; DFS-based separation; hardware calibration','static-human adaptation is ours'),('ratnam2024','user PDF, pp10820-10833, Eq2/27; arxiv2307.12126v2','10.1109/TWC.2024.3376332','gain/timing/CPE model; conditional reference ML; prior hardware distortion calibration','joint chain gate, actual sparse tones and masks are ours'),('wical','local published PDF pp3752-3765; alignment pp3755-3756; arxiv2505.21408','10.1109/JSAC.2025.3584540','switched URAs and three-stage alignment','our linear graph is different; no source performance imported'),('spotfi','author-hosted paper https://web.stanford.edu/~skatti/pubs/sigcomm15-spotfi.pdf, Secs3.1-3.2; DOI confirmed in https://dineshb.ucsd.edu/files/papers/SpotFi.pdf first-page footer','10.1145/2785956.2787487','joint angle/delay, smoothing and transmitting-device direct-path selection','unassigned reflected paths here; no accuracy imported'),('csitool','https://dhalperi.github.io/linux-80211n-csitool/index.html','','Intel5300 thirty groups signed8 complex','format reference'),('reader','https://github.com/dhalperi/linux-80211n-csitool-supplementary/tree/master/matlab','','binary decoding/permutation reference','does not verify physical wiring'),('picoscenes','https://github.com/wifisensing/RXS-Parsing-Core','','RxSBasic/CSI segment format','narrow decoder asserts supported observed version')]
 pd.DataFrame(refs,columns=['key','verified_source','doi','supported_claim','adaptation_boundary']).assign(checked='2026-11-01').to_csv(E/'reference_audit.csv',index=False)
 for p in [R/'results/numeric_claim_audit.json',R/'results/summary.json',R/'results/physical_evaluation.json']:shutil.copy2(p,E/p.name)
 (E/'corrections_20261101.md').write_text('''# Corrections and scope changes\n\n- Empty E1/E3 were confirmed continuous in one hardware session; filename dates are unreliable. E1/E2 and E3/E4 are duration-matched comparisons only. No reboot is inferred from directory dates.\n- The primary occupied and confirmed empty acquisitions are different sessions. Empty is an absence label, not phase/angle/delay truth.\n- Complete overlap graph now has eight edges. Earlier reduced graph and all archived outputs remain in the pre-edit backups.\n- Current noncoherent fusion maximizes each delay gauge separately before quality-weighted angular addition. It is not 12-element coherent MUSIC.\n- New 768-packet prefix checks are not pooled with previous dense-prefix all-run gate statistics.\n- New maximum tangent-plane deviation and retained legacy phase-span metric are explicitly distinguished.\n- Thesis scope expanded from algorithm-only to complete supervisor-review chapters. Physical inference remains unvalidated.\n''',encoding='utf-8')
 print('Numeric, traceability, status and citation records updated. No physical accuracy claim released.')
if __name__=='__main__':main()
