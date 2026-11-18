"""Additional actual-output stability checks, clearly distinct from candidate crossfit.
Reconstruct outputs from saved parameters; does not re-fit correction or alter gates.
"""
import json,importlib.util
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core import *
spec=importlib.util.spec_from_file_location('decoder',R/'provenance/analyze.py')
dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)
def main():
    for p in sorted((R/'results').glob('run_*.json')):
        d=json.loads(p.read_text());out=R/'results'/f'actual_{d["index"]:03d}.json'
        if out.exists():continue
        x,host,nic=dec.decode(Path(d['path']));rows=[]
        for s in d['states']:
            if s.get('status')!='computed':continue
            key=f'{d["index"]:03d}_{s["state"]}';v=np.load(R/'packets'/f'{key}.npz');a=unit(x[v['packet_id']].astype(complex))
            b=apply(a,v['B_tau_ns'],v['B_psi_rad']);c=a.copy();mask=v['C_applied'];c[mask]=apply(a[mask],v['C_tau_ns'][mask],v['C_psi_rad'][mask])
            for method,z in [('A',a),('B',b),('C',c)]:
                ref=z[:CFG['prefix_packets']].mean(0)
                # Mean(A) can approach zero due to CPE cancellation; first-packet fallback recorded.
                fallback=np.linalg.norm(ref)<.1
                if fallback:ref=z[0]
                ds=[];ps=[];co=[]
                for q in range(CFG['prefix_packets'],len(z),8):
                    t,ph,cor,_=fit(z[q],ref);ds.append(t);ps.append(score(z[q],ref,np.ones((3,K)))[2]);co.append(cor)
                zref=np.sum(z*ref.conj()[None],axis=(1,2))/max(np.linalg.norm(ref),1e-15)
                rows.append(dict(index=d['index'],group=d['group'],role=d['role'],state=s['state'],method=method,
                    actual_residual_delay_iqr_ns=iqr(ds),actual_phase_sd_deg=circ_sd(ps)*180/np.pi,
                    actual_mean_vector_norm=float(np.linalg.norm(z.mean(0))),prefix_mean_norm=float(np.linalg.norm(ref)),
                    reference_fallback_first_packet=bool(fallback),actual_reference_correlation=float(np.median(co))))
        out.write_text(json.dumps(rows,indent=2),encoding='utf-8');print('Actual outputs',d['index'],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
