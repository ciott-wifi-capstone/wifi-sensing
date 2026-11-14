"""Change just the candidate generator to a joint G2/G3 short-baseline subspace."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import predict_branches as primary
c=primary.core


def pooled_profile(data,mapping,outers,ref,test,blocks):
    fitted=[c.fit(d[ref]["full"],np.stack([d[ref][f"Q{i}"] for i in range(1,5)])) for d in data]
    out={}
    for block in blocks:
        parts={}
        for j,(s,d,fit) in enumerate(zip(mapping,data,fitted)):
            mat,base,use,_=c.projectors("background_null",d[ref]["full"],d[test][block],fit)
            parts[j]=c.state_cost(outers[s],mat,base,use,False)
        for n,band in enumerate(primary.BANDS):
            residual=c.joint(parts,range(2),n)[2]
            out[block,band]=primary.Spectrum(residual)
    return out


def main():
    reg=json.loads((primary.ROOT/"blind_registry.json").read_text(encoding="utf8"))
    assert all(c.base.sha(primary.ROOT/name)==v for name,v in reg.items())
    outers=c.steering_outers();out=[];checks=[]
    primary_audit=pd.read_csv(primary.ROOT/"packet_allocations.csv")
    for site in primary.POINTS:
        packets,_=c.read_packets(site,c.previous.INPUTS[site])
        m,_,audit=primary.packet_moments(site,packets)
        cols=["point","arm","run","role","state","packets","selected_ordinals_sha256"]
        verify=pd.DataFrame(audit)[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(verify,primary_audit[primary_audit.point==site][cols].reset_index(drop=True))
        for arm,states in primary.ARMS.items():
            trials=[];strength=[]
            for ref,test in [(1,2),(1,3),(3,2)]:
                blocks=("full","Q1","Q2","Q3","Q4") if ref==1 else ("full",)
                data=[{r:m[r,s,j] for r in (1,2,3)} for j,s in enumerate(states)]
                views={role:primary.profiles(d,s,"background_null",outers,ref,test,blocks) for role,s,d in zip(primary.ROLES,states,data)}
                views["P"]=pooled_profile(data[:2],states[:2],outers,ref,test,blocks)
                trials.append(views)
                if ref==1:
                    beams=[c.previous.beam_contrast(d[test]["full"]-d[ref]["full"],s) for d,s in zip(data,states)]
                    strength.append(float(np.mean(beams,axis=0).max()))
            seeds,n=primary.shortlist(trials[0]["P"]["full","full"])
            ratio=strength[0]/strength[1] if strength[1]>0 else np.nan
            for margin in (0.,1.,3.):
                entry,details=primary.assess(seeds,trials,ratio,margin)
                k=dict(point=site,model="background_null",arm=arm,threshold_db=margin)
                out.append(dict(**k,total_separated=n,**entry))
                checks.extend(dict(**k,**rec) for rec in details)
    pd.DataFrame(out).to_csv(HERE/"blind_decisions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(HERE/"branch_checks.csv",index=False,encoding="utf-8-sig")
    files=["protocol.json","predict_two_short.py","blind_decisions.csv","branch_checks.csv"]
    (HERE/"blind_registry.json").write_text(json.dumps({f:c.base.sha(HERE/f) for f in files},indent=2),encoding="utf8")
    print("Two-short candidates and votes frozen. Packet identities verified against primary.")


if __name__=="__main__":
    with threadpool_limits(limits=1):main()

