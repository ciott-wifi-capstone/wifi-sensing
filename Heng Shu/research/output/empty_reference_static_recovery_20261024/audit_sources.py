"""Read metadata and headers; decode only the six confirmed empty runs. No fitting."""
import re,struct,collections
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from common import *
def headers(path):
    raw=Path(path).read_bytes();pos=0;out=[];payload=[]
    while pos<len(raw):
        size,magic,ver,ns=struct.unpack_from('<IIHB',raw,pos);assert magic==0x20150315 and ver==1
        end=pos+4+size;assert end<=len(raw);s=pos+11;row={}
        for _ in range(ns):
            length,n=struct.unpack_from('<IB',raw,s);name=raw[s+5:s+5+n].rstrip(b'\0').decode();v=struct.unpack_from('<H',raw,s+5+n)[0];b=s+7+n
            if name=='RxSBasic':
                assert v==4;row.update(nic=struct.unpack_from('<Q',raw,b+2)[0],host=struct.unpack_from('<Q',raw,b+10)[0])
                row['header_ht40_ok']=struct.unpack_from('<hhh',raw,b+18)==(5230,5240,40)
            if name=='CSI':
                assert v==3;row['ant']=raw[b+32];row['subcarrier_offset']=struct.unpack_from('<h',raw,b+33)[0]
                payload.append(raw[b+39:b+231])
            s+=length+4
        out.append(row);pos=end
    assert pos==len(raw);return pd.DataFrame(out),payload
def independent_frame(payload,ant):
    # Halperin's destination permutation semantics, independent scalar bit decoder.
    order=[(ant>>(2*m))&3 for m in range(3)];assert sorted(order)==[0,1,2]
    val=int.from_bytes(payload,'little');out=np.empty((3,30),complex)
    for k in range(30):
        for src in range(3):
            bit=51*k+3+16*src;a=(val>>bit)&255;b=(val>>(bit+8))&255
            out[order[src],k]=complex(a if a<128 else a-256,b if b<128 else b-256)
    return out
def main():
    assert sha(R/'config.json')==(R/'config.sha256').read_text().strip()
    data=pd.read_csv(R/'empty_manifest.csv');runs=[];states=[];events=[];hashes=[]
    for row in data.itertuples(index=False):
        p=Path(row.path);assert sha(p)==row.sha256;m=load(p.parent/'run.json');h,pay=headers(p)
        assert h.header_ht40_ok.all() and (h.subcarrier_offset==0).all()
        ev=re.findall(r'([\d.]+)\s+READY (G[1-4]|C[12]) chains=([\d,]+) t=(\d+) seq_idx=(\d)/6',(p.parent/'events.log').read_text())
        for host,state,chains,dev,seq in ev:
            assert [int(z) for z in chains.split(',')]==rat.MAP[state]
            assert int(seq)==list(rat.MAP).index(state)
            events.append(dict(index=row.index,capture=row.capture_id,run=row.run,state=state,host_relative_s=float(host)-m['start_ns']/1e9,device_us=int(dev),device_relative_s=(int(dev)-int(ev[0][3]))/1e6))
        perms=collections.Counter(int(x) for x in h.ant)
        # Cross-source mask discrepancy recorded, not used to select a convenient order.
        amb=int(sum(n for a,n in perms.items() if (a&3)==2))
        rr=dict(index=row.index,capture=row.capture_id,run=row.run,role=row.new_role,path=str(p),sha256=row.sha256,
            requested_seconds=m['requested_seconds'],frames=len(h),window_frames=m['window']['records'],
            host_start_ns=int(h.host.iloc[0]),host_end_ns=int(h.host.iloc[-1]),host_nonmonotonic=int((np.diff(h.host)<=0).sum()),
            nic_start=int(h.nic.iloc[0]),nic_end=int(h.nic.iloc[-1]),nic_nonmonotonic=int((np.diff(h.nic)<=0).sum()),
            host_interval_median_ms=float(np.median(np.diff(h.host))/1e6),host_gap_max_ms=float(np.max(np.diff(h.host))/1e6),
            boot_id=m['before']['boot_id'],ifindex=m['before']['ifindex'],phy=m['before']['phy'].split('/')[-1],identity_unchanged=m['identity_unchanged'],
            logger_initialization_note=m.get('radio_initialization'),switch_reset_note=m.get('esp32_reset'),
            first_device_us=int(ev[0][3]),last_device_us=int(ev[-1][3]),permutation_counts=json.dumps(perms),
            upstream_onebit_mask_disagreement_packets=amb,metadata_sha256=sha(p.parent/'run.json'),events_sha256=sha(p.parent/'events.log'))
        hashes.append(dict(index=row.index,raw_sha256=row.sha256,metadata_sha256=rr['metadata_sha256'],events_sha256=rr['events_sha256']))
        if row.new_role!='provenance_only':
            x,host,nic=dec.decode(p);assert np.array_equal(host,h.host) and np.array_equal(nic,h.nic)
            for q in np.linspace(0,len(x)-1,min(len(x),64),dtype=int):assert np.array_equal(x[q],independent_frame(pay[q],int(h.ant.iloc[q])))
            ts=(host-m['start_ns'])/1e9;starts,labels,guard=dec.align(p.parent/'events.log',m['start_ns']/1e9);rr['guard_ms']=guard*1e3
            shift=np.int64(10**12)
            shifted_relative=(host+shift)-(np.int64(m['start_ns'])+shift)
            rr['relative_timestamp_shift_max_error_ns']=int(np.max(abs(shifted_relative-(host-m['start_ns']))))
            assert rr['relative_timestamp_shift_max_error_ns']==0
            for state in rat.MAP:
                ids=[];win=[];margins=[]
                for j,(st,si) in enumerate(zip(starts[:-1],labels[:-1])):
                    if st+.025>m['requested_seconds'] or dec.STATES[si]!=state:continue
                    l,u=np.searchsorted(ts,[st+guard,st+.025-guard]);ind=np.arange(l,u);ids.extend(ind);win.extend([j]*len(ind));margins.extend(np.minimum(ts[ind]-(st+guard),st+.025-guard-ts[ind]))
                ids=np.array(ids,dtype=int);old=np.load(OLD/'packets'/f'{row.index:03d}_{state}.npz');assert np.array_equal(ids,old['packet_id'])
                z=rat.unit(x[ids].astype(complex));am=abs(z);low=am<CFG['amplitude_support_fraction']*np.median(am,axis=2,keepdims=True)
                sp=invariants(z);disc=np.linalg.norm(sp['covariance']-old['cov_A'])/np.linalg.norm(old['cov_A']);assert disc<1e-10
                np.savez_compressed(R/'cache'/f'{row.index:03d}_{state}.npz',x=z,packet_id=ids,host_ns=host[ids],nic_timestamp=nic[ids],window_id=np.array(win),margin_s=np.array(margins),relative_s=ts[ids],ant_sel=h.ant.to_numpy()[ids],**sp)
                states.append(dict(index=row.index,capture=row.capture_id,run=row.run,role=row.new_role,state=state,packets=len(ids),windows=len(set(win)),deep_fade_fraction=float(low.mean()),pair_coherence_median=float(np.median(sp['pair_coherence'])),baseline_cov_discrepancy=float(disc),raw_norm_median=float(np.median(np.linalg.norm(x[ids],axis=(1,2)))),guard_ms=guard*1e3))
        runs.append(rr);print('Audited',row.capture_id,row.run,len(h),flush=True)
    pd.DataFrame(runs).to_csv(R/'results/source_audit.csv',index=False);pd.DataFrame(states).to_csv(R/'results/state_audit.csv',index=False);pd.DataFrame(events).to_csv(R/'results/events.csv',index=False);pd.DataFrame(hashes).to_csv(R/'results/input_hashes.csv',index=False)
    dump(R/'results/audit_summary.json',dict(all_12_hashes_match=True,confirmed_empty_runs=6,states=len(states),decode_reference_frames_per_run=64,old_selection_exact_match=True,source_dates_not_used=True,
         note='Current upstream PicoScenes header has a differing one-bit first-ant mask; own parser agrees with original Halperin two-bit map. No automatic reorder based on data resemblance. Independent scalar payload decoder checked.'))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
