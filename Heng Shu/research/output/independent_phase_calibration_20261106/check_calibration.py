"""Known-source calibration intake, fixed relative-phase fitting and held validation.

No instrument control. Reads the existing run.json/events.log/PicoScenes format.
Human direction selection is outside this active-source checker.
"""
from pathlib import Path
import argparse, json, sys
from hashlib import sha256
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'coherent_array_validation_20261106'))
import validate_coherent as c
ST=c.ST;MAP=c.base.MAP;F=c.F;Y=c.base.Y;ANG=np.arange(-80.,81.)

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def valid_xyz(x):return isinstance(x,list) and len(x)==3 and all(isinstance(v,(int,float)) and np.isfinite(v) for v in x)

def preflight(manifest):
    missing=[];records=manifest.get('records',[])
    if not manifest.get('rx_epoch_id'):missing.append('rx_epoch_id 未填')
    if not manifest.get('operator_confirms_no_reinitialization'):missing.append('尚未确认整个采集段无重新初始化')
    if not valid_xyz(manifest.get('rx_xyz_m')) or not manifest.get('rx_geometry_measured'):missing.append('RX 实测坐标/高度未填')
    if manifest.get('state_map')!=MAP:missing.append('状态映射与当前硬件映射不一致')
    frequencies=manifest.get('frequency_hz',[])
    if len(frequencies)!=30 or not np.allclose(np.array(frequencies),F):
        missing.append('当前解析器仅支持已核实的 HT40 频率')
    ids=[r.get('id') for r in records]
    if len(set(ids))!=len(ids):missing.append('记录ID重复')
    if not {'CAL0_R1','CAL0_R2','CAL0_R3'}.issubset(ids):missing.append('缺少三个 CAL0 记录')
    expected_roles={'CAL0_R1':'fit','CAL0_R2':'development','CAL0_R3':'holdout'}
    allowed_roles={'fit','development','holdout','validation','return'}
    for r in records:
        if r.get('role') not in allowed_roles:missing.append(str(r.get('id'))+' 不是主动源标定记录角色')
        if r.get('id') in expected_roles and r.get('role')!=expected_roles[r['id']]:missing.append(r['id']+' 拟合/开发/留出身份被更改')
    directions=[]
    if valid_xyz(manifest.get('rx_xyz_m')):
        rx=manifest['rx_xyz_m']
        for r in records:
            if r.get('role')=='validation' and valid_xyz(r.get('actual_tx_xyz_m')):
                tx=r['actual_tx_xyz_m']
                directions.append(float(np.degrees(np.arctan2(rx[1]-tx[1],rx[0]-tx[0]))))
            if valid_xyz(r.get('actual_tx_xyz_m')) and abs(r['actual_tx_xyz_m'][2]-rx[2])>.05:
                missing.append(str(r.get('id'))+' 高度差超过5cm，当前水平AoA验收需复核等高布置或改用三维模型')
        if not any(a<=-10 for a in directions) or not any(a>=10 for a in directions):
            missing.append('最小完整验收需要未拟合的正、负侧向源记录，不能只有0度重复')
    folders=[str(Path(r['run_dir']).resolve()) for r in records if r.get('run_dir')]
    if len(set(folders))!=len(folders):missing.append('多个记录重复使用同一run目录')
    identities=[]
    for r in records:
        prefix=r.get('id','unknown')
        if not valid_xyz(r.get('actual_tx_xyz_m')) or not r.get('measured'):missing.append(prefix+' TX 实测坐标/高度未填')
        if r.get('rx_epoch_id')!=manifest.get('rx_epoch_id') or not r.get('rx_epoch_id'):missing.append(prefix+' RX会话编号未核实')
        if not r.get('run_dir'):
            missing.append(prefix+' 原始数据路径未填');continue
        folder=Path(r['run_dir'])
        if not all((folder/n).is_file() for n in ('run.json','events.log')) or len(list(folder.glob('*.csi')))!=1:
            missing.append(prefix+' 需要唯一CSI文件及run.json/events.log');continue
        m=read(folder/'run.json')
        if not m.get('continuity_gate') or m.get('before')!=m.get('after') or m.get('failure'):
            missing.append(prefix+' 轮内连续性检查未通过')
        if m.get('radio_initialization')!='none; logger restarts only':missing.append(prefix+' 初始化日志标记需要人工核对')
        identities.append(m.get('before'))
        if abs(float(m.get('requested_seconds',0))-float(r.get('expected_duration_s',10)))>.01:missing.append(prefix+' 时长与预先记录不一致')
    if identities and any(x!=identities[0] for x in identities):missing.append('不同记录的 RX 身份/无线配置不一致')
    return dict(ready=not missing,missing=missing,status='ready_for_offline_check' if not missing else 'pending_actual_capture_or_metadata',
        note='同一ID不能代替日志证据；此检查不证明直达波占优。')

def load_record(rec):
    folder=Path(rec['run_dir']);meta=read(folder/'run.json');path=next(folder.glob('*.csi'))
    x,host,nic=c.base.decoder.decode(path);ta,_=c.base.frame_details(path)
    alignment=c.base.decoder.align(folder/'events.log',meta['start_ns']/1e9)
    if alignment is None:raise ValueError('不能对齐状态事件')
    starts,order,guard=alignment
    ev=c.base.EVENT.findall((folder/'events.log').read_text(encoding='utf8'))
    assert all(list(map(int,e[2].split(',')))==MAP[e[1]] for e in ev),'状态链映射错误'
    assert np.all(np.diff(host)>0) and np.all(np.diff(nic)>0)
    t=(host-meta['start_ns'])/1e9;dur=float(meta['requested_seconds'])
    idx=np.searchsorted(starts,t,side='right')-1;v=np.maximum(idx,0);dt=t-starts[v]
    keep=(ta==c.base.APPARENT_TA)&(t>=0)&(t<dur)&(idx>=0)&(dt>=max(.003,guard))&(dt<min(.021,.025112-guard))
    states={};quality=[];ledger=[]
    for si,s in enumerate(ST):
        ids=np.where(keep&(order[v]==si))[0]
        eligible_visits=np.where((order==si)&(starts>=0)&(starts<dur-.025112))[0]
        used=np.unique(idx[ids]);coverage=len(set(eligible_visits)&set(used))/max(len(eligible_visits),1)
        minimum=int(900*dur/10)
        ok=len(ids)>=minimum and coverage>=.90
        quality.append(dict(id=rec['id'],state=s,packets=len(ids),visit_coverage=coverage,minimum=minimum,pass_capture=ok))
        if len(ids)<864:raise ValueError(rec['id']+' '+s+' 不足864包，无法做本轮等预算检查')
        # Bound calibration calculation at1024; source angle comparisons use the first864 chosen below.
        ii=ids[c.select(len(ids),min(1024,len(ids)))];states[s]=x[ii]
        ledger.extend(dict(id=rec['id'],state=s,purpose='phase_model',packet=int(i),time_s=float(t[i])) for i in ii)
    cyc={n:[np.where(keep&(idx==6*n+j))[0] for j in range(6)] for n in range(len(starts)//6)}
    eligible=[n for n,g in cyc.items() if all(len(a)>=6 for a in g)]
    if len(eligible)<24:raise ValueError(rec['id']+' 不足24完整周期')
    use=np.array(eligible)[c.select(len(eligible),24)]
    arr=[]
    for n in use:
        g=[]
        for j,s in enumerate(ST):
            ids=cyc[n][j][c.select(len(cyc[n][j]),6)];g.append(x[ids])
            ledger.extend(dict(id=rec['id'],state=s,purpose='twelve',packet=int(i),time_s=float(t[i])) for i in ids)
        arr.append(g)
    ids=np.where(keep&(order[v]==1))[0];ids=ids[c.select(len(ids),864)]
    ledger.extend(dict(id=rec['id'],state='G2',purpose='G2_equal_budget',packet=int(i),time_s=float(t[i])) for i in ids)
    provenance=dict(id=rec['id'],csi_path=str(path),sha256=digest(path),events_sha256=digest(folder/'events.log'),run_json_sha256=digest(folder/'run.json'))
    return dict(states=states,switched=np.asarray(arr),g2=x[ids],quality=quality,ledger=ledger,provenance=provenance)

def propagation(state,tx,rx):
    ids=np.array(MAP[state])-1
    xyz=np.column_stack([np.full(3,rx[0]),rx[1]-Y[ids],np.full(3,rx[2])])
    d=np.linalg.norm(xyz-np.array(tx),axis=1)
    return np.exp(-2j*np.pi/c.base.C*(d[:,None]-d[0])*F)

def phase_samples(x,state,tx,rx):
    u=c.unit(x.astype(complex));v=u*np.conj(u[:,0:1,:])
    observed=np.divide(v,abs(v),out=np.full_like(v,np.nan),where=abs(v)>1e-15)
    return observed/propagation(state,tx,rx)[None]

def fit_models(record,tx,rx):
    models={k:{} for k in ('constant','affine')};masks={}
    for s,x in record['states'].items():
        z=phase_samples(x,s,tx,rx)
        mean=np.nanmean(z,axis=0)
        norm=c.unit(x.astype(complex));amplitude=np.median(abs(norm*np.conj(norm[:,0:1,:])),axis=0)
        keep=(abs(mean)>=.8)&(amplitude>=.1*np.median(amplitude,axis=1,keepdims=True))
        keep[0]=True;masks[s]=keep
        for model in models:models[model][s]=np.ones((3,30),complex)
        for j in (1,2):
            use=keep[j]&c.TRAIN
            if use.sum()<12:continue # coverage gate will reject; do not invent a fitted response.
            phi=np.angle(np.sum(mean[j,use]))
            models['constant'][s][j]=np.exp(-1j*phi)
            unit_mean=np.exp(1j*np.angle(mean[j:j+1]))
            t,p,_,_=c.rat.fit(unit_mean,np.ones((1,30)),use[None])
            models['affine'][s][j]=np.exp(1j*(2*np.pi*c.FB*t*1e-9+p))
    return models,masks

def phase_check(record,tx,rx,model,masks,rec_id):
    rows=[]
    for s,x in record['states'].items():
        z=phase_samples(x,s,tx,rx)*model[s][None]
        for j in (1,2):
            mask=masks[s][j];v=z[:,j,mask];good=np.isfinite(v)
            frac=mask.mean()*good.mean() if v.size else 0.
            ph=abs(np.angle(v[good]))*180/np.pi
            p90=float(np.quantile(ph,.9)) if len(ph) else np.nan
            shift=float(abs(np.angle(np.mean(v[good])))*180/np.pi) if len(ph) else np.nan
            rows.append(dict(id=rec_id,state=s,chain=j+1,retained_fraction=frac,phase_p90_deg=p90,mean_residual_deg=shift,
                pass_phase=bool(frac>=.8 and np.isfinite(p90) and p90<=15),return_pass=bool(frac>=.8 and p90<=15 and shift<=10)))
    return rows

def source_peak(cov,y,mask):
    _,v=np.linalg.eigh(c.herm(cov));h=v[:,:,-1]
    N=np.eye(len(y))-h[:,:,None]*h[:,None,:].conj()
    a=np.exp(2j*np.pi/c.base.C*F[:,None,None]*y[None,:,None]*np.sin(np.radians(ANG))[None,None,:])
    cost=np.maximum(np.einsum('kma,kmn,kna->ka',a.conj(),N,a).real,0)
    out=[]
    for ids in (np.arange(30),np.arange(15),np.arange(15,30)):
        required=int(np.ceil(.8*len(ids)))
        ids=ids[mask[ids]]
        out.append(float(ANG[np.argmin(cost[ids].sum(0))]) if len(ids)>=required else np.nan)
    return out

def source_angles(record,rot,masks):
    rows=[]
    for s in ST[:4]:
        x=record['g2'] if s=='G2' else record['states'][s]
        x=x*rot[s][None]
        cov=90*c.base.spatial(x);ang=source_peak(cov,Y[np.array(MAP[s])-1],masks[s].all(0))
        rows.append(dict(arm=s,angle=ang[0],low=ang[1],high=ang[2],closure_fraction=np.nan))
    vectors=[];good=[]
    for cycle in record['switched']:
        reps=[]
        for si,s in enumerate(ST):
            x=c.unit(cycle[si].astype(complex)*rot[s][None]);ref=x[0]
            corrected=[]
            for packet in x:
                t,p,_,_=c.rat.fit(packet,ref);corrected.append(c.rat.apply(packet,t,p))
            reps.append(np.mean(corrected,axis=0))
        v,info,_=c.assemble(np.array(reps),2);vectors.append(v)
        good.append(info['all_tree_edges_pass'] and info['pass_closure'])
    cov=c.cov_vectors(np.array(vectors)).mean(0)
    mask=np.array([masks[s].all(0) for s in ST]).all(0)
    ang=source_peak(cov,Y,mask)
    rows.append(dict(arm='twelve',angle=ang[0],low=ang[1],high=ang[2],closure_fraction=float(np.mean(good))))
    return rows

def run(manifest_path,out):
    manifest=read(manifest_path);pf=preflight(manifest)
    if not pf['ready']:return pf
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('输出目录非空；请用新目录保留旧检查')
    records={r['id']:r for r in manifest['records']};loaded={};quality=[];ledger=[];provenance=[]
    for ident,r in records.items():
        z=load_record(r);loaded[ident]=z;quality.extend(z['quality']);ledger.extend(z['ledger']);provenance.append(z['provenance'])
    if len({r['sha256'] for r in provenance})!=len(provenance):
        raise ValueError('发现相同CSI哈希，不能把重复文件当作独立源记录')
    for name,data in [('capture_quality.csv',quality),('packet_indices.csv',ledger),('input_manifest.csv',provenance)]:
        pd.DataFrame(data).to_csv(out/name,index=False,encoding='utf-8-sig')
    if not all(r['pass_capture'] for r in quality):
        (out/'STATUS.json').write_text(json.dumps(dict(status='capture_quality_failed',human_localization_verified=False)),encoding='utf8')
        return dict(status='capture_quality_failed',out=str(out))
    rx=manifest['rx_xyz_m'];fit=records['CAL0_R1'];dev=records['CAL0_R2']
    models,masks=fit_models(loaded[fit['id']],fit['actual_tx_xyz_m'],rx)
    dev_rows=[];choice=None
    for name,model in models.items():
        rr=phase_check(loaded[dev['id']],dev['actual_tx_xyz_m'],rx,model,masks,dev['id'])
        dev_rows.extend(dict(model=name,**r) for r in rr)
        if choice is None and all(r['pass_phase'] for r in rr):choice=name
    pd.DataFrame(dev_rows).to_csv(out/'development_models.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'fitted_models.npz',**{f'{name}_{s}':v for name,d in models.items() for s,v in d.items()},**{f'mask_{s}':v for s,v in masks.items()})
    frozen=dict(chosen_model=choice,fit_id=fit['id'],development_id=dev['id'],manifest_sha256=digest(manifest_path),
        code_sha256=digest(__file__),fitted_models_sha256=digest(out/'fitted_models.npz'),
        claim='effective relative array response hypothesis, no absolute ToF or human detection')
    (out/'model_freeze.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
    if choice is None:
        status=dict(status='no_model_passed_development',physical_calibration_verified=False,out=str(out))
        (out/'STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8');return status
    checked=[];pred=[]
    for ident,r in records.items():
        if r['role'] in ('fit','development'):continue
        checked.extend(phase_check(loaded[ident],r['actual_tx_xyz_m'],rx,models[choice],masks,ident))
        for variant,rotation in [('uncorrected',{s:np.ones((3,30),complex) for s in ST}),('frozen_calibrated',models[choice])]:
            pred.extend(dict(id=ident,variant=variant,**a) for a in source_angles(loaded[ident],rotation,masks))
    pd.DataFrame(pred).to_csv(out/'blind_source_peaks.csv',index=False,encoding='utf-8-sig')
    (out/'prediction_freeze.json').write_text(json.dumps(dict(peaks_sha256=digest(out/'blind_source_peaks.csv'),model_sha256=digest(out/'model_freeze.json'))),encoding='utf8')
    pd.DataFrame(checked).to_csv(out/'held_phase_checks.csv',index=False,encoding='utf-8-sig')
    for r in pred:
        rec=records[r['id']];tx=rec['actual_tx_xyz_m'];yc=0 if r['arm']=='twelve' else float(np.mean(Y[np.array(MAP[r['arm']])-1]))
        gt=float(np.degrees(np.arctan2(rx[1]-tx[1]-yc,rx[0]-tx[0])))
        r.update(theory_angle=gt,error=abs(r['angle']-gt),role=rec['role'])
    scored=pd.DataFrame(pred);scored.to_csv(out/'source_angles_scored.csv',index=False,encoding='utf-8-sig')
    checks=pd.DataFrame(checked);hold=checks[checks.id.map(lambda k:records[k]['role']=='return')]
    phase_ok=bool(checks.pass_phase.all() and hold.return_pass.all())
    active=scored[(scored.variant=='frozen_calibrated')&(scored.role=='validation')]
    angle_rows=[]
    for arm,g in active.groupby('arm'):
        usable=np.isfinite(g.angle)&(abs(g.angle)<79)&(abs(g.low-g.high)<=6)
        if arm=='twelve':usable=usable&(g.closure_fraction>=.9)
        mean=g.loc[usable,'error'].mean();p90=g.loc[usable,'error'].quantile(.9)
        each=g.assign(direction=g.id.map(lambda k:records[k]['position_id'])).groupby('direction').error.mean()
        ok=bool(usable.mean()>=.9 and mean<=5 and p90<=8 and each.max()<=8)
        angle_rows.append(dict(arm=arm,available=int(usable.sum()),total=len(g),MAE=mean,P90=p90,maximum_direction_MAE=each.max(),pass_source=ok))
    pd.DataFrame(angle_rows).to_csv(out/'source_validation_summary.csv',index=False,encoding='utf-8-sig')
    status=dict(status='source_validation_supported' if phase_ok and len(angle_rows)==5 and all(r['pass_source'] for r in angle_rows) else 'source_validation_failed_or_incomplete',
        phase_passed=phase_ok,human_localization_verified=False,absolute_tof_verified=False,
        scope='Only this receiver epoch and measured active-source directions. Human test and 12-vs3 gain still require separate evaluation.',out=str(out))
    (out/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8');return status

def self_test():
    rng=np.random.default_rng(28);rx=[3.65,0,1.0]
    def simulated(tx):
        states={}
        for s in ST:
            ids=np.array(MAP[s])-1
            xyz=np.column_stack([np.full(3,3.65),-Y[ids],np.ones(3)])
            d=np.linalg.norm(xyz-np.array(tx),axis=1)
            a=np.exp(-2j*np.pi/c.base.C*d[:,None]*F)
            hw=np.exp(1j*(.35*np.sin(ids[:,None]+1)+2*np.pi*c.FB[None]*ids[:,None]*1.2e-9))
            common=np.exp(1j*(rng.uniform(-np.pi,np.pi,160)[:,None,None]+2*np.pi*c.FB[None,None]*rng.uniform(-50,50,160)[:,None,None]*1e-9))
            states[s]=(a*hw)[None]*common
        return dict(states=states)
    t0=[0,0,1.0];data=simulated(t0);models,masks=fit_models(data,t0,rx)
    dev=phase_check(simulated(t0),t0,rx,models['affine'],masks,'SIM_DEV')
    assert all(r['phase_p90_deg']<.02 for r in dev)
    tests=[]
    for theta in (-20,20,-45,45):
        tx=[3.65-3.65*np.cos(np.radians(theta)),-3.65*np.sin(np.radians(theta)),1.0]
        record=simulated(tx);res=phase_check(record,tx,rx,models['affine'],masks,'SIM_VALID')
        assert all(r['phase_p90_deg']<.02 for r in res)
        for s,x in record['states'].items():
            if s not in ST[:4]:continue
            a=source_peak(90*c.base.spatial(x*models['affine'][s][None]),Y[np.array(MAP[s])-1],np.ones(30,bool))[0]
            theory=np.degrees(np.arctan2(-tx[1]-np.mean(Y[np.array(MAP[s])-1]),3.65-tx[0]))
            assert abs(a-theory)<1.1,(s,theta,a,theory)
        record['g2']=record['states']['G2']
        record['switched']=np.array([[record['states'][s][6*i:6*i+6] for s in ST] for i in range(24)])
        full=source_angles(record,models['affine'],masks)[-1]
        assert abs(full['angle']-theta)<=1 and full['closure_fraction']>.99,full
        tests.append(theta)
    corrupt=simulated(t0);corrupt['states']['G1'][:,1]*=np.exp(1j*.7)
    bad=phase_check(corrupt,t0,rx,models['affine'],masks,'SIM_BAD')
    assert any(not r['pass_phase'] for r in bad)
    pending=preflight(read(ROOT/'session_template.json'));assert not pending['ready']
    result=dict(status='software_self_test_passed',synthetic_angles=tests,held_bad_branch_detected=True,
        incomplete_manifest_rejected=True,full12_source_angle_and_closure=True,
        physical_calibration_verified=False,new_measurements_processed=False)
    (ROOT/'self_test_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','run','self-test']);ap.add_argument('--manifest',type=Path,default=ROOT/'session_template.json');ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    if a.mode=='self-test':res=self_test()
    elif a.mode=='preflight':res=preflight(read(a.manifest))
    else:
        if a.out is None:ap.error('run requires --out pointing to a new empty output directory')
        res=run(a.manifest,a.out)
    print(json.dumps(res,ensure_ascii=False,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
