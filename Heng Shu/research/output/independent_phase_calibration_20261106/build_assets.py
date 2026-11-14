"""New field protocol and bounded audit of cached branch quality. No new capture."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'coherent_array_validation_20261106'
STATES=['G1','G2','G3','G4','C1','C2']
MAP={'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]}

def main():
    reg=json.loads((OLD/'prediction_freeze.json').read_text())
    rows=[];quality=[];refs=[]
    for site in 'ABCDE':
        f=OLD/f'{site}_selected.npz'
        assert sha256(f.read_bytes()).hexdigest()==reg[f.name]
        x=np.load(f)['switched'].astype(complex)
        df=pd.read_csv(OLD/f'{site}_packet_reference.csv')
        for si,s in enumerate(STATES):
            z=x[:,:,si];power=np.mean(abs(z)**2,axis=-1)
            share=power/np.maximum(power.sum(-1,keepdims=True),1e-15)
            for ch in range(3):
                rows.append(dict(point=site,state=s,chain=ch+1,physical_antenna=MAP[s][ch],
                    median_within_packet_power_share=float(np.median(share[...,ch])),
                    median_raw_magnitude=float(np.median(abs(z[...,ch,:]))),
                    exact_zero_fraction=float(np.mean(abs(z[...,ch,:])==0))))
            d=df[df.state==s]
            quality.append(dict(point=site,state=s,checked_packets=len(d),pass_packets=int(d.pass_common.sum()),
                pass_fraction=d.pass_common.mean(),chain_phase_spread_p90_deg=d.chain_phase_spread_deg.quantile(.9),
                chain_delay_spread_p90_ns=d.chain_delay_spread_ns.quantile(.9),held_corr_median=d.held_corr.median()))
        refs.append(dict(file=str(f),sha256=reg[f.name]))
    q=pd.DataFrame(quality);p=pd.DataFrame(rows)
    q.to_csv(ROOT/'existing_state_quality.csv',index=False,encoding='utf-8-sig')
    p.to_csv(ROOT/'existing_chain_power.csv',index=False,encoding='utf-8-sig')
    group=q.groupby('state',sort=False)[['checked_packets','pass_packets']].sum()
    group['pass_fraction']=group.pass_packets/group.checked_packets
    group.to_csv(ROOT/'existing_state_quality_summary.csv',encoding='utf-8-sig')
    positions=[]
    for label,theta,scope in [('T0',0,'fit_and_return'),('Tn20',-20,'minimal_validation'),('Tp20',20,'minimal_validation'),('Tn45',-45,'extended_validation'),('Tp45',45,'extended_validation')]:
        positions.append(dict(position_id=label,planned_angle_deg=theta,planned_x_m=3.65-3.65*np.cos(np.radians(theta)),
            planned_y_m=-3.65*np.sin(np.radians(theta)),planned_range_m=3.65,scope=scope,
            actual_x_m=None,actual_y_m=None,actual_z_m=None,actual_measured=False))
    pd.DataFrame(positions).to_csv(ROOT/'TX_positions_to_measure.csv',index=False,encoding='utf-8-sig')
    records=[]
    for group_id,pos,role in [('CAL0','T0','calibration'),('VALn20','Tn20','validation'),('RETn20','T0','return'),('VALp20','Tp20','validation'),('RETp20','T0','return')]:
        for run in (1,2,3):
            use=({'1':'fit','2':'development','3':'holdout'}[str(run)] if group_id=='CAL0' else role)
            records.append(dict(id=f'{group_id}_R{run}',position_id=pos,role=use,expected_duration_s=10,
                actual_tx_xyz_m=[None,None,None],measured=False,rx_epoch_id='',run_dir='',notes=''))
    manifest=dict(version=1,status='template_not_captured',session_id='',rx_epoch_id='',operator_confirms_no_reinitialization=False,
        rx_xyz_m=[3.65,0,None],rx_geometry_measured=False,frequency_hz=(5.23e9+np.arange(-58,59,4)*312500).tolist(),
        state_map=MAP,antenna_y_local_m=((6.5-np.arange(1,13))*.028).tolist(),records=records)
    (ROOT/'session_template.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame([{k:v for k,v in r.items() if k!='actual_tx_xyz_m'}|{'actual_x_m':'','actual_y_m':'','actual_z_m':''} for r in records]).to_csv(ROOT/'capture_log.csv',index=False,encoding='utf-8-sig')
    gates=dict(status='proposed engineering limits, frozen before new validation data',phase_p90_deg=15,
        return_mean_phase_deg=10,minimum_retained_tone_fraction=.8,minimum_valid_packet_fraction=.9,
        per_G_and_full12_MAE_deg=5,per_G_and_full12_P90_deg=8,per_direction_MAE_deg=8,
        output_fraction=.9,low_high_difference_deg=6,phase_model_order=['constant','affine'],
        fitting_ids=['CAL0_R1'],development_ids=['CAL0_R2'],holdout_ids=['CAL0_R3','VAL*','RET*'],
        nominal_min_packets_per_state_per10s=900,expected_full_cycles_per10s=66,
        parser='existing PicoScenes Intel5300 HT40 5230/5240MHz, chain permutation and target MAC checks',
        source='WiCAL II-B, exact spherical propagation adapted to current linear array',
        human_status='not tested by active-source calibration')
    (ROOT/'acceptance_protocol.json').write_text(json.dumps(gates,ensure_ascii=False,indent=2),encoding='utf8')
    (ROOT/'input_provenance.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    ax.plot([0,3.65],[0,0],color='#74818f',lw=2)
    ax.plot(3.65,0,'ks',ms=10);ax.annotate('RX 固定\n阵列中点 (3.65, 0)',(3.65,0),xytext=(15,8),textcoords='offset points')
    for p0 in positions:
        if '45' in p0['position_id']:continue
        a,b=p0['planned_x_m'],p0['planned_y_m'];ax.plot(a,b,'o',ms=10,color='#2469c4')
        ax.plot([a,3.65],[b,0],'--',color='#a3b5c9',lw=1)
        labelxy='(0.000, 0.000)' if p0['position_id']=='T0' else f'({a:.3f}, {b:+.3f})'
        ax.annotate(f"{p0['position_id']}  {p0['planned_angle_deg']:+d}°\n{labelxy}",(a,b),xytext=(8,8),textcoords='offset points')
    ax.text(.7,-1.95,'顺序  T0 → T−20 → T0 → T+20 → T0\n每处 3 × 10 s，全部无人。移动 TX，RX 不动。\n这是计划坐标，采前实测；验证方向不用于拟合。',fontsize=11)
    ax.set(xlim=(-.4,5.15),ylim=(-2.3,1.9),aspect='equal',xlabel='全局 x / m（原 TX 指向 RX）',ylabel='全局 y / m',title='最小独立标定验证：仅移动 TX')
    ax.grid(alpha=.2);fig.savefig(ROOT/'TX独立验证点位.png',dpi=180);fig.savefig(ROOT/'TX独立验证点位.svg');plt.close(fig)
    table=p[p.state=='G1'][['point','chain','physical_antenna','median_within_packet_power_share']]
    report=f'''# 采前新增诊断：G1 需要单独检查

来源为上一轮已冻结的 A–E 有界抽样缓存，每点三轮、每轮24周期、每状态每周期6包。没有采集新数据。

## 逐状态公共误差模型检查

{group.to_markdown(floatfmt='.4f')}

G1 只有 {group.loc['G1','pass_fraction']:.2%} 的抽查包通过全部公共模型门槛，其他五状态整体约99.8%–100%。这不应被整体约90%的平均数掩盖。

## G1 各链幅度情况

{table.to_markdown(index=False,floatfmt='.4f')}

第二条接收链对应物理天线2，包内功率占比中位数约2.4%–6.2%。这些是原始CSI数字量化幅度的相对统计，不能当作绝对接收功率或SNR。弱分量容易使独立相位估计受噪声影响，因此当前检查失败并不能单独证明链2发生了独立振荡器变化，也不能据此断定天线损坏。

下一次先看 G1 是否仍表现异常。若0°同方向重复仍不稳，暂停完整十二阵元验收。在保存原件、断开当前标定有效性的前提下，再检查天线2对应的支路、接头、开关端口；换接后必须新建硬件会话并重做标定。若异常随已知TX方向而变化而回零恢复，也应考虑深衰落或方向响应。不能静默删除天线2后继续称为十二阵元结果。
'''
    (ROOT/'采前G1专项诊断.md').write_text(report,encoding='utf8')

if __name__=='__main__':main()
