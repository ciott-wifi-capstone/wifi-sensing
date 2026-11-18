from geometry_update import *

def exact_bistatic_steering(xy, antenna_ids, hardware_delay_s=0.):
    """Full-frequency spherical point-scatterer model; no amplitude / hardware phase calibration.

    Used only for model checks here, NOT to estimate a location from uncalibrated data.
    Input xy must be a candidate independent of test ground truth during future inference.
    """
    x,y=xy;ys=Y[np.array(antenna_ids)-1]
    path=np.hypot(x-3.65,y)+np.hypot(x,y-ys)
    return np.exp(-2j*np.pi*(FC+F[None,:])*(path[:,None]/C+hardware_delay_s))

def main():
    # Exact spherical model tends to a plane-wave spatial ratio at very large range.
    beta=np.deg2rad(30);r=1e5
    s=exact_bistatic_steering((r*np.cos(beta),r*np.sin(beta)),MAP['C1'])
    ratio=s/s[0:1]
    ys=Y[np.array(MAP['C1'])-1];target=np.exp(2j*np.pi*(FC+F[None,:])*(ys[:,None]-ys[0])*np.sin(beta)/C)
    assert np.max(np.abs(np.angle(ratio*target.conj())))<1e-4
    # Symmetry check at boresight must preserve near-field path curvature, not force equal phases.
    ys=Y;paths=3.65+np.hypot(2.,ys)
    assert np.allclose(paths,paths[::-1]) and paths[0]>paths[5]
    inputs=pd.read_csv(DEST/'inputs.csv');cov=np.load(DEST/'covariances.npz');rows=[]
    ast={st:steering(Y[np.array(ids)-1]) for st,ids in MAP.items()}
    for key in cov.files:
        idx,st=key.split('_');idx=int(idx)
        for q in [1,2,3,4]:
            s=music(cov[key],ast[st],q);i,j=np.unravel_index(s.argmax(),s.shape)
            rows.append(dict(index=idx,group=inputs.iloc[idx].analysis_group,state=st,model_order=q,angle_deg=ANG[i],raw_delay_ns=TAU[j]))
    pd.DataFrame(rows).to_csv(DEST/'model_order_sensitivity.csv',index=False,encoding='utf-8-sig')
    print('Exact spherical model checks passed; model-order cases',len(rows))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
