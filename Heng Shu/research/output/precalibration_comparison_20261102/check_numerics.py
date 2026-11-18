from models import *
from scipy.linalg import null_space
checks=[]
def ck(name,error,tol):
 checks.append(dict(test=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
# A synthetic identical channel with a chain2 pi jump must select that jump.
rng=np.random.default_rng(317);anchor=np.exp(1j*rng.uniform(-np.pi,np.pi,(6,2,30)));v=anchor*np.array([-1,1])[None,:,None];rot,diag=rotation(v,anchor,'pi');ck('global_pi_choice_exact',np.max(abs(rot-np.array([-1,1]))),1e-12)
# Geometry removed when estimating hardware must be restored, not zeroed, in data.
hw=np.exp(1j*(.4+.03*(F-F.mean())/1e6))[None,None,:]*np.ones((6,2,1));scene=GEOM*hw
calibrated=scene/(scene/GEOM);ck('calibration_preserves_known_propagation',np.max(abs(calibrated-GEOM)),1e-12)
# Model fit recovers an affine relative hardware response under an exact direct source.
u=np.repeat(scene[:,None,:,:],128,axis=1);t=fit([0],{0:u},'affine');ck('affine_model_synthetic_direct_source',np.max(abs(t['ghat']-hw)),1e-12)
t=fit([0],{0:u},'tone');ck('interpolated_tone_affine_held_frequencies',np.max(abs(t['ghat'][:,:,KTE]-hw[:,:,KTE])),1e-12)
# Unit-modulus inverse calibration does not change amplitudes.
x=rng.normal(size=(10,3,30))+1j*rng.normal(size=(10,3,30));g=np.exp(1j*rng.normal(size=(3,30)));ck('phase_only_amplitude_invariant',np.max(abs(abs(x/g)-abs(x))),1e-12)
# Projection uses a semiunitary basis and transforms the manifold identically.
y=np.array(GEO['array_y_local_m'][:3]);d=np.sqrt(3.65**2+y*y);a0=np.exp(-2j*np.pi*5.23e9*(d-d[0])/299792458.);W=null_space(a0.conj()[None,:]).conj().T;ck('projection_noise_identity',np.linalg.norm(W@W.conj().T-np.eye(2)),1e-12);ck('projection_known_direct_null',np.linalg.norm(W@a0),1e-12)
a=np.exp(2j*np.pi*5.23e9*y*np.sin(np.deg2rad(30))/299792458.);x=np.array([2*a0+(.3+.2j)*a]);ck('projected_signal_model_agrees',np.linalg.norm(np.einsum('am,nm->na',W,x)-(.3+.2j)*(W@a)[None,:]),1e-12)
pd.DataFrame(checks).to_csv(R/'numerical_controls.csv',index=False);assert all(r['passed'] for r in checks);print(pd.DataFrame(checks).to_string(index=False))
