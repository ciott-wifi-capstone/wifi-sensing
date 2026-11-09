"""Audited adapter invoking the unchanged local third-party Widar2 SAGE core."""
import ast,builtins,copy,time,itertools
from pathlib import Path
import numpy as np
C=299792458.;FC=5.230e9;DF=np.arange(-58,59,4)*312500.
MAP={'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]}
def coords(group):return (np.array(MAP[group])-6.5)*.028

def load_core(path):
    keep=['sage_const_build','sage_signal','sage_expectation','sage_maximization','sage_sfg'];tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in keep];assert len(nodes)==5
    ns={'np':np,'copy':copy,'time':time,'int':lambda x:builtins.int(np.asarray(x).ravel()[0])};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    calls={name:0 for name in keep}
    for name in keep:
        original=ns[name]
        def wrap(*args,_f=original,_name=name,**kw):calls[_name]+=1;return _f(*args,**kw)
        ns[name]=wrap
    inventory=[dict(function=n.name,line=n.lineno,end_line=n.end_lineno,extracted=n.name in keep) for n in tree.body if isinstance(n,ast.FunctionDef)]
    return ns,calls,inventory

def steering(group,policy):
    aa=np.arange(*[policy['angle_grid_deg'][0],policy['angle_grid_deg'][1]+.001,policy['angle_grid_deg'][2]])
    tt=np.arange(*[policy['delay_grid_ns'][0],policy['delay_grid_ns'][1]+.001,policy['delay_grid_ns'][2]])*1e-9
    y=coords(group);y-=y.mean()
    A=np.exp(2j*np.pi*FC/C*y[:,None]*np.sin(np.radians(aa))[None,:]);F=np.exp(-2j*np.pi*DF[:,None]*tt[None,:])
    return aa,tt,A,F

def fit(h,group,core,policy):
    aa,tt,A,F=steering(group,policy);L=policy['paths'];con=core['sage_const_build'](1,30,3,L,policy['max_iterations'],1.25e6,0,.028*FC/C,tt[None],np.radians(aa)[None],np.array([[0.]]),1)
    con.update(tof_candidates=F,aoa_candidates=A,doppler_candidates=np.ones((1,1),complex));initial_energy=float(np.sum(abs(h)**2))
    best=None;startresults=[]
    for start in range(policy['deterministic_starts']):
        par=np.zeros((4,L),complex);ix=np.zeros((3,L),int);res=h.copy()
        for ell in range(L):
            match=(A.conj().T@res.T)@F.conj();power=abs(match)**2
            if start and ell==0:
                # Fixed alternate initialization, not target-coordinate based.
                ia0,it0=np.unravel_index(power.argmax(),power.shape)
                mask=(abs(aa-aa[ia0])[:,None]<12)&(abs((tt-tt[it0])*1e9)[None,:]<8);power[mask]=-1
            ia,it=np.unravel_index(power.argmax(),power.shape);atom=F[:,it,None]*A[:,ia][None,:];amp=np.vdot(atom,res)/90
            par[:,ell]=[tt[it],np.radians(aa[ia]),0,amp];ix[:,ell]=[it,ia,0];res-=core['sage_signal'](par[:,ell],ix[:,ell],con)[0]
        initial_res=float(np.sum(abs(res)**2)/max(initial_energy,1e-30))
        par,_=core['sage_sfg'](h[None],par,ix,con,np.array([.1e-9,np.radians(.1),0,1e-8]),4,4)
        pred=sum((par[3,l]*np.exp(-2j*np.pi*DF[:,None]*par[0,l].real)*np.exp(2j*np.pi*FC/C*(coords(group)-coords(group).mean())[None,:]*np.sin(par[1,l].real)) for l in range(L)),np.zeros_like(h))
        error=float(np.sum(abs(h-pred)**2)/max(initial_energy,1e-30));startresults.append(error)
        if best is None or error<best[1]:best=(par.copy(),error,initial_res,start)
    par,error,initial_res,start=best;order=np.argsort(-abs(par[3])**2,kind='stable')
    return par[:,order],dict(relative_residual_energy=error,initial_relative_residual_energy=initial_res,winning_start=start,start_residual_energy=startresults)

def locate(angle_deg,excess_ns,group):
    yc=coords(group).mean();b=np.array([3.65,-yc]);d=np.linalg.norm(b);S=d+C*excess_ns*1e-9;n=np.array([np.cos(np.radians(angle_deg)),np.sin(np.radians(angle_deg))]);den=2*(S-b@n)
    if excess_ns<=0 or den<=1e-12:return dict(x_m=np.nan,y_m=np.nan,rx_range_m=np.nan,total_ToF_ns=np.nan,valid_geometry=False)
    r=(S*S-d*d)/den
    return dict(x_m=float(r*n[0]),y_m=float(yc+r*n[1]),rx_range_m=float(r),total_ToF_ns=float(S/C*1e9),valid_geometry=bool(r>=0))
