"""Explicit deep-fade / insufficient-frequency negative controls, no real-data tuning."""
import json
import numpy as np
from core import *

def main():
    tests=[]
    zero=np.zeros((128,3,K),complex);gate=common_gate(zero);out,p=correct(zero,'C',gate['common_model_pass'])
    tests.append(dict(case='all_features_faded',gate=gate['common_model_pass'],accepted=int(p['accepted'].sum()),unchanged=bool(np.array_equal(out,zero)),passed=not gate['common_model_pass'] and not p['accepted'].any() and np.array_equal(out,zero)))
    ref=np.zeros((3,K),complex);ref[0,2]=1;ref[0,3]=1
    x=apply(ref,-12.,-.8);t,ph,res=fit_wls(x,ref,12.)
    tests.append(dict(case='only_two_usable_frequencies',wls_nonfinite=bool(not np.isfinite(t)),passed=bool(not np.isfinite(t) and not np.isfinite(ph) and np.isinf(res)),note='WLS rejects insufficient support; a high ML correlation on sparse support alone is not identifiability evidence'))
    d=dict(config_sha256=__import__('hashlib').sha256((R/'config.json').read_bytes()).hexdigest(),tests=tests,all_pass=all(t['passed'] for t in tests))
    (R/'results/synthetic_fades.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print(json.dumps(d,indent=2));assert d['all_pass']
if __name__=='__main__':main()
