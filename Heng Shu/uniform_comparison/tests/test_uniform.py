"""Public, data-free regression checks; no captured measurements are loaded."""
import sys,json,unittest
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import predict_uniform as p
class UniformTests(unittest.TestCase):
    def test_archived_synthetic_checks(self):
        captured={};old=p.dump
        try:
            p.dump=lambda name,obj:captured.update({name:obj})
            p.synthetic()
        finally:p.dump=old
        self.assertFalse(captured['software_checks.json']['physical_validation'])
        self.assertTrue(all(v for k,v in captured['software_checks.json'].items() if k!='physical_validation'))
    def test_geometry_and_mapping(self):
        self.assertEqual(tuple(p.ST),('G1','G2','G3','G4','C1','C2'))
        self.assertEqual(len(p.BASE.F),30)
        self.assertTrue(np.allclose(np.diff(p.BASE.Y),-.028))
        self.assertEqual(p.BASE.MAP['C1'],[1,8,6])
        self.assertEqual(p.BASE.MAP['C2'],[10,8,6])
    def test_no_output_for_empty_spectrum(self):
        angle,alts,margin=p.peaks(np.full(181,np.nan))
        self.assertTrue(np.isnan(angle));self.assertEqual(alts,[])
    def test_no_physical_release(self):
        view={};power={}
        for r in [2,3]:
            for block in ['first','last','full','Q1','Q2','Q3','Q4']:
                name=f'R{r}_{block}';view[name]=np.tile((p.ANG-12)**2+1,(3,1));power[name]=2. if r==2 else 1.
        power['R1_held']=1.
        self.assertFalse(p.assess(view,power,2)['physical_release'])
if __name__=='__main__':unittest.main(verbosity=2)
