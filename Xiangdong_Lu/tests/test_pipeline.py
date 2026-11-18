"""Numerical/data-integrity tests: synthetic fixtures are never measured data."""
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import replay as r
from wical.original.background_phase_qc import decode
from wical.original.extended_analysis import scalar_decode
from wical.estimate import fuse, profiles, inputs, PLANE
from run_example import load_covariance, estimate
from evaluate_results import evaluate


def packed_frame(samples,perm):
    frame=np.zeros(385,dtype=np.uint8)
    frame[161]=sum(p<<(2*j) for j,p in enumerate(perm))
    bit=0
    for tone in range(30):
        bit+=3
        for rx in range(3):
            z=samples[tone,rx]
            for offset,value in ((0,int(z.real)),(8,int(z.imag))):
                for k in range(8):
                    if (value & 255)>>k & 1:
                        pos=bit+offset+k
                        frame[168+pos//8]|=1<<(pos%8)
            bit+=16
    return frame


class PipelineTests(unittest.TestCase):
    def test_decoder_signed_values_all_six_antsel_permutations(self):
        values=np.array([-128,-1,0,1,127],dtype=float)
        h=np.resize(values,90).reshape(30,3)+1j*np.resize(values[::-1],90).reshape(30,3)
        for perm in itertools.permutations(range(3)):
            with self.subTest(perm=perm):
                frame=packed_frame(h,perm)
                decoded,valid=decode(frame[None,:])
                expected=np.empty_like(h)
                for rx,dest in enumerate(perm):expected[:,dest]=h[:,rx]
                self.assertTrue(valid[0])
                np.testing.assert_array_equal(decoded[0],expected)
                np.testing.assert_array_equal(decoded[0],scalar_decode(frame))

    def test_invalid_antsel_is_flagged_not_used(self):
        frame=np.zeros((2,385),np.uint8)
        frame[:,161]=[0,63]
        values,valid=decode(frame)
        self.assertFalse(valid.any())
        self.assertEqual(np.count_nonzero(values),0)

    def test_physical_rf_mapping(self):
        self.assertEqual(r.MAP,{'G1':[1,2,3],'G2':[4,5,6],'G3':[7,8,9],
                               'G4':[10,11,12],'C1':[1,8,6],'C2':[10,8,6]})

    def test_complete_public_fixture_hits_known_synthetic_angle(self):
        tables=estimate(load_covariance(ROOT/'examples/synthetic_covariance.csv'))
        rows=[x for x in tables['blind_predictions'] if x['run']=='run2'
              and x['mapping']=='cov' and x['branch'] in ('raw','background')]
        self.assertEqual(len(rows),4)
        self.assertEqual({x['method'] for x in rows},{'G3','J6_product'})
        for row in rows:self.assertEqual(row['angle_deg'],23.)
        self.assertNotIn('native_sage',tables)

    def test_fixture_missing_duplicate_and_nonhermitian_rejected(self):
        lines=(ROOT/'examples/synthetic_covariance.csv').read_text().splitlines()
        variants=[lines[:-1],lines+[lines[1]],lines.copy()]
        fields=variants[-1][2].split(',');fields[-1]=str(float(fields[-1])+1)
        variants[-1][2]=','.join(fields)
        with tempfile.TemporaryDirectory() as d:
            for i,variant in enumerate(variants):
                p=Path(d)/f'{i}.csv';p.write_text('\n'.join(variant))
                with self.subTest(i=i),self.assertRaises(ValueError):load_covariance(p)

    def test_background_and_independent_complex_state_gains(self):
        covariance=load_covariance(ROOT/'examples/synthetic_covariance.csv')
        bg=np.broadcast_to(np.eye(3)*.05,covariance.shape).copy()
        q=np.ones((6,30,3),complex)
        x=inputs(covariance,q,bg)
        p=profiles(x)
        gains=np.exp(1j*np.arange(6)) * np.arange(1,7)
        changed=profiles(x*gains[:,None,None])
        u=p/p.mean(axis=1,keepdims=True)
        v=changed/changed.mean(axis=1,keepdims=True)
        np.testing.assert_allclose(u,v,rtol=1e-12,atol=1e-12)
        self.assertEqual(int(r.ANGLES[np.argmax(p[2])]),23)
        self.assertFalse(np.array_equal(x,inputs(covariance,q,np.zeros_like(bg))))

    def test_robust_fusion_fallback_when_long_gates_fail(self):
        covariance=load_covariance(ROOT/'examples/synthetic_covariance.csv')
        x=inputs(covariance,np.ones((6,30,3),complex),np.zeros_like(covariance))
        p=profiles(x);pn=profiles(x,True)
        out,weights,_=fuse(p,pn,x,np.full(6,120),np.ones(6),np.zeros(6,bool))
        np.testing.assert_array_equal(out['J6_robust'],out['F4'])
        np.testing.assert_array_equal(out['J6_near'],out['F4_near'])
        self.assertEqual(np.count_nonzero(weights),0)

    def test_measured_metrics_match_all_three_reported_branches(self):
        rows,result=evaluate(ROOT/'results/acde')
        self.assertEqual(len(rows),24)
        expected={'raw':(29.78,24.95),'background':(19.70,7.28),
                  'kong_phase_hypothesis':(52.40,28.65)}
        for branch,(g3,six) in expected.items():
            self.assertAlmostEqual(result['conditional_mae_deg'][branch]['G3'],g3,delta=.005)
            self.assertAlmostEqual(result['conditional_mae_deg'][branch]['J6_product'],six,delta=.005)
        self.assertAlmostEqual(result['background_conditional_mae_reduction_percent'],63.,delta=.05)
        self.assertFalse(result['physically_verified_accuracy'])
        self.assertEqual({x['point'] for x in rows},{'A','B_new','C','D'})

    def test_evaluator_refuses_changed_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'tables';shutil.copytree(ROOT/'results/acde',dest)
            p=dest/'blind_predictions.csv';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):evaluate(dest)

    def test_external_native_core_is_optional_and_hash_checked(self):
        self.assertEqual(r.optional_core(None),(None,{},None))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'not_the_core.py';p.write_text('raise RuntimeError()')
            with self.assertRaisesRegex(ValueError,'audited revision'):r.optional_core(p)


if __name__=='__main__':unittest.main()
