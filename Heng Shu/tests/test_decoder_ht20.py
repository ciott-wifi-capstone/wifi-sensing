"""Synthetic fixed Intel HT20 records; no collected CSI or metadata."""
from pathlib import Path
import importlib.util,struct,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_ht20',ROOT/'research/output/static_human_aoa_validation_20261101/provenance/decoder_ht20.py')
dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)
def record(order=(1,2,0),nic=12345,count=10):
    real=np.arange(90).reshape(3,30)-45;imag=20-np.arange(90).reshape(3,30);bits=0
    for k in range(30):
        for j in range(3):
            bit=51*k+3+16*j;bits|=(int(real[j,k])&255)<<bit;bits|=(int(imag[j,k])&255)<<(bit+8)
    h=bytearray(20);struct.pack_into('<IH',h,0,nic,count);h[8]=3;h[9]=1;h[15]=sum(x<<(2*j) for j,x in enumerate(order));h[16]=192
    return bytes((0,213,187))+h+bits.to_bytes(192,'little'),(real+1j*imag)[np.argsort(order)]
class DecoderTests(unittest.TestCase):
    def parse(self,blob):
        with tempfile.TemporaryDirectory(prefix='wifi-synthetic-record-') as d:
            p=Path(d)/'synthetic.dat';p.write_bytes(blob);return dec.decode(p)
    def test_signed_values_and_permutation(self):
        blob,truth=record();out=self.parse(blob);np.testing.assert_array_equal(out[0][0],truth)
    def test_nic_counter_wrap(self):
        a,_=record(nic=2**32-50,count=65535);b,_=record(nic=50,count=0);out=self.parse(a+b)
        self.assertEqual(out[3].tolist(),[100]);self.assertEqual(out[4].tolist(),[1]);self.assertAlmostEqual(out[2][1],.0001)
    def test_bad_frame_length(self):
        with self.assertRaises(AssertionError):self.parse(record()[0][:-1])
    def test_bad_record_header(self):
        b=bytearray(record()[0]);b[2]=0
        with self.assertRaises(AssertionError):self.parse(b)
    def test_bad_permutation(self):
        with self.assertRaises(AssertionError):self.parse(record(order=(0,0,1))[0])
if __name__=='__main__':unittest.main(verbosity=2)
