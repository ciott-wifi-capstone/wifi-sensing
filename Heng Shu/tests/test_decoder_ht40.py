"""Independent synthetic PicoScenes frame fixtures. No real captures included."""
from pathlib import Path
import importlib.util,struct,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_ht40',ROOT/'research/output/static_human_aoa_validation_20261101/provenance/decoder_ht40.py')
dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)
def segment(name,version,payload):
    n=name.encode('ascii');length=1+len(n)+2+len(payload)
    return struct.pack('<IB',length,len(n))+n+struct.pack('<H',version)+payload
def frame(order=(2,0,1),magic=0x20150315,bandwidth=40):
    real=np.arange(90).reshape(3,30)-45;imag=35-np.arange(90).reshape(3,30)
    bits=0
    for k in range(30):
        for j in range(3):
            bit=51*k+3+16*j
            bits|=(int(real[j,k])&255)<<bit
            bits|=(int(imag[j,k])&255)<<(bit+8)
    rx=bytearray(33);struct.pack_into('<HQQhhh',rx,0,0x5300,12345,987654321,5230,5240,bandwidth)
    rx[24]=1;struct.pack_into('<H',rx,25,bandwidth);rx[30:33]=bytes((1,0,3))
    cs=bytearray(231);struct.pack_into('<HBH',cs,0,0x5300,1,bandwidth);struct.pack_into('<H',cs,25,30)
    cs[27:30]=bytes((1,3,0));struct.pack_into('<H',cs,30,1);cs[32]=sum(x<<(2*j) for j,x in enumerate(order))
    struct.pack_into('<I',cs,35,192);cs[39:231]=bits.to_bytes(192,'little')
    body=segment('RxSBasic',4,rx)+segment('CSI',3,cs)
    raw=struct.pack('<IIHB',7+len(body),magic,1,2)+body
    return raw,(real+1j*imag)[np.argsort(order)]
class DecoderTests(unittest.TestCase):
    def parse(self,blob):
        with tempfile.TemporaryDirectory(prefix='wifi-synthetic-frame-') as d:
            p=Path(d)/'synthetic.csi';p.write_bytes(blob);return dec.decode(p)
    def test_signed_values_and_permutation(self):
        blob,truth=frame();c,host,nic=self.parse(blob)
        np.testing.assert_array_equal(c[0],truth)
        self.assertEqual(host.tolist(),[987654321]);self.assertEqual(nic.tolist(),[12345])
    def test_identity_chain_order(self):
        blob,truth=frame((0,1,2));c,_,_=self.parse(blob);np.testing.assert_array_equal(c[0],truth)
    def test_invalid_magic_rejected(self):
        with self.assertRaises(AssertionError):self.parse(frame(magic=0)[0])
    def test_incompatible_band_rejected(self):
        with self.assertRaises(AssertionError):self.parse(frame(bandwidth=20)[0])
    def test_invalid_permutation_rejected(self):
        with self.assertRaises(AssertionError):self.parse(frame(order=(0,0,1))[0])
    def test_truncation_rejected(self):
        with self.assertRaises((AssertionError,struct.error)):self.parse(frame()[0][:-1])
if __name__=='__main__':unittest.main(verbosity=2)
