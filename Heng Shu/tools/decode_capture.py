"""Read one trusted private capture with the observed-format decoder only."""
from pathlib import Path
import argparse,importlib.util,sys,struct
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 if not __debug__:raise SystemExit('Do not use Python -O: archived parser format checks require assertions.')
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--format',choices=['ht20','ht40'],required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if not a.input.is_file():ap.error('input file does not exist')
 if a.output.suffix.lower()!='.npz':ap.error('output must have .npz extension')
 if a.output.exists():ap.error('output exists; choose a new file')
 src=ROOT/'research/output/static_human_aoa_validation_20261101/provenance'/('decoder_'+a.format+'.py')
 sp=importlib.util.spec_from_file_location('decoder',src);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 try:values=m.decode(a.input)
 except (AssertionError,IndexError,ValueError,struct.error) as e:raise SystemExit('Unsupported or invalid capture; no output written. '+str(e))
 if a.format=='ht40':arrays=dict(csi=values[0],host_ns=values[1],nic_timestamp=values[2])
 else:arrays=dict(csi=values[0],header=values[1],time_s=values[2],nic_delta_us=values[3],counter_delta=values[4],antenna_permutation=values[5])
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arrays)
 print('Decoded '+str(len(values[0]))+' frames. Output contains private measurements; do not commit it.')
if __name__=='__main__':main()
