from models import *
import struct,hashlib
m=pd.read_csv(S/'input_manifest.csv');rows=[]
for key in DEV+EVAL:
 r=m[m['index']==key].iloc[0];path=Path(r.path);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r.sha256;pos=0;ants=[]
 while pos<len(raw):
  size,magic,ver,ns=struct.unpack_from('<IIHB',raw,pos);assert magic==0x20150315 and ver==1;end=pos+size+4;s=pos+11
  for _ in range(ns):
   length,n=struct.unpack_from('<IB',raw,s);name=raw[s+5:s+5+n].rstrip(b'\0').decode();b=s+7+n
   if name=='CSI':ants.append(raw[b+32])
   s+=length+4
  pos=end
 ants=np.array(ants);perm=np.stack([(ants>>(2*j))&3 for j in range(3)],1);assert np.all(np.sort(perm,axis=1)==[0,1,2])
 for state in ST:
  z=np.load(S/'packets'/f'{key}_{state}.npz');chosen=ants[z['packet_id']]
  for code,n in zip(*np.unique(chosen,return_counts=True)):rows.append(dict(record=key,state=state,permutation_byte=int(code),selected_packets=int(n),all_frame_permutations_valid=True,raw_sha_verified=True,decoder_action='raw slot to physical port via argsort(perm); unchanged',physical_wiring_independently_verified=False))
pd.DataFrame(rows).to_csv(R/'permutation_audit.csv',index=False);print(pd.DataFrame(rows).groupby('record').agg(n_codes=('permutation_byte','nunique'),selected_packets=('selected_packets','sum')).to_string())
