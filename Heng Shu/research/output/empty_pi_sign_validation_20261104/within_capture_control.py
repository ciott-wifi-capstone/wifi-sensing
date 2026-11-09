from validate_sign import *
def main():
 rows=[]
 for cap,idx in [('E1',78),('E2',81),('E3',84),('E4',87)]:
  for state in ST:
   ref=template(load(idx,state)['x'])
   for off in [1,2]:
    _,g=examine(load(idx+off,state)['x'],ref)
    rows.append(dict(capture=cap,run=off+1,state=state,**g))
 df=pd.DataFrame(rows);df.to_csv(O/'within_capture_common_gate_control.csv',index=False)
 print(df.groupby(['capture','run']).common_model_pass.agg(['sum','count']).to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
