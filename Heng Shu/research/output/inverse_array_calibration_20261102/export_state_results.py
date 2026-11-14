from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
S=R/'scaled'
d=pd.read_csv(S/'state_predictions.csv')
keys=[0,3,12,15,21,24,30,33];states=['G1','G2','G3','G4','C1','C2']
pos={0:'+1 m',3:'+1 m',12:'-1 m',15:'-1 m',21:'+2 m',24:'+2 m',30:'-2 m',33:'-2 m'}
truth={0:-26.450916,3:-26.450916,12:26.450916,15:26.450916,21:-44.857118,24:-44.857118,30:44.857118,33:44.857118}
text='# 三阵元各状态独立估计结果\n\n日期：2026-11-03。直接导出现有冻结结果，没有重新拟合或运行实验。\n\n每个状态单独计算二维MUSIC，表中为该状态二维谱的全局最大峰角度参数；不是融合结果，也没有按人体真值挑峰。未确认峰属于人体，角度未通过物理标定。网格间隔3°。时延为未校准参数，不能换算物理距离。\n\nG1=[1,2,3]；G2=[4,5,6]；G3=[7,8,9]；G4=[10,11,12]；C1=[1,8,6]；C2=[10,8,6]。G状态为等距三阵元；C状态为非等距三阵元参考组合。没有进行12阵元相干拼接。\n\n记录0/12用于共享初相拟合；3/15为同站位重复，21/24/30/33为未用于拟合的±2m位置。全部均为已查看过的回顾数据。本轮仅覆盖8条缓存记录，每状态使用前64包之后的704包。baseline已含既有条件式公共误差处理；它不是完全未经处理的原始CSI。tx_only与tx_human分别包含π分支修正和对应共享相位候选。\n'
for model,label in [('baseline','基线'),('tx_only','仅TX约束共享相位'),('tx_human','TX＋人体联合共享相位')]:
 p=d[d.model==model].pivot(index='record',columns='state',values='angle_deg').loc[keys,states]
 text+='\n## '+label+'：角度 / °\n\n|记录|站位|用途|理论人体角度|G1|G2|G3|G4|C1|C2|\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for k,row in p.iterrows():
  role='标定' if k in [0,12] else '重复验证' if k in [3,15] else '未拟合位置'
  text+='|'+f'{k}|{pos[k]}|{role}|{truth[k]:+.2f}|'+'|'.join(f'{v:+.0f}' for v in row)+'|\n'
 text+='\n'
 print(model);print(p.to_string())
text+='\n## 使用提醒\n\n不能从六组结果里挑出最接近真值的一组作为最终结果；不能把C组的空间混叠峰自动当作人体。完整CSV保留所有8个处理分支、每状态角度和时延参数。各状态时延参考未统一，不能直接把参数差当作人体超额时延。三阵元独立估计只能利用该子阵的孔径；非相干融合不等于恢复完整12阵元的相干孔径。\n'
(R/'三阵元逐状态结果_20261103.md').write_text(text,encoding='utf8')
d['position_label']=d.record.map(pos);d['reference_human_angle_deg']=d.record.map(truth)
d.to_csv(R/'三阵元逐状态全部结果_20261103.csv',index=False,encoding='utf-8-sig')
fig,axs=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
for ax,model,title in zip(axs,['baseline','tx_only','tx_human'],['Baseline','Shared phase: TX only','Shared phase: TX + human']):
 p=d[d.model==model].pivot(index='record',columns='state',values='angle_deg').loc[keys,states]
 im=ax.imshow(p.to_numpy(),vmin=-90,vmax=90,cmap='coolwarm',aspect='auto')
 ax.set_xticks(range(6),states);ax.set_title(title)
 ax.set_yticks(range(8),[f'{k}: {pos[k]}'+(' (cal)' if k in [0,12] else '') for k in keys])
 for i in range(8):
  for j in range(6):ax.text(j,i,f'{p.iloc[i,j]:+.0f}',ha='center',va='center',fontsize=8,color='white' if abs(p.iloc[i,j])>60 else 'black')
fig.suptitle('Independent three-element state peaks (deg); paths unassigned',fontsize=12)
fig.subplots_adjust(left=.12,right=.91,top=.85,bottom=.12,wspace=.13)
fig.colorbar(im,cax=fig.add_axes([.93,.16,.015,.64]),label='Angle parameter (deg)')
fig.savefig(R/'三阵元逐状态对照_20261103.png',dpi=170)
fig.savefig(R/'三阵元逐状态对照_20261103.pdf')
