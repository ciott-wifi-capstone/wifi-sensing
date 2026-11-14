"""Vector diagrams and measured closure plot for supervisor review."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from common import *
TH=ROOT/'output/thesis_algorithm_draft_20261019';F=TH/'figures'
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,name):
 fig.savefig(F/f'{name}.pdf');fig.savefig(R/'review'/f'{name}.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));ax.set(xlim=(-.6,6.6),ylim=(-2,2));ax.axis('off')
pos={'G1':(0,0),'C1':(2,0),'G2':(3,1.5),'G3':(3,-1.5),'C2':(4,0),'G4':(6,0)}
edges=[('G1','C1','e0: ant 1',0),('G2','C1','e1: ant 6',0),('G2','C2','e2: ant 6',0),('G3','C1','e3: ant 8',0),('G3','C2','e4: ant 8',0),('G4','C2','e5: ant 10',0),('C1','C2','e6: ant 6',.27),('C1','C2','e7: ant 8',-.27)]
for s,t,label,rad in edges:
 a=np.array(pos[s]);b=np.array(pos[t]);patch=FancyArrowPatch(a,b,connectionstyle=f'arc3,rad={rad}',arrowstyle='-',color='#47677f',lw=1.5,shrinkA=20,shrinkB=20);ax.add_patch(patch);mid=(a+b)/2
 if rad:mid[1]=-.4 if rad>0 else .4
 else:mid[1]+=.12 if mid[1]>=0 else -.12
 ax.text(*mid,label,ha='center',va='center',fontsize=8.7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
for s,(x,y) in pos.items():ax.scatter(x,y,s=1050,c='#dceaf3',edgecolors='#1f4663',zorder=5);ax.text(x,y,s,ha='center',va='center',zorder=6)
ax.text(0,-.6,'leaf',ha='center',fontsize=9);ax.text(6,-.6,'leaf',ha='center',fontsize=9);save(fig,'state_graph')

fig,ax=plt.subplots(figsize=(9,10.5));ax.set(xlim=(0,10),ylim=(0,12));ax.axis('off');colors={'A':'#e3efe7','B':'#e3edf5','C':'#fff1d8','stop':'#f3e0dc'}
def box(x,y,w,h,text,status):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',fc=colors[status],ec='#3c5060',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.2,linespacing=1.35)
def arrow(a,b,dashed=False):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#405568','lw':1.2,'linestyle':'--' if dashed else '-'})
box(.5,10.85,9,.75,'[A] Source hashes, role split, corrected geometry\n39 primary records; Office separate; empty is an absence label','A')
box(.5,9.6,9,.75,'[A/B] Decode CSI, reorder chains, map guarded state windows\nPreserve short records, missing states and quality flags','A');arrow((5,10.77),(5,10.43))
box(.2,7.8,4.3,1.15,'[A/B] Gated common phase / timing\nA raw; B phase-only; C per-record\nD frozen Office reference only\nFailed correction retains A','B');arrow((2.4,9.52),(2.4,9.03))
box(5.2,7.8,4.3,1.15,'[B] Shared-antenna graph diagnostics\nClosure + frozen transfer prediction\n[C] Independent hardware calibration\nRequired before coherent assembly','B');arrow((7.4,9.52),(7.4,9.03))
box(.2,5.9,4.3,1.15,'[B] State-wise 2D MUSIC\nCorrect tones, antenna map and order\nFixed local maxima; unassigned paths\nQuality-weighted angular fusion','B');arrow((2.4,7.72),(2.4,7.13))
box(5.2,5.9,4.3,1.15,'[B] Existing empty covariance audit\n[C] Validated background model\nSame-session empty-human-empty\nData / steering / noise consistency','C');arrow((7.4,7.72),(7.4,7.13),True)
box(.2,3.9,4.3,1.1,'[B] Current release boundary\nRetain diagnostics and rejection ledger\nNo calibrated human direction\nNo human-angle or position error','stop');arrow((2.4,5.82),(2.4,5.08));arrow((5.12,4.45),(4.58,4.45),True)
box(5.2,3.9,4.3,1.1,'[C] Human + direct-path validation\nBackground null and repeatability tests\nSame-record excess delay only\nFailed identity/reference: stop release','C');arrow((7.4,5.82),(7.4,5.08),True);arrow((4.58,6.5),(5.12,6.5),True)
box(.2,1.7,4.3,1.15,'[C] Calibrated direction branch\nNoncoherent fusion if paths agree\nNo full-aperture coherent gain\nReport coverage and ambiguity','C');arrow((6.2,3.82),(2.4,2.93),True)
box(5.2,1.7,4.3,1.15,'[C] Joint-array / position branch\nAll calibration + coherence gates\nCompatible full-frequency spherical model\nValidated AoA + excess-delay path','C');arrow((7.4,3.82),(7.4,2.93),True)
ax.text(5,.9,'Status A: tested software/data checks   B: implemented diagnostics   C: pending physical branch',ha='center',fontsize=8.6)
ax.text(5,.45,'Solid links: implemented analysis     Dashed links: validation-dependent integration',ha='center',fontsize=9)
fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);save(fig,'algorithm_flow')

cl=pd.concat([pd.read_csv(R/'results'/f'graph_{s}_closure.csv') for s in ['development','evaluation']]);cl=cl[(cl.stage=='evaluation')&(cl.loop=='C1_C2_shared6_vs8')]
fig,axs=plt.subplots(1,2,figsize=(9,3.7))
for key,g in cl.groupby('index'):
 for ax,col in zip(axs,['phase_abs_deg','delay_abs_ns']):ax.plot(g.cycle,g[col],lw=.9,label=str(key))
for ax,label,gate in zip(axs,['Closure phase discrepancy (deg)','Closure delay discrepancy (ns)'],[15,5]):ax.axhline(gate,c='k',ls='--',lw=.8);ax.set(xlabel='Observed complete switch cycle',ylabel=label);ax.grid(alpha=.2)
axs[0].legend(ncol=3,fontsize=8,title='Evaluation record');fig.tight_layout();save(fig,'graph_closure_new')
print('Updated vector workflow, complete graph, and measured closure plot.')
