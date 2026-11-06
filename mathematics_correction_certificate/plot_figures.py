"""Render all six manuscript figures with consistent typography.
Uses saved CSVs; Figure 1 recalculates only its declared scalar fixed points.
Run after reproduction scripts to restore the publication figure style.
No archived CSV/JSON or simulation code is modified.
"""
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'lines.linewidth':2,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';PURPLE='#CC79A7'
def read(path):return np.genfromtxt(P/path,delimiter=',',names=True)
def finish(fig,path):fig.savefig(P/path,bbox_inches='tight');plt.close(fig)
# Figure 1: same scalar recurrence and convergence criterion as reproduce.py.
ms=np.arange(1,9);zs=[]
for m in ms:
 x=0.;d=2-.5*m
 for _ in range(20000):
  x=.2*x+.4*np.tanh(x)+d
  if abs(.2*x+.4*np.tanh(x)+d-x)<1e-10:break
 zs.append(x)
fig,ax=plt.subplots(figsize=(7.2,3.6),layout='constrained')
ax.plot(ms,expit(zs),'o-',color=BLUE,label='Numerical equilibrium')
ax.axvline(4,color=ORANGE,ls='--',label='Zero-drive boundary: m = 4')
ax.axhline(.5,color='.4',ls=':',label='Decision boundary: p = 1/2')
ax.set(xlabel='Source multiplicity m',ylabel=r'Equilibrium probability $p^*$',ylim=(0,1),xticks=ms)
ax.legend(loc='lower left');ax.grid(alpha=.15);finish(fig,'duplication.pdf')
# Figure 2.
r=read('reproduction/recovery.csv');fig,ax=plt.subplots(figsize=(7.2,3.7),layout='constrained')
ax.plot(r['t'],expit(r['no_reference']),color=BLUE,label='No reference')
ax.plot(r['t'],expit(r['reference']),color=ORANGE,ls='--',label='Reference from t = 160')
ax.axvline(80,color='.4',ls=':',label='Deduplication: t = 80');ax.axvline(160,color='.4',ls='-.',label='Reference introduced: t = 160')
ax.axhline(.5,color='.6',lw=1);ax.set(xlabel='Iteration t',ylabel=r'Probability $p_t$',ylim=(0,1),xlim=(0,300));ax.legend(loc='upper left',ncol=2,fontsize=9);finish(fig,'recovery.pdf')
# Figure 3: measured grid and separate analytic fold, with correctly centered coordinates.
r=read('reproduction/grid.csv');xs=np.unique(r['lambda']);ys=np.unique(r['e']);prob=expit(r['z'].reshape(len(ys),len(xs)))
fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained')
mesh=ax.pcolormesh(xs,ys,prob,shading='nearest',vmin=0,vmax=1,cmap='viridis',rasterized=True)
fig.colorbar(mesh,ax=ax,label=r'Numerical equilibrium probability $p^*$')
cs=ax.contour(xs,ys,prob,levels=[.5],colors=[ORANGE],linewidths=2)
q=np.arccosh(np.sqrt(1.2/.8));dc=1.2*np.tanh(q)-.8*q
ee=np.linspace(0,.999,1000);ll=(dc-.1)/(1-ee);ok=ll<=.8
ax.plot(ll[ok],ee[ok],color='white',ls='--',lw=2,path_effects=[pe.Stroke(linewidth=3.2,foreground='black'),pe.Normal()])
ax.axhline(1,color='white',ls=':',lw=1.3)
ax.legend(handles=[Line2D([],[],color='black',ls='--',label='Theoretical fold'),Line2D([],[],color=ORANGE,label=r'Numerical $p^*=1/2$ contour'),Line2D([],[],color='.5',ls=':',label='Zero reference: e = 1')],loc='upper right',framealpha=.95,fontsize=9)
ax.set(xlabel=r'Reference strength $\lambda$',ylabel='Adverse reference bias e',xlim=(0,.8),ylim=(0,1.5));finish(fig,'reference_grid.pdf')
# Figure 4.
r=read('heterogeneous.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
for j,(ax,limit,title) in enumerate(zip(axes,[100,6],['(a) Full trajectory','(b) Initial correction'])):
 for k in range(1,21):ax.plot(r['t'],r[f'z_{k}'],color='.8',lw=.8)
 ax.plot(r['t'],r['minimum_signed_state'],color=BLUE,label='Minimum signed state');ax.plot(r['t'],r['lower_bound'],color=ORANGE,ls='--',label='Theoretical lower bound')
 ax.axhline(0,color='.4',lw=1);ax.set(xlim=(0,limit),xlabel='Iteration t',title=title)
axes[0].set_ylabel('Signed state (h = 1)');axes[1].legend(loc='lower right',fontsize=9.5);finish(fig,'heterogeneous_bound.pdf')
# Figure 5.
r=read('revision_results/sensitivity.csv');fig,axes=plt.subplots(2,3,figsize=(9.2,6.1),layout='constrained')
for j,(name,label) in enumerate([('a','a'),('gamma',r'$\gamma$'),('kappa',r'$\kappa$')]):
 # load text column separately because genfromtxt default would turn sweep names into NaN
 import csv
 rows=[x for x in csv.DictReader((P/'revision_results/sensitivity.csv').open()) if x['sweep']==name and float(x['input_margin'])==.05]
 x=np.array([float(z['value']) for z in rows]);dc=np.array([float(z['dc']) for z in rows])
 axes[0,j].plot(x,dc,color=BLUE);axes[0,j].set(xlabel=label,ylabel=r'Theoretical threshold $d_c$',title=f'({chr(97+j)}) Threshold')
 for key,color,ls,label2 in [('network_crossing',BLUE,'-','Network'),('envelope_crossing',ORANGE,'--','Scalar envelope')]:axes[1,j].plot(x,[int(z[key]) for z in rows],color=color,ls=ls,marker='o',ms=2.5,label=label2)
 axes[1,j].set(xlabel=label,ylabel='First positive iteration',title=f'({chr(100+j)}) '+r'$D=d_c+0.05$')
 for ax in axes[:,j]:
  ax.grid(alpha=.2);ax.tick_params(labelsize=12);ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14);ax.title.set_size(13)
axes[1,0].legend(fontsize=11);finish(fig,'revision_results/sensitivity.pdf')
# Figure 6.
fig,axes=plt.subplots(1,2,figsize=(7.2,3.9),layout='constrained')
for name,color,ls in [('periodic',BLUE,'-'),('long_dwell',ORANGE,'--'),('random',GREEN,':'),('disconnected',PURPLE,'-.')]:
 r=read(f'revision_results/switching_{name}.csv');sel=r['t']<=30
 axes[0].plot(r['t'][sel],r['minimum_signed_state'][sel],color=color,ls=ls,label=name.replace('_',' '))
 axes[1].plot(r['t'][sel],r['brier'][sel],color=color,ls=ls,label=name.replace('_',' '))
axes[0].plot(r['t'][sel],r['envelope'][sel],color='black',ls='--',label='Scalar lower envelope');axes[0].axhline(0,color='.5',lw=.8)
axes[1].plot(r['t'][sel],r['brier_bound'][sel],color='black',ls='--',label='Brier upper envelope')
axes[0].set(xlabel='Iteration t',ylabel='Minimum signed state',title='(a) State correction');axes[1].set(xlabel='Iteration t',ylabel='Mean Brier loss',title='(b) Loss bound')
for ax in axes:ax.legend(fontsize=9.5)
finish(fig,'revision_results/switching.pdf')
print('Rendered all six figures from the saved results and declared scalar recurrence.')
