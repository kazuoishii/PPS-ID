"""Step-2 experiments. Paired episode t intervals; deterministic parameter and switching checks.
Run from any directory; outputs are confined to revision_results/ next to this script.
"""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t as student_t
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revision_results';OUT.mkdir(exist_ok=True)

def write_csv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def threshold(a,g,k):
 if g*k<=1-a:return 0.
 q=np.arccosh(np.sqrt(g*k/(1-a)))/k
 return float(g*np.tanh(k*q)-(1-a)*q)

paired=[]
for graph in ['Complete','Cycle','Modular','Cycle_reverse']:
 x=np.genfromtxt(ROOT/'reproduction'/f'{graph.lower()}_episodes.csv',delimiter=',',names=True)
 for metric in ['accuracy','brier']:
  diff=x[f'{metric}_reuse']-x[f'{metric}_unique'];n=len(diff)
  mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(n));half=float(student_t.ppf(.975,n-1)*se)
  paired.append(dict(graph=graph,metric=metric,n=n,mean_difference=mean,paired_se=se,ci95_low=mean-half,ci95_high=mean+half))
write_csv('paired_ci.csv',paired)
n=20
complete=np.ones((n,n))/n;cycle=np.roll(np.eye(n),-1,axis=1)
modular=np.ones((n,n))*.1/n
modular[:10,:10]+=.9/10;modular[10:,10:]+=.9/10
block=np.zeros((n,n));block[:10,:10]=.1;block[10:,10:]=.1
identity=np.eye(n)
graphs=[complete,cycle,modular,block,identity]
for W in graphs:assert np.all(W>=0) and np.allclose(W.sum(axis=1),1)

def run(a,g,k,D,schedule,steps,retain=False):
 z=np.linspace(-8,2,n);v=-8.;rows=[];first=None;vfirst=None;mingap=0.;maxlossgap=0.
 for t in range(steps+1):
  gap=float(z.min()-v);loss=float(np.mean(expit(-z)**2));bound=float(expit(-v)**2)
  assert gap>=-1e-10 and loss<=bound+1e-10
  mingap=min(mingap,gap);maxlossgap=max(maxlossgap,loss-bound)
  if z.min()>0 and first is None:first=t
  if v>0 and vfirst is None:vfirst=t
  if first is not None:assert z.min()>0
  if vfirst is not None:assert v>0
  if retain:rows.append(dict(t=t,minimum_signed_state=float(z.min()),envelope=float(v),brier=loss,brier_bound=bound,graph_index=int(schedule[t]) if t<steps else -1))
  if t<steps:
   d=D+.05*(1+np.sin(.17*t+2*np.pi*np.arange(n)/n))
   z=a*z+g*(graphs[schedule[t]]@np.tanh(k*z))+d
   v=a*v+g*np.tanh(k*v)+D
 cert=int(np.floor(8/(D-threshold(a,g,k))))+1
 assert first is not None and vfirst is not None and first<=cert and vfirst<=cert
 return dict(network_crossing=first,envelope_crossing=vfirst,certified_time=cert,min_state_gap=mingap,max_brier_excess=maxlossgap),rows

sensitivity=[]
sweeps=[('a',np.linspace(0,.8,33)),('gamma',np.linspace(.2,2,37)),('kappa',np.linspace(.3,2,35))]
for name,values in sweeps:
 for value in values:
  a,g,k=.2,1.2,1.
  if name=='a':a=float(value)
  elif name=='gamma':g=float(value)
  else:k=float(value)
  dc=threshold(a,g,k)
  for margin in [.01,.05,.2]:
   D=dc+margin;steps=int(np.floor(8/margin))+3
   result,_=run(a,g,k,D,np.arange(steps)%3,steps)
   sensitivity.append(dict(sweep=name,value=float(value),a=a,gamma=g,kappa=k,dc=dc,input_margin=margin,D=D,**result))
write_csv('sensitivity.csv',sensitivity)

rng=np.random.default_rng(20261103);steps=300
times=np.arange(steps)
schedules={'periodic':times%3,'long_dwell':(times//40)%3,'random':rng.integers(0,3,steps),'disconnected':np.where((times//25)%2==0,3,4)}
switching=[]
for name,seq in schedules.items():
 result,rows=run(.2,1.2,1.,.3,seq,steps,True)
 switching.append(dict(schedule=name,steps=steps,**result))
 write_csv(f'switching_{name}.csv',rows)
write_csv('switching_summary.csv',switching)

plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9,'lines.linewidth':1.8,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(10,5.8),layout='constrained')
for j,(name,_) in enumerate(sweeps):
 subset=[r for r in sensitivity if r['sweep']==name and r['input_margin']==.05]
 x=[r['value'] for r in subset]
 axes[0,j].plot(x,[r['dc'] for r in subset],label='Theoretical threshold')
 axes[0,j].set(xlabel={'a':'a','gamma':r'$\gamma$','kappa':r'$\kappa$'}[name],ylabel=r'$d_c$',title='One parameter varied')
 axes[1,j].plot(x,[r['network_crossing'] for r in subset],label='Network')
 axes[1,j].plot(x,[r['envelope_crossing'] for r in subset],ls='--',label='Scalar envelope')
 axes[1,j].set(xlabel=axes[0,j].get_xlabel(),ylabel='First positive crossing',title=r'$D=d_c+0.05$')
 for ax in axes[:,j]:ax.grid(alpha=.2)
axes[0,0].legend();axes[1,0].legend()
fig.savefig(OUT/'sensitivity.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for name in schedules:
 x=np.genfromtxt(OUT/f'switching_{name}.csv',delimiter=',',names=True)
 axes[0].plot(x['t'][:31],x['minimum_signed_state'][:31],label=name.replace('_',' '))
axes[0].plot(x['t'][:31],x['envelope'][:31],color='black',ls='--',label='Scalar envelope')
axes[0].axhline(0,color='gray',lw=.8);axes[0].set(xlabel='Iteration',ylabel='Minimum signed state');axes[0].legend()
for name in schedules:
 x=np.genfromtxt(OUT/f'switching_{name}.csv',delimiter=',',names=True)
 axes[1].plot(x['t'][:31],x['brier'][:31],label=name.replace('_',' '))
axes[1].plot(x['t'][:31],x['brier_bound'][:31],color='black',ls='--',label='Brier upper envelope')
axes[1].set(xlabel='Iteration',ylabel='Mean Brier loss');axes[1].legend()
fig.savefig(OUT/'switching.pdf');plt.close(fig)
summary={'paired_interval_method':'Two-sided paired episode Student t intervals, df=999; marginal, conditional on saved mask/graphs; not adjusted for multiplicity. Agents and graphs are not independent replicates.','sensitivity_runs':len(sensitivity),'sensitivity_all_checks_passed':True,'switching':switching,'seed':20261103}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('PAIRED',json.dumps(paired[:6],indent=2))
