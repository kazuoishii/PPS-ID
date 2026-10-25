"""Reproduce Section 8.6; writes trajectories, summary, and Figure 4 locally."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
n=20; a=.2; gamma=1.2; lam=2.
W=np.full((n,n),.1/n)
W[:10,:10]+=.9/10; W[10:,10:]+=.9/10
b=np.linspace(-.5,.2,n); z=np.linspace(-8,2,n)
delta=float(np.min(b+lam)-gamma); M=delta/(1-a)
rows=[]
for t in range(101):
    L=a**t*(-8)+(1-a**t)*M
    assert z.min()>=L-1e-12
    rows.append([t,z.min(),L,*z])
    if t<100: z=a*z+gamma*(W@np.tanh(z))+b+lam
r=np.asarray(rows)
np.savetxt(P/'heterogeneous.csv',r,delimiter=',',header=','.join(['t','minimum_signed_state','lower_bound']+[f'z_{i+1}' for i in range(n)]),comments='')
cross=int(r[r[:,1]>0][0,0]); assert np.all(r[r[:,0]>=cross,1]>0)
summary=dict(delta=delta,M=M,network_crossing=cross,bound_crossing=int(r[r[:,2]>0][0,0]),L1=float(r[1,2]),L2=float(r[2,2]),minimum_bound_gap=float(np.min(r[:,1]-r[:,2])),checked_times=101,updates=100)
(P/'heterogeneous_results.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(9,3.3))
for ax,limit,title in zip(axes,[100,6],['Full trajectory','Initial correction']):
    for j in range(n): ax.plot(r[:,0],r[:,3+j],color='.8',lw=.6)
    ax.plot(r[:,0],r[:,1],color='#1764ab',lw=1.7,label='Minimum state')
    ax.plot(r[:,0],r[:,2],color='#bd3828',ls='--',lw=1.7,label='Theoretical lower bound')
    ax.axhline(0,color='.3',lw=.8); ax.set_xlim(0,limit)
    ax.set_xlabel('Iteration'); ax.set_title(title)
axes[0].set_ylabel('Signed state (h = 1)');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(P/'heterogeneous_bound.pdf');plt.close(fig)
print(json.dumps(summary,indent=2))
