"""Reproduce model calculations from archived inputs by default. See README.md."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv,json
import argparse
SOURCE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=SOURCE/'generated', help='Separate output directory; archived inputs and results are not overwritten.')
parser.add_argument('--regenerate-inputs', action='store_true', help='Regenerate inputs using the recorded RNG API and draw order; save only to output directory.')
args=parser.parse_args()
P=args.output_dir.resolve()
if P==SOURCE: raise ValueError('Use a separate output directory to preserve archived results.')
P.mkdir(parents=True,exist_ok=True)
def sig(x): return 1/(1+np.exp(-x))
def solve(x,d,g=.4,W=None,tol=1e-10):
    for t in range(20000):
        y=.2*x+g*(np.tanh(x) if W is None else np.tanh(x)@W.T)+d
        x=y
        residual=np.max(np.abs(.2*x+g*(np.tanh(x) if W is None else np.tanh(x)@W.T)+d-x))
        if residual<tol:return x,t+1,residual
    return x,20000,residual
q=np.arccosh(np.sqrt(1.2/.8));dc=1.2*np.tanh(q)-.8*q
neg=solve(np.array(-3.),.1,1.2)[0];pos=solve(np.array(3.),.1,1.2)[0]
ms=np.arange(1,9);zs=np.array([solve(np.array(0.),2-.5*m)[0] for m in ms])
# The exact zero-drive fixed point is zero, including the declared tie.
plt.figure(figsize=(6.4,3.2));plt.axhline(.5,color='gray',ls='--');plt.ylim(0,1);plt.plot(ms,sig(zs),'o-');plt.axvline(4,color='gray',ls='--');plt.xlabel('Multiplicity');plt.ylabel('Equilibrium probability');plt.tight_layout();plt.savefig(P/'duplication.pdf');plt.close()
tr=[];x=y=0.
for t in range(301):
    tr.append((t,x,y));d=-.9 if t<80 else .1
    x=.2*x+1.2*np.tanh(x)+d
    y=.2*y+1.2*np.tanh(y)+d+(.3 if t>=160 else 0)
tr=np.array(tr);np.savetxt(P/'recovery.csv',tr,delimiter=',',header='t,no_reference,reference',comments='')
plt.figure(figsize=(6.4,3.2));plt.axhline(.5,color='lightgray');plt.axvline(80,color='gray',ls=':',label='Deduplication');plt.axvline(160,color='gray',ls='--',label='Reference introduced');plt.ylim(0,1);plt.plot(tr[:,0],sig(tr[:,1]),label='No reference');plt.plot(tr[:,0],sig(tr[:,2]),label='Reference');plt.legend();plt.xlabel('Iteration');plt.ylabel('Probability');plt.tight_layout();plt.savefig(P/'recovery.pdf');plt.close()
lam=np.linspace(0,.8,81);es=np.linspace(0,1.5,76);L,E=np.meshgrid(lam,es);d=.1+L*(1-E);x=np.full(d.shape,neg);counts=np.zeros(d.shape,int);active=np.ones(d.shape,bool)
for t in range(20000):
    y=.2*x+1.2*np.tanh(x)+d;x=np.where(active,y,x)
    residual=np.abs(.2*x+1.2*np.tanh(x)+d-x)
    done=active&(residual<1e-9);counts[done]=t+1;active[done]=False
    if not active.any():break
counts[active]=20000
np.savetxt(P/'grid.csv',np.column_stack([L.ravel(),E.ravel(),x.ravel(),residual.ravel(),counts.ravel()]),delimiter=',',header='lambda,e,z,residual,iterations',comments='')
plt.figure(figsize=(6.4,3.2));plt.imshow(sig(x),origin='lower',extent=[0,.8,0,1.5],aspect='auto',vmin=0,vmax=1,interpolation='nearest');plt.colorbar(label='Probability');ee=es[es<1];ll=(dc-.1)/(1-ee);ok=ll<=.8;plt.plot(ll[ok],ee[ok],'w--',label='Exact scalar fold');plt.legend(loc='upper right');plt.xlabel('Reference strength');plt.ylabel('Adverse reference bias');plt.tight_layout();plt.savefig(P/'reference_grid.pdf');plt.close()
n=20;complete=np.ones((n,n))/n;cycle=np.roll(np.eye(n),-1,axis=1);modular=np.full((n,n),.1/n)
modular[:10,:10]+=.9/10;modular[10:,10:]+=.9/10
graphs={'Complete':complete,'Cycle':cycle,'Modular':modular}
z=np.linspace(-8,2,n);v=-8.;nc=vc=None;gap=1e9;lossgap=1e9
for t in range(301):
    gap=min(gap,z.min()-v);lossgap=min(lossgap,sig(-v)**2-np.mean(sig(-z)**2))
    if nc is None and z.min()>0:nc=t
    if vc is None and v>0:vc=t
    dt=.3+.05*(1+np.sin(.17*t+2*np.pi*np.arange(n)/n))
    z=.2*z+1.2*(list(graphs.values())[t%3]@np.tanh(z))+dt;v=.2*v+1.2*np.tanh(v)+.3
z=np.linspace(-8,2,n);conservative_gap=1e9
for t in range(101):
    floor=.2**t*(-8)+(1-.2**t)*.375;conservative_gap=min(conservative_gap,z.min()-floor)
    z=.2*z+1.2*(modular@np.tanh(z))+np.linspace(-.5,.2,n)+2
# Default replication uses authoritative archived arrays, without modifying them.
if args.regenerate_inputs:
    rng=np.random.default_rng(20261006)
    h=rng.integers(0,2,1000)
    X=rng.normal((2*h[:,None]-1)*.5,1,size=(1000,6))
    mask=rng.random((n,6))<.65
    mask[:,0]=True
    np.savez(P/'inputs.npz',h=h,X=X,mask=mask,**graphs)
else:
    with np.load(SOURCE/'inputs.npz', allow_pickle=False) as archive:
        h=archive['h']; X=archive['X']; mask=archive['mask']
        graphs={name:archive[name] for name in ['Complete','Cycle','Modular']}
    assert h.shape==(1000,) and X.shape==(1000,6) and mask.shape==(20,6)
    for W in graphs.values():
        assert W.shape==(20,20) and np.all(W>=0) and np.allclose(W.sum(axis=1),1)
graphs['Cycle_reverse']=np.roll(np.eye(n),1,axis=1)
b=.1*(X@mask.T);reuse=b+.4*X[:,0,None];rows=[];paired=[];boundgap=1e9
for name,W in graphs.items():
    results=[]
    for label,drive in [('unique',b),('reuse',reuse)]:
        z,it,res=solve(np.zeros_like(b),drive,W=W);p=sig(z);acc=((p>=.5)==h[:,None]).mean(axis=1);loss=((p-h[:,None])**2).mean(axis=1);results.append((z,acc,loss))
        rows.append([name,label,acc.mean(),acc.std(ddof=1)/np.sqrt(1000),loss.mean(),loss.std(ddof=1)/np.sqrt(1000),it,res])
    z0,a0,l0=results[0];z1,a1,l1=results[1];da=a1-a0;dl=l1-l0
    paired.append([name,da.mean(),da.std(ddof=1)/np.sqrt(1000),dl.mean(),dl.std(ddof=1)/np.sqrt(1000)])
    boundgap=min(boundgap,np.min(np.max(np.abs(reuse-b),axis=1)/.4-np.max(np.abs(z1-z0),axis=1)))
    np.savetxt(P/(name.lower()+'_episodes.csv'),np.column_stack([h,a0,a1,l0,l1]),delimiter=',',header='h,accuracy_unique,accuracy_reuse,brier_unique,brier_reuse',comments='')
for fn,header,data in [('table.csv',['graph','use','accuracy','accuracy_se','brier','brier_se','iterations','residual'],rows),('paired.csv',['graph','accuracy_difference','accuracy_difference_se','brier_difference','brier_difference_se'],paired)]:
    with (P/fn).open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(data)
report=dict(q=float(q),dc=float(dc),negative_root=float(neg),positive_root=float(pos),grid_guaranteed_points=int((d>dc+1e-5).sum()),grid_failures=int(((d>dc+1e-5)&(x<=0)).sum()),grid_max_residual=float(residual.max()),network_crossing=nc,envelope_crossing=vc,certified_time=int(np.floor(8/(.3-dc))+1),envelope_min_gap=float(gap),loss_min_gap=float(lossgap),conservative_min_gap=float(conservative_gap),input_bound_min_gap=float(boundgap),table=rows,paired=paired)
(P/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
