"""Reproduce prediction-based tables and empirical figures, without training.
Run from the package root: python code/reproduce_analysis.py --output regenerated
"""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def js(p,q):
 p=np.maximum(p.astype(np.float64),1e-12);q=np.maximum(q.astype(np.float64),1e-12)
 p=p/p.sum(1,keepdims=True);q=q/q.sum(1,keepdims=True);m=(p+q)/2
 return float(((p*np.log(p/m)+q*np.log(q/m))/2).sum(1).mean())
def ece(p,y,bins):
 c=p.argmax(1)==y;conf=p.max(1);value=0.
 for j in range(bins):
  mask=(conf>=j/bins)&((conf<(j+1)/bins) if j<bins-1 else (conf<=1))
  if mask.any():value+=mask.mean()*abs(c[mask].mean()-conf[mask].mean())
 return float(value)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('regenerated'));args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
 states={};audit=[];metrics=[];turn=[];summ=[];margin=[];checks=[]
 specs=[('cifar10_soft',range(42,47),10000),('cifar10_mixed',range(42,47),10000),('sst2',[42,43],872),('imdb',[42,43],1000)]
 for ds,seeds,n in specs:
  for seed in seeds:
   for stage in ['M0']+[f'{b}_M{g}' for b in ['recursive','fixed'] for g in range(1,6)]:
    f=ROOT/'data'/ds/f'seed_{seed}'/stage/'observations.npz';a=np.load(f,allow_pickle=False);p=a['probabilities'].astype(np.float64);y=a['labels'];idx=a['sample_indices'] if 'sample_indices' in a else a['indices'];b='initial' if stage=='M0' else stage.split('_')[0];g=0 if stage=='M0' else int(stage.split('M')[1])
    assert len(y)==n and p.shape==(n,10 if ds.startswith('cifar') else 2)
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    states[ds,seed,b,g]=(p,y,idx)
    p0,y0,idx0=states[ds,seed,'initial',0];assert np.array_equal(y,y0) and np.array_equal(idx,idx0)
    prev=p0 if g<=1 else states[ds,seed,b,g-1][0];teacher=p0 if b=='fixed' else prev;c=p.argmax(1)==y
    metrics.append(dict(dataset=ds,seed=seed,branch=b,generation=g,accuracy=float(c.mean()),JS_previous=js(p,prev) if g else 0.,JS_initial=js(p,p0),JS_teacher=js(p,teacher),ECE10=ece(p,y,10),ECE15=ece(p,y,15),ECE20=ece(p,y,20),entropy=float(-(p*np.log(np.clip(p,1e-300,None))).sum(1).mean()),confidence=float(p.max(1).mean())))
    audit.append(dict(dataset=ds,seed=seed,stage=stage,examples=n,classes=p.shape[1],probability_sum_max_error=float(abs(p.sum(1)-1).max()),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
   for b in ['recursive','fixed']:
    ps=[states[ds,seed,'initial',0][0]]+[states[ds,seed,b,g][0] for g in range(1,6)];cs=[p.argmax(1)==y0 for p in ps];qs=np.array([c.mean() for c in cs]);dq=np.diff(qs);recover=np.zeros(n,bool);cw_total=wc_total=0;G=A=0.
    for g in range(1,6):
     before,after=cs[g-1:g+1];cw=int((before&~after).sum());wc=int((~before&after).sum());cw_total+=cw;wc_total+=wc;G+=max(-dq[g-1],0);A+=max(dq[g-1],0)
     turn.append(dict(dataset=ds,seed=seed,branch=b,generation=g,correct_to_wrong=cw,wrong_to_correct=wc,persistent_wrong=int((~before&~after).sum()),persistent_correct=int((before&after).sum()),G=G,A=A,L=max(qs[0]-qs[g],0)))
     if g<5:recover|=(before&~after)&np.logical_or.reduce(cs[g+1:])
     m=np.sort(ps[g-1],axis=1);gap=m[:,-1]-m[:,-2];delta=abs(ps[g]-ps[g-1]).max(1);risk=gap<=2*delta+1e-12;flip=ps[g].argmax(1)!=ps[g-1].argmax(1);assert not np.any(flip&~risk);assert abs(dq[g-1])<=risk.mean()+1e-12
     margin.append(dict(dataset=ds,seed=seed,branch=b,generation=g,margin_risk_fraction=risk.mean(),class_change_fraction=flip.mean(),absolute_accuracy_change=abs(dq[g-1]),certified_stable_fraction=(~risk).mean(),bound_slack=risk.mean()-abs(dq[g-1])))
    summ.append(dict(dataset=ds,seed=seed,branch=b,Q0=qs[0],Q5=qs[-1],G=G,A=A,L=max(qs[0]-qs[-1],0),signed_endpoint_loss=qs[0]-qs[-1],loss_then_recovery=int(recover.sum()),correct_to_wrong=cw_total,wrong_to_correct=wc_total))
    assert abs((qs[-1]-qs[0])-(A-G))<1e-12
 for seed in range(42,47):
  assert np.array_equal(states['cifar10_soft',seed,'initial',0][0],states['cifar10_mixed',seed,'initial',0][0])
  for ds in ['cifar10_soft','cifar10_mixed']:assert np.array_equal(states[ds,seed,'recursive',1][0],states[ds,seed,'fixed',1][0])
 df=pd.DataFrame(metrics);summary=pd.DataFrame(summ);turn=pd.DataFrame(turn);margin=pd.DataFrame(margin)
 pd.DataFrame(audit).to_csv(out/'prediction_audit.csv',index=False);df.to_csv(out/'generation_metrics.csv',index=False);summary.to_csv(out/'trajectory_summary.csv',index=False);turn.to_csv(out/'correctness_transitions.csv',index=False);margin.to_csv(out/'margin_validation_all.csv',index=False)
 original=margin[margin.dataset!='cifar10_mixed'];assert len(original)==90;original.to_csv(out/'margin_validation_original90.csv',index=False);original.groupby(['dataset','branch']).mean(numeric_only=True).reset_index().to_csv(out/'margin_summary_original90.csv',index=False)
 # Compare with previously reported tables and raw source metrics.
 for ds,fn in [('cifar10_soft','calibration_entropy.csv'),('cifar10_mixed','mixed_calibration_entropy.csv')]:
  reported=pd.read_csv(ROOT/'reported_tables'/fn);now=df[df.dataset==ds]
  merged=reported.merge(now,on=['seed','branch','generation'],suffixes=('_reported','_recomputed'));assert len(merged)==55
  for key in ['accuracy','ECE10','ECE15','ECE20','entropy','confidence']:
   err=float(abs(merged[key+'_reported']-merged[key+'_recomputed']).max());assert err<1e-12;checks.append(dict(table=fn,field=key,max_absolute_difference=err))
 reported=pd.read_csv(ROOT/'reported_tables/bert_generation_metrics.csv').replace({'dataset':{'SST2':'sst2','IMDb':'imdb'}})
 bdf=df[df.dataset.isin(['sst2','imdb'])].copy();zero=bdf[bdf.branch=='initial'];bdf=bdf[bdf.branch!='initial']
 for b in ['recursive','fixed']:z=zero.copy();z['branch']=b;bdf=pd.concat([bdf,z])
 merged=reported.merge(bdf,on=['dataset','seed','branch','generation'],suffixes=('_reported','_recomputed'));assert len(merged)==48
 for key in ['accuracy','JS_previous','JS_initial']:
  err=float(abs(merged[key+'_reported']-merged[key+'_recomputed']).max());assert err<1e-12;checks.append(dict(table='bert_generation_metrics.csv',field=key,max_absolute_difference=err))
 bs=[]
 for ds in ['cifar10_soft','cifar10_mixed']:
  dif=np.array([states[ds,s,'fixed',5][0].argmax(1)==states[ds,s,'fixed',5][1] for s in range(42,47)],float)-np.array([states[ds,s,'recursive',5][0].argmax(1)==states[ds,s,'recursive',5][1] for s in range(42,47)],float)
  seed_diff=dif.mean(1)*100;ci=t.interval(.95,4,loc=seed_diff.mean(),scale=seed_diff.std(ddof=1)/np.sqrt(5));per_example=dif.mean(0);rng=np.random.default_rng(20261012);values=np.array([per_example[rng.integers(0,len(per_example),len(per_example))].mean()*100 for _ in range(2000)])
  bs.append(dict(dataset=ds,fixed_minus_recursive_pp=seed_diff.mean(),paired_seed_t95_pp=list(ci),conditional_test_bootstrap95_pp=np.quantile(values,[.025,.975]).tolist(),replicates=2000,bootstrap_seed=20261012))
 (out/'uncertainty.json').write_text(json.dumps(bs,indent=2))
 assert np.allclose(bs[0]['conditional_test_bootstrap95_pp'],[1.19195,1.902],atol=1e-8)
 cka=pd.read_csv(ROOT/'data/cka.csv');assert len(cka)==150 and not cka.duplicated(['seed','branch','generation','layer']).any();assert np.isfinite(cka[['CKA_previous','CKA_initial']]).all().all();assert cka[['CKA_previous','CKA_initial']].min().min()>=0 and cka[['CKA_previous','CKA_initial']].max().max()<=1
 cka.groupby(['branch','generation','layer'])[['CKA_previous','CKA_initial']].agg(['mean','std']).to_csv(out/'cka_summary.csv')
 plt.rcParams.update({'font.size':9})
 def vals(ds,b,key):
  return np.array([[df[(df.dataset==ds)&(df.seed==s)&(df.branch==('initial' if g==0 else b))&(df.generation==g)][key].iloc[0] for g in range(6)] for s in (range(42,47) if ds.startswith('cifar') else [42,43])])
 fig,axs=plt.subplots(2,2,figsize=(9,6.2))
 for ds,label,style in [('cifar10_soft','Soft only','--'),('cifar10_mixed','Mixed labels','-')]:
  for b,color in [('recursive','#176a9b'),('fixed','#b85c21')]:
   for ax,key,scale,ylabel in [(axs[0,0],'accuracy',100,'Accuracy (%)'),(axs[1,0],'ECE15',1,'Top-label ECE (15 bins)'),(axs[1,1],'entropy',1,'Prediction entropy (nats)')]:
    v=vals(ds,b,key)*scale;ax.errorbar(range(6),v.mean(0),yerr=v.std(0,ddof=1),color=color,linestyle=style,marker='o',markersize=3,capsize=2,label=f'{label}: {b}');ax.set_ylabel(ylabel)
   qs=vals(ds,b,'accuracy');g=np.c_[np.zeros(5),np.cumsum(np.maximum(-np.diff(qs,axis=1),0),axis=1)]*100;axs[0,1].errorbar(range(6),g.mean(0),yerr=g.std(0,ddof=1),color=color,linestyle=style,marker='o',markersize=3,capsize=2)
 axs[0,1].set_ylabel('Cumulative accuracy decreases (pp)')
 for ax in axs.flat:ax.set_xlabel('Generation');ax.set_xticks(range(6));ax.grid(alpha=.2)
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=8);fig.tight_layout(rect=(0,.085,1,1));fig.savefig(out/'figures/mixed_comparison.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(10,3.5))
 for ax,layer in zip(axs,['pool1 GAP','pool2 GAP','FC ReLU']):
  for b,color in [('recursive','#176a9b'),('fixed','#b85c21')]:
   for key,sty,ref in [('CKA_previous','-','previous'),('CKA_initial','--','initial')]:
    v=cka[(cka.layer==layer)&(cka.branch==b)].pivot(index='seed',columns='generation',values=key).to_numpy();ax.errorbar(range(1,6),v.mean(0),yerr=v.std(0,ddof=1),color=color,linestyle=sty,marker='o',markersize=3,capsize=2,label=f'{b}: {ref}')
  ax.set_title(layer);ax.set_xlabel('Generation');ax.set_ylim(.84,1.005);ax.set_xticks(range(1,6));ax.grid(alpha=.2)
 axs[0].set_ylabel('Linear CKA');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=8);fig.tight_layout(rect=(0,.17,1,1));fig.savefig(out/'figures/cka_trajectories.pdf');plt.close(fig)
 for ds,title in [('sst2','SST2'),('imdb','IMDb')]:
  fig,axs=plt.subplots(2,3,figsize=(10,5.8))
  for i,s in enumerate([42,43]):
   for b in ['recursive','fixed']:
    for ax,key,scale,label in zip(axs[i],['accuracy','JS_previous','JS_initial'],[100,1,1],['Accuracy (%)','JS previous (nats)','JS initial (nats)']):ax.plot(range(6),vals(ds,b,key)[i]*scale,marker='o',label=b);ax.set_title(f'{title}, seed {s}: {label}',fontsize=9);ax.set_xlabel('Generation');ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.tight_layout();fig.savefig(out/f'figures/{ds}_validation.pdf');plt.close(fig)
 source_checks=[]
 for ds in ['cifar10_soft','cifar10_mixed','sst2','imdb']:
  for f in (ROOT/'protocols'/ds).rglob('metrics.csv'):
   raw=pd.read_csv(f).replace({'branch':{'teacher':'initial'}})
   for _,row in raw.iterrows():
    key='Q' if 'Q' in raw else 'accuracy';match=df[(df.dataset==ds)&(df.seed==int(row.seed))&(df.branch==row.branch)&(df.generation==int(row.generation))]
    assert len(match)==1;err=abs(float(row[key])-float(match.accuracy.iloc[0]));assert err<1e-12
    source_checks.append(dict(dataset=ds,seed=int(row.seed),branch=row.branch,generation=int(row.generation),absolute_accuracy_difference=err))
 assert len(source_checks)==154
 pd.DataFrame(source_checks).to_csv(out/'source_metric_checks.csv',index=False)
 pd.DataFrame(checks).to_csv(out/'reported_table_checks.csv',index=False)
 report=dict(prediction_arrays=len(audit),margin_transitions=len(margin),original_margin_transitions=len(original),cka_records=len(cka),reported_field_checks=len(checks),source_accuracy_checks=len(source_checks),all_checks_passed=True,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__)
 (out/'validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
