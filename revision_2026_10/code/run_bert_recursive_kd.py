#!/usr/bin/env python3
"""SST-2 matched recursive/fixed BERT distillation. Resume completed stages only."""
import argparse,csv,json,random,time,sys,math
from pathlib import Path

def variation(q):
    g=sum(max(a-b,0) for a,b in zip(q,q[1:])); a=sum(max(b-a,0) for a,b in zip(q,q[1:]))
    return g,a,max(q[0]-q[-1],0)

def main():
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',default='/content/drive/MyDrive/PPS_ID/bert_pilot')
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--generations',type=int,default=2)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--teacher-epochs',type=int,default=3)
    p.add_argument('--train-size',type=int,default=5000);p.add_argument('--eval-size',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--temperature',type=float,default=4)
    p.add_argument('--model',default='google-bert/bert-base-uncased');p.add_argument('--revision',default='main')
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        g,u,l=variation([.8,.7,.85,.82]);assert math.isclose(g,.13) and math.isclose(u,.15) and l==0
        assert math.isclose(.82-.8,u-g);print('PASS: cumulative decreases, gains, endpoint identities');return
    if min(a.generations,a.epochs,a.teacher_epochs,a.batch_size,a.max_length)<=0 or a.lr<=0 or a.temperature<=0 or a.train_size<0 or a.eval_size<0:
        p.error('Invalid nonpositive setting')
    import numpy as np,torch,transformers,datasets
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from datasets import load_dataset
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Enable a GPU runtime in Colab.')
    device=torch.device(a.device);out=Path(a.output)
    settings=vars(a).copy();settings.pop('resume');settings.pop('self_test')
    if out.exists():
        if not a.resume:raise FileExistsError(f'{out} exists. Use --resume for identical settings, or a new output directory.')
        old=json.loads((out/'config.json').read_text())
        if old['settings']!=settings:raise ValueError('Resume requires identical settings, including seeds and generation count.')
    else:out.mkdir(parents=True);old={}
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    def seed_all(s):
        random.seed(s);np.random.seed(s);torch.manual_seed(s)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    revision=old.get('model_revision',a.revision)
    tok=AutoTokenizer.from_pretrained(a.model,revision=revision)
    seed_all(0);probe=AutoModelForSequenceClassification.from_pretrained(a.model,revision=revision,num_labels=2)
    revision=getattr(probe.config,'_commit_hash',None) or revision;del probe
    ds=load_dataset('nyu-mll/glue','sst2')
    # Identical training subset for every seed/branch, independent of training RNG.
    ids=np.random.default_rng(2026).permutation(len(ds['train']))
    if a.train_size:ids=ids[:min(a.train_size,len(ids))]
    tr=ds['train'].select(ids.tolist());ev=ds['validation']
    if a.eval_size:ev=ev.select(range(min(a.eval_size,len(ev))))
    config=dict(settings=settings,status='running',model_revision=revision,train_size=len(tr),eval_size=len(ev),
      evaluation_split='SST-2 validation (not hidden-label test)',subset_seed=2026,
      train_fingerprint=tr._fingerprint,eval_fingerprint=ev._fingerprint,
      torch_version=torch.__version__,transformers_version=transformers.__version__,datasets_version=datasets.__version__,python_version=sys.version,
      supervision='M0:CE; students:KL_only',initialization='each model starts from same pretrained BERT revision; classifier randomly initialized',
      Q='validation_accuracy_fraction',D='mean_JS_natural_log_at_temperature_1',optimizer='AdamW weight_decay=0.01; constant_lr; gradient_clip=1.0')
    def atomic_json(path,value):
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
    atomic_json(out/'config.json',config);np.save(out/'train_indices.npy',ids)
    def tokenize(data):
        enc=tok(list(data['sentence']),truncation=True,padding='max_length',max_length=a.max_length,return_tensors='pt')
        return [{**{k:v[i] for k,v in enc.items()},'labels':torch.tensor(data['label'][i])} for i in range(len(data))]
    train=tokenize(tr);evaluation=tokenize(ev);eval_loader=DataLoader(evaluation,batch_size=a.batch_size,shuffle=False)
    rows=[]
    def write_csv(path,records):
        if not records:return
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        tmp.replace(path)
    def fresh(s):
        seed_all(s)
        return AutoModelForSequenceClassification.from_pretrained(a.model,revision=revision,num_labels=2).to(device)
    def js(x,y):
        x=np.maximum(x.astype('float64'),1e-12);y=np.maximum(y.astype('float64'),1e-12)
        x/=x.sum(1,keepdims=True);y/=y.sum(1,keepdims=True);m=(x+y)/2
        return float(((x*np.log(x/m)+y*np.log(y/m))/2).sum(1).mean())
    def stage(folder,s,epochs,teacher=None):
        # done.json is written last. An interrupted stage is restarted from its fresh initialization.
        if (folder/'done.json').exists():
            z=np.load(folder/'observations.npz');return z['probabilities'],z['labels'],float(json.loads((folder/'done.json').read_text())['Q'])
        folder.mkdir(parents=True,exist_ok=True);student=fresh(s)
        if teacher is not None:teacher.eval();teacher.requires_grad_(False)
        loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(s))
        opt=torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=.01);logs=[];started=time.monotonic()
        for epoch in range(epochs):
            student.train();total=0;count=0;ts=time.monotonic()
            for batch in loader:
                labels=batch['labels'].to(device);inputs={k:v.to(device) for k,v in batch.items() if k!='labels'}
                logits=student(**inputs).logits
                if teacher is None:loss=torch.nn.functional.cross_entropy(logits,labels)
                else:
                    with torch.no_grad():target=torch.softmax(teacher(**inputs).logits/a.temperature,dim=-1)
                    loss=torch.nn.functional.kl_div(torch.log_softmax(logits/a.temperature,dim=-1),target,reduction='batchmean')*a.temperature**2
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1);opt.step()
                total+=loss.item()*len(labels);count+=len(labels)
            logs.append(dict(epoch=epoch+1,mean_training_loss=total/count,seconds=time.monotonic()-ts));write_csv(folder/'training.csv',logs)
            print(f'{folder.parent.name}/{folder.name}: epoch {epoch+1}/{epochs}, loss={total/count:.6f}, seconds={logs[-1]["seconds"]:.1f}',flush=True)
        student.eval();probs=[];ys=[]
        with torch.no_grad():
            for batch in eval_loader:
                probs.append(torch.softmax(student(**{k:v.to(device) for k,v in batch.items() if k!='labels'}).logits,dim=-1).cpu().numpy());ys.append(batch['labels'].numpy())
        probs=np.concatenate(probs);ys=np.concatenate(ys);q=float((probs.argmax(1)==ys).mean())
        student.save_pretrained(folder/'model',safe_serialization=True);tok.save_pretrained(folder/'model')
        np.savez_compressed(folder/'observations.npz',probabilities=probs,labels=ys,sample_indices=np.array(ev['idx']))
        atomic_json(folder/'done.json',dict(Q=q,seconds=time.monotonic()-started));del student,opt
        if device.type=='cuda':torch.cuda.empty_cache()
        return probs,ys,q
    def load_teacher(folder):return AutoModelForSequenceClassification.from_pretrained(folder/'model').to(device)
    for s in a.seeds:
        root=out/f'seed_{s}';initial=root/'M0';p0,labels,q0=stage(initial,s,a.teacher_epochs)
        rows.append(dict(seed=s,branch='initial',generation=0,teacher_generation='',Q=q0,D_previous=0.,D_initial=0.,D_teacher=0.,G=0.,A=0.,L=0.));write_csv(out/'metrics.csv',rows)
        for branch in ['recursive','fixed']:
            previous=p0;qs=[q0]
            for generation in range(1,a.generations+1):
                folder=root/f'{branch}_M{generation}'
                teacher_folder=initial if branch=='fixed' or generation==1 else root/f'recursive_M{generation-1}'
                teacher=None if (folder/'done.json').exists() else load_teacher(teacher_folder)
                probs,ys,q=stage(folder,(s+100003*generation)%2**32,a.epochs,teacher)
                if teacher is not None:del teacher
                assert np.array_equal(labels,ys);qs.append(q);g,u,l=variation(qs)
                rows.append(dict(seed=s,branch=branch,generation=generation,teacher_generation=generation-1 if branch=='recursive' else 0,
                    Q=q,D_previous=js(probs,previous),D_initial=js(probs,p0),D_teacher=js(probs,previous if branch=='recursive' else p0),G=g,A=u,L=l))
                write_csv(out/'metrics.csv',rows);previous=probs
                print(f'{branch} seed={s} M{generation}: Q={q:.4f} G={g:.4f} L={l:.4f}',flush=True)
    summary=[]
    for branch,generation in sorted({(r['branch'],r['generation']) for r in rows}):
        group=[r for r in rows if r['branch']==branch and r['generation']==generation];r=dict(branch=branch,generation=generation,n=len(group))
        for k in ['Q','D_previous','D_initial','D_teacher','G','A','L']:
            v=[x[k] for x in group];r[k+'_mean']=float(np.mean(v));r[k+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else ''
        summary.append(r)
    write_csv(out/'summary.csv',summary);config['status']='completed';atomic_json(out/'config.json',config)
    print(f'Completed. Results: {out}',flush=True)
if __name__=='__main__':main()
