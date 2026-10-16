#!/usr/bin/env python3
"""Compare two identical M0->M1 starts; no existing model files are changed."""
import argparse, hashlib, json, random, gc, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',choices=['mps','cpu'],default='mps')
p.add_argument('--steps',type=int,default=3)
a=p.parse_args()
if a.steps<1: p.error('steps must be positive')
a.output.mkdir(parents=True,exist_ok=False)
c=json.loads((a.source/'config.json').read_text()); s=c['settings']; seed=s['seeds'][0]
revision=c['model_revision']; device=torch.device(a.device)
torch.use_deterministic_algorithms(True,warn_only=True)
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def digest(t):
    t=t.detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes());return h.hexdigest()
def weights(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(digest(v).encode())
    return h.hexdigest()
def rng():
    d={'cpu':digest(torch.get_rng_state())}
    if device.type=='mps':d['mps']=digest(torch.mps.get_rng_state())
    return d

tok=AutoTokenizer.from_pretrained(s['model'],revision=revision)
ds=load_dataset('nyu-mll/glue','sst2')
ids=np.load(a.source/'train_indices.npy',allow_pickle=False)
tr=ds['train'].select(ids.tolist())
if tr._fingerprint!=c['train_fingerprint']:raise RuntimeError('Training fingerprint differs from original')
enc=tok(list(tr['sentence']),truncation=True,padding='max_length',max_length=s['max_length'],return_tensors='pt')
train=[{**{k:v[i] for k,v in enc.items()},'labels':torch.tensor(tr['label'][i]),'row_id':torch.tensor(int(ids[i]))} for i in range(len(tr))]
results=[]; started=time.monotonic()
for trial in range(2):
    teacher=AutoModelForSequenceClassification.from_pretrained(a.source/f'seed_{seed}/M0/model').to(device)
    stage_seed=(seed+100003)%2**32
    random.seed(stage_seed);np.random.seed(stage_seed);torch.manual_seed(stage_seed)
    student=AutoModelForSequenceClassification.from_pretrained(s['model'],revision=revision,num_labels=2).to(device)
    teacher.eval();teacher.requires_grad_(False)
    loader=DataLoader(train,batch_size=s['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(stage_seed))
    opt=torch.optim.AdamW(student.parameters(),lr=s['lr'],weight_decay=.01)
    record={'trial':trial,'initial_weights':weights(student),'teacher_weights':weights(teacher),'rng_after_initialization':rng(),'steps':[]}
    student.train()
    for step,batch in enumerate(loader):
        if step>=a.steps:break
        inputs={k:v.to(device) for k,v in batch.items() if k not in ['labels','row_id']}
        row={'step':step,'row_ids':batch['row_id'].tolist(),'input_hashes':{k:digest(v) for k,v in inputs.items()},'rng_before_forward':rng()}
        logits=student(**inputs).logits
        row['student_logits']=logits.detach().cpu().tolist();row['rng_after_student']=rng()
        with torch.no_grad():target=torch.softmax(teacher(**inputs).logits/s['temperature'],dim=-1)
        row['teacher_probabilities']=target.cpu().tolist()
        loss=torch.nn.functional.kl_div(torch.log_softmax(logits/s['temperature'],dim=-1),target,reduction='batchmean')*s['temperature']**2
        row['loss']=loss.item()
        opt.zero_grad(set_to_none=True);loss.backward()
        row['classifier_gradient']=student.classifier.weight.grad.detach().cpu().tolist()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1);opt.step()
        row['weights_after_update']=weights(student);record['steps'].append(row)
        print(f'trial {trial+1}, step {step+1}: loss={loss.item():.8f}',flush=True)
    results.append(record)
    (a.output/f'trial_{trial+1}.json').write_text(json.dumps(record,indent=2))
    del student,teacher,opt;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()

x,y=results
comparison={'initial_weights_equal':x['initial_weights']==y['initial_weights'],'teacher_weights_equal':x['teacher_weights']==y['teacher_weights'],'initial_rng_equal':x['rng_after_initialization']==y['rng_after_initialization'],'steps':[]}
for u,v in zip(x['steps'],y['steps']):
    row={'step':u['step'],'batch_equal':u['row_ids']==v['row_ids'] and u['input_hashes']==v['input_hashes'],'rng_before_equal':u['rng_before_forward']==v['rng_before_forward'],'weights_after_equal':u['weights_after_update']==v['weights_after_update']}
    for key in ['student_logits','teacher_probabilities','classifier_gradient']:row[key+'_max_abs_difference']=float(np.max(np.abs(np.array(u[key])-np.array(v[key]))))
    row['loss_abs_difference']=abs(u['loss']-v['loss']);comparison['steps'].append(row)
report={'device':a.device,'source':str(a.source),'model_revision':revision,'torch':torch.__version__,'seconds':time.monotonic()-started,'comparison':comparison,'note':'A short prefix diagnostic; agreement here does not establish full-training determinism.'}
(a.output/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
