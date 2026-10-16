import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--data',default='./data');p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=1000);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
indices=np.random.default_rng(20261012).permutation(10000)[:a.samples];np.save(out/'test_indices.npy',indices)
protocol={'samples':a.samples,'selection_seed':20261012,'layers':['pool1 GAP','pool2 GAP','FC ReLU'],'CKA':'biased centered linear Gram CKA float64','torch':torch.__version__,'device':a.device}
if (out/'protocol.json').exists():
 old=json.loads((out/'protocol.json').read_text());assert old['samples']==a.samples and old['selection_seed']==20261012
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.2023,.1994,.2010))]);ds=datasets.CIFAR10(a.data,train=False,download=True,transform=t);loader=DataLoader(Subset(ds,indices.tolist()),batch_size=128,shuffle=False)
def gram(folder):
 cache=out/(folder.parent.name+'_'+folder.name+'.npz')
 if cache.exists():
  x=np.load(cache);return [x[k] for k in ['g0','g1','g2']]
 m=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*8*8,256),nn.ReLU(),nn.Linear(256,10)).to(a.device)
 m.load_state_dict(torch.load(folder/'model.pt',map_location=a.device,weights_only=True));m.eval();vals=[[],[],[]];hooks=[]
 for j,k in enumerate([2,5,8]):
  def hook(module,args,v,j=j):
   if v.ndim==4:v=v.mean(dim=(2,3))
   vals[j].append(v.detach().cpu().numpy())
  hooks.append(m[k].register_forward_hook(hook))
 with torch.no_grad():
  for x,y in loader:m(x.to(a.device))
 gs=[]
 for v in vals:
  x=np.concatenate(v).astype(np.float64);x-=x.mean(0,keepdims=True);gs.append(x@x.T)
 for h in hooks:h.remove()
 np.savez_compressed(cache,**{f'g{i}':g for i,g in enumerate(gs)});return gs
def cka(x,y):
 d=np.linalg.norm(x)*np.linalg.norm(y)
 return float(np.sum(x*y)/d) if d>1e-20 else float('nan')
rows=[];started=time.monotonic()
for s in range(42,47):
 base=Path(a.root)/f'seed_{s}';initial=gram(base/'M0')
 for b in ['recursive','fixed']:
  prev=initial
  for g in range(1,6):
   current=gram(base/f'{b}_M{g}')
   for l in range(3):rows.append(dict(seed=s,branch=b,generation=g,layer=protocol['layers'][l],CKA_previous=cka(current[l],prev[l]),CKA_initial=cka(current[l],initial[l])))
   prev=current
   with open(out/'cka.csv','w') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
   print(s,b,g,'done',flush=True)
print('Seconds',time.monotonic()-started)
