"""Re-evaluate published CIFAR-10 checkpoints on the ordered 10,000 test images.

pip install torch numpy pyarrow pillow
python audit_cifar_checkpoint_inference.py --checkpoint-root checkpoints/cifar10 --prediction-root revision_2026_10/data --dataset-parquet /path/test.parquet --output /path/audit
No training or parameter updates are performed. Dataset images are not redistributed.
"""
import argparse,csv,hashlib,io,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
import pyarrow.parquet as pq

DATASET_URL='https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint-root',type=Path,required=True);p.add_argument('--prediction-root',type=Path,required=True);p.add_argument('--dataset-parquet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--only',default=None,help='Optional regime:seed:model filter for inspecting a recorded discrepancy');args=p.parse_args()
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);args.output.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((args.checkpoint_root/'CHECKPOINT_MANIFEST.json').read_text());table=pq.read_table(args.dataset_parquet);ys=table['label'].to_numpy();assert len(ys)==10000
 images=np.stack([np.asarray(Image.open(io.BytesIO(v['bytes'])).convert('RGB')) for v in table['img'].to_pylist()]);x=torch.from_numpy(images.transpose(0,3,1,2).copy()).float().div_(255)
 mean=torch.tensor([.4914,.4822,.4465]).reshape(1,3,1,1);std=torch.tensor([.2023,.1994,.2010]).reshape(1,3,1,1);x=(x-mean)/std
 model=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*8*8,256),nn.ReLU(),nn.Linear(256,10));model.eval()
 rows=[];cache={};changed_examples=[];t=time.time()
 for i,rec in enumerate(manifest['models'],1):
  if args.only and args.only != f"{rec['regime']}:{rec['seed']}:{rec['model']}":continue
  f=args.checkpoint_root/rec['checkpoint_path'];h=rec['checkpoint_sha256'];assert hashlib.sha256(f.read_bytes()).hexdigest()==h
  old=np.load(args.prediction_root/rec['regime']/f"seed_{rec['seed']}"/rec['model']/'observations.npz',allow_pickle=False);assert np.array_equal(ys,old['labels']);assert np.array_equal(old['sample_indices'],np.arange(10000))
  if h in cache:probs=cache[h]
  else:
   model.load_state_dict(torch.load(f,map_location='cpu',weights_only=True));parts=[]
   with torch.inference_mode():
    for offset in range(0,len(x),128):parts.append(torch.softmax(model(x[offset:offset+128]),dim=1).numpy())
   probs=np.concatenate(parts);cache[h]=probs
  reference=old['probabilities'];assert probs.shape==reference.shape==(10000,10);diff=np.abs(probs.astype(np.float64)-reference.astype(np.float64));pred=probs.argmax(1);saved=reference.argmax(1)
  row={'regime':rec['regime'],'seed':rec['seed'],'model':rec['model'],'checkpoint_sha256':h,'test_examples':10000,'prediction_class_mismatches':int(np.sum(pred!=saved)),'saved_correct':int(np.sum(saved==ys)),'rerun_correct':int(np.sum(pred==ys)),'max_probability_abs_difference':float(diff.max()),'mean_probability_abs_difference':float(diff.mean()),'probabilities_allclose_atol_1e_5_rtol_1e_5':bool(np.allclose(probs,reference,atol=1e-5,rtol=1e-5))}
  for j in np.flatnonzero(pred!=saved):
   top=np.argsort(reference[j])[-2:][::-1]
   changed_examples.append({'regime':rec['regime'],'seed':rec['seed'],'model':rec['model'],'test_index':int(j),'true_label':int(ys[j]),'saved_prediction':int(saved[j]),'rerun_prediction':int(pred[j]),'saved_top_two_classes':top.tolist(),'saved_top_two_probabilities':reference[j,top].tolist(),'saved_top_two_margin':float(reference[j,top[0]]-reference[j,top[1]]),'rerun_probabilities_for_saved_top_two_classes':probs[j,top].tolist(),'max_probability_abs_difference_for_example':float(diff[j].max())})
  rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(f"{i}/110 {rec['regime']} seed {rec['seed']} {rec['model']}: class mismatches={row['prediction_class_mismatches']}, max probability difference={row['max_probability_abs_difference']:.9g}; elapsed={time.time()-t:.1f}s",flush=True)
 with (args.output/'inference_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'scope':'Inference only, no retraining; original CUDA saved predictions versus CPU re-evaluation','model_locations':len(rows),'distinct_checkpoint_hashes_evaluated':len(cache),'test_examples_per_model':10000,'class_mismatches_total':sum(r['prediction_class_mismatches'] for r in rows),'all_saved_accuracies_reproduced':all(r['saved_correct']==r['rerun_correct'] for r in rows),'max_probability_abs_difference':max(r['max_probability_abs_difference'] for r in rows),'all_probabilities_allclose_atol_1e_5_rtol_1e_5':all(r['probabilities_allclose_atol_1e_5_rtol_1e_5'] for r in rows),'dataset_source':DATASET_URL,'dataset_parquet_sha256':hashlib.sha256(args.dataset_parquet.read_bytes()).hexdigest(),'dataset_labels_and_indices_match':True,'normalization_mean':[.4914,.4822,.4465],'normalization_std':[.2023,.1994,.2010],'batch_size':128,'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':'cpu','threads':args.threads,'elapsed_seconds':time.time()-t}
 (args.output/'changed_examples.json').write_text(json.dumps(changed_examples,indent=2)+'\n')
 (args.output/'inference_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
