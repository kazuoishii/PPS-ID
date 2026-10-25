"""Finite-precision checks of the zero-threshold regime; not a proof."""
from pathlib import Path
import json
import numpy as np
out=[]
for a,g,k in [(0.2,0.4,1.),(0.2,0.8,1.),(0.,1.,1.)]:
 n=8
 graphs=[np.ones((n,n))/n,np.roll(np.eye(n),1,axis=1)]
 for D in [.01,.3]:
  z=np.linspace(-8,2,n);v=-8.;cross=None;err=0.
  for t in range(10000):
   assert z.min()>=v-1e-11
   assert np.mean((1/(1+np.exp(z)))**2)<=(1/(1+np.exp(v)))**2+1e-11
   err=max(err,max(0.,v-z.min()))
   if v>0 and cross is None:cross=t
   if t>=int(np.floor(8/D))+1:assert v>0
   d=D+.01*(1+np.sin(t+np.arange(n)))
   z=a*z+g*(graphs[t%2]@np.tanh(k*z))+d
   v=a*v+g*np.tanh(k*v)+D
  assert cross is not None
  out.append(dict(a=a,gamma=g,kappa=k,D=D,crossing=cross,certified_time=int(np.floor(8/D))+1,max_envelope_violation=err))
 # Zero-drive sign invariance tested before numerical underflow near zero.
 x=-1.
 for t in range(100):
  assert x<0
  x=a*x+g*np.tanh(k*x)
 # Negative drive fixed-point counterexample, computed by iteration.
 x=-1.;D=-.1
 for t in range(10000):x=a*x+g*np.tanh(k*x)+D
 assert x<0 and abs(a*x+g*np.tanh(k*x)+D-x)<1e-12
Path(__file__).with_name('zero_threshold_results.json').write_text(json.dumps(out,indent=2))
print('PASS: six switching/envelope checks; three zero-drive sign checks; three negative fixed-point checks.')
