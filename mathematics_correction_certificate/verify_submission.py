"""Run all numerical checks in a temporary copy and compare archived outputs."""
from pathlib import Path
import csv,json,shutil,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent

def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-10,atol=1e-12),(a,b)
    else:assert a==b,(a,b)

def check_file(a,b):
    if a.suffix=='.json':compare(json.loads(a.read_text()),json.loads(b.read_text()))
    else:
        with a.open() as f:aa=list(csv.reader(f))
        with b.open() as f:bb=list(csv.reader(f))
        assert len(aa)==len(bb)
        for row1,row2 in zip(aa,bb):
            assert len(row1)==len(row2)
            for x,y in zip(row1,row2):
                try:u,v=float(x),float(y)
                except ValueError:assert x==y
                else:assert np.isclose(u,v,rtol=1e-10,atol=1e-12),(a.name,x,y)

with tempfile.TemporaryDirectory() as tmp:
    work=Path(tmp)/'work'
    shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('generated','*.zip','*.aux','*.log','*.out'))
    out=Path(tmp)/'outputs'
    for name in ['verify_certificate.py','verify_zero_threshold.py','verify_heterogeneous.py']:
        subprocess.run([sys.executable,str(work/name)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(work/'reproduction/reproduce.py'),'--output-dir',str(out)],check=True,stdout=subprocess.DEVNULL)
    for name in ['certificate_results.json','switching.csv','boundary.csv','zero_threshold_results.json','heterogeneous_results.json','heterogeneous.csv']:
        check_file(ROOT/name,work/name)
    for a in (ROOT/'reproduction').iterdir():
        if a.suffix in ['.csv','.json']:check_file(a,out/a.name)
    print('PASS: saved-input reproduction, all archived CSV/JSON results, switching/boundary, zero-threshold, and heterogeneous checks.')
