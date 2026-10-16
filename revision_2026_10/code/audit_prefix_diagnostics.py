"""Audit supplied three-update repeated-trial records, without retraining."""
import json
from pathlib import Path
import numpy as np

def audit(root):
    out={}
    for backend in ('cpu','mps'):
        folder=root/'diagnostics'/backend
        a=json.loads((folder/'trial_1.json').read_text())
        b=json.loads((folder/'trial_2.json').read_text())
        supplied=json.loads((folder/'comparison.json').read_text())
        comparison={
            'initial_weights_equal':a['initial_weights']==b['initial_weights'],
            'teacher_weights_equal':a['teacher_weights']==b['teacher_weights'],
            'initial_rng_equal':a['rng_after_initialization']==b['rng_after_initialization'],
            'steps':[]}
        assert len(a['steps'])==len(b['steps'])==3
        for x,y in zip(a['steps'],b['steps']):
            assert x['step']==y['step']
            row={'step':x['step'],
                 'batch_equal':x['row_ids']==y['row_ids'] and x['input_hashes']==y['input_hashes'],
                 'rng_before_equal':x['rng_before_forward']==y['rng_before_forward'],
                 'weights_after_equal':x['weights_after_update']==y['weights_after_update']}
            for source,target in [('student_logits','student_logits_max_abs_difference'),('teacher_probabilities','teacher_probabilities_max_abs_difference'),('classifier_gradient','classifier_gradient_max_abs_difference'),('loss','loss_abs_difference')]:
                row[target]=float(np.max(np.abs(np.asarray(x[source],dtype=np.float64)-np.asarray(y[source],dtype=np.float64))))
            comparison['steps'].append(row)
        assert comparison==supplied['comparison'],backend
        out[backend]={'updates_per_trial':3,'trials':2,'comparison_matches':True,'comparison':comparison,
                      'max_logit_difference':max(v['student_logits_max_abs_difference'] for v in comparison['steps']),
                      'max_loss_difference':max(v['loss_abs_difference'] for v in comparison['steps']),
                      'max_head_gradient_difference':max(v['classifier_gradient_max_abs_difference'] for v in comparison['steps'])}
    return {'all_checks_passed':True,'scope':'saved three-update records only; not a training rerun or verification of the separately reported 100-update diagnostic','backends':out}

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=audit(root)
    (root/'diagnostics/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
