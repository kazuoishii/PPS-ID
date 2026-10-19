"""Verify the switching certificate and its exact boundary counterexample."""
from pathlib import Path
import json
import numpy as np

P = Path(__file__).resolve().parent
a, gamma, kappa, D, n = 0.2, 1.2, 1.0, 0.3, 20
q = np.arccosh(np.sqrt(gamma * kappa / (1 - a))) / kappa
dc = gamma * np.tanh(kappa * q) - (1 - a) * q
complete = np.ones((n, n)) / n
cycle = np.roll(np.eye(n), -1, axis=1)
modular = np.full((n, n), 0.1 / n)
modular[:10, :10] += 0.9 / 10
modular[10:, 10:] += 0.9 / 10
graphs = [complete, cycle, modular]
z, v = np.linspace(-8, 2, n), -8.0
rows = []
for t in range(301):
    loss = np.mean((1 / (1 + np.exp(z))) ** 2)
    bound = (1 / (1 + np.exp(v))) ** 2
    assert z.min() >= v - 1e-12
    assert loss <= bound + 1e-12
    rows.append((t, z.min(), v, loss, bound))
    if t < 300:
        d = D + 0.05 * (1 + np.sin(0.17 * t + 2 * np.pi * np.arange(n) / n))
        z = a * z + gamma * (graphs[t % 3] @ np.tanh(kappa * z)) + d
        v = a * v + gamma * np.tanh(kappa * v) + D
rows = np.array(rows)
np.savetxt(P / 'switching.csv', rows, delimiter=',',
           header='t,network_min,envelope,brier,brier_bound', comments='')
boundary = []
for name, W in zip(['Complete', 'Cycle', 'Modular'], graphs):
    z = np.full(n, -q)
    for t in range(301):
        err = float(np.max(np.abs(z + q)))
        assert np.all(z < 0) and err < 1e-10
        boundary.append((name, t, float(z.min()), err))
        z = a * z + gamma * (W @ np.tanh(kappa * z)) + dc
with (P / 'boundary.csv').open('w') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['graph', 't', 'signed_min', 'deviation_from_negative_fold'])
    writer.writerows(boundary)
result = dict(q=float(q), dc=float(dc),
              network_crossing=int(rows[rows[:, 1] > 0][0, 0]),
              envelope_crossing=int(rows[rows[:, 2] > 0][0, 0]),
              certified_time=int(np.floor(8 / (D - dc))) + 1,
              boundary_max_error=max(x[3] for x in boundary))
(P / 'certificate_results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
