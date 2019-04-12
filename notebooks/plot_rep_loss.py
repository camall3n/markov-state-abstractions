import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

tag = 'exp1-'

logfiles = sorted(glob.glob(os.path.join('logs',tag+'*','train-*.txt')))

seeds = [f.split('-')[-1].split('.')[0] for f in logfiles]
logs = [open(f,'r').read().splitlines() for f in logfiles]

def get_results(log):
    results = [json.loads(item) for item in log]
    fields = results[0].keys()
    data = dict([(f, np.asarray([item[f] for item in results])) for f in fields])
    return data

results = [get_results(log) for log in logs]
total_loss = [(
    1.0 * result['L_inv'] +
    0.1 * result['L_fwd'] +
    1.0 * result['L_cpc'] +
    0.1 * (result['L_fac']-1)
) for result in results]

fig, ax = plt.subplots(figsize=(8,6))
for s, result, loss in zip(seeds, results, total_loss):
    ax.plot(result['step'], loss, color='C0', label=s, alpha=0.3)
ax.set_title('Total Loss (by seed)')
ax.set_xlabel('Updates')
# ax.set_xlim([0,3000])

target_y = 0.25
target_x = 2000

ax.vlines([target_x], ymin=0, ymax=target_y, linestyles='dashed')
ax.hlines([target_y], xmin=0, xmax=target_x, linestyles='dashed')
ax.text(1000,0,'successful runs',ha='center')
plt.show()

n_good = np.sum(np.asarray([l[target_x//100] for l in total_loss]) < target_y)
print(n_good, 'successful runs out of', len(total_loss),'total')
