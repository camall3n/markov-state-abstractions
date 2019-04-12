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
mi = np.asarray([r['MI'] for r in results])
fig, ax = plt.subplots(figsize=(8,6))
for s, result in zip(seeds, results):
    ax.plot(result['step'], result['MI'], color='C0', label=s, alpha=0.3)
ax.set_title('Normalized Mutual Information (by seed)')
ax.set_xlabel('Updates')
# ax.set_xlim([100,3000])
plt.show()
