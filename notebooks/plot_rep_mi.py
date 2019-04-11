import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

tag = 'v30-'

logfiles = sorted(glob.glob(os.path.join('logs',tag+'*','train-*.txt')))

seeds = [f.split('-')[-1].split('.')[0] for f in logfiles]
logs = [open(f,'r').read().splitlines() for f in logfiles]
results = [[json.loads(item) for item in log] for log in logs]
data = [list(zip(*[[item[field] for field in item.keys()] for item in r])) for r in results]
fields = results[0][0].keys()
results = [dict(zip(fields, [np.asarray(v) for v in d]) ) for d in data]

fig, ax = plt.subplots()
for s, result in zip(seeds, results):
    ax.plot(result['step'], result['MI'], label=s)
ax.legend()
ax.set_title('Normalized Mutual Information (by seed)')
ax.set_xlabel('Updates')
plt.show()
